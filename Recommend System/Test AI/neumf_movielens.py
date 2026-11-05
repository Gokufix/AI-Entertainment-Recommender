"""
NeuMF MovieLens 100K — End-to-end example

What this script does:
1. Downloads MovieLens 100K if it is not already present.
2. Loads the ratings.
3. Treats ratings >= 4 as positive implicit feedback.
4. Creates negative samples.
5. Builds a PyTorch Dataset/DataLoader.
6. Defines and trains a NeuMF model.
7. Saves the trained model.
8. Generates top-10 movie recommendations for a selected user.

Install:
    pip install torch pandas numpy

Run:
    python neumf_movielens.py

This is an educational implementation. For a rigorous recommender-system
experiment, use a per-user train/test split and evaluate HR@K/NDCG@K.
"""

import os
import random
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "ml-100k.zip"
EXTRACTED_DIR = DATA_DIR / "ml-100k"

SEED = 42

POSITIVE_RATING_THRESHOLD = 4
NUM_NEGATIVES = 4

MF_DIM = 8
MLP_LAYERS = (64, 32, 16, 8)

BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 10

MODEL_PATH = "neumf_movielens.pt"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Download MovieLens
# ============================================================

def download_movielens():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACTED_DIR.exists():
        print("MovieLens dataset already exists.")
        return

    if not ZIP_PATH.exists():
        print("Downloading MovieLens 100K...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print(f"Downloaded to: {ZIP_PATH}")

    print("Extracting dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print(f"Extracted to: {EXTRACTED_DIR}")


# ============================================================
# Load ratings
# ============================================================

def load_ratings():
    ratings_path = EXTRACTED_DIR / "u.data"

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    print("\nRatings:")
    print(ratings.head())

    print(f"\nNumber of ratings: {len(ratings):,}")
    print(f"Number of users: {ratings['user_id'].nunique():,}")
    print(f"Number of movies: {ratings['movie_id'].nunique():,}")

    return ratings


# ============================================================
# Load movie titles
# ============================================================

def load_movies():
    movies_path = EXTRACTED_DIR / "u.item"

    # MovieLens 100K uses '|' as the delimiter.
    movies = pd.read_csv(
        movies_path,
        sep="|",
        header=None,
        encoding="latin-1",
        usecols=[0, 1],
        names=["movie_id", "title"]
    )

    return movies


# ============================================================
# Encode IDs
# ============================================================

def encode_ids(ratings):
    # Convert user/movie IDs into contiguous integers starting at 0.
    user_ids = sorted(ratings["user_id"].unique())
    movie_ids = sorted(ratings["movie_id"].unique())

    user_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(user_ids)
    }

    movie_to_idx = {
        movie_id: idx
        for idx, movie_id in enumerate(movie_ids)
    }

    idx_to_user = {
        idx: user_id
        for user_id, idx in user_to_idx.items()
    }

    idx_to_movie = {
        idx: movie_id
        for movie_id, idx in movie_to_idx.items()
    }

    ratings = ratings.copy()

    ratings["user"] = ratings["user_id"].map(user_to_idx)
    ratings["item"] = ratings["movie_id"].map(movie_to_idx)

    return (
        ratings,
        user_to_idx,
        movie_to_idx,
        #idx_to_user,
        idx_to_movie
    )


# ============================================================
# Negative sampling
# ============================================================

def create_training_examples(
    ratings,
    num_items,
    num_negatives=4,
    positive_threshold=4
):
    """
    Create implicit-feedback examples.

    A rating >= positive_threshold becomes label 1.

    For each positive interaction, sample `num_negatives` items that
    the user has not positively interacted with.
    """

    positive = ratings[
        ratings["rating"] >= positive_threshold
    ].copy()

    # For every user, keep the set of positively interacted items.
    user_positive_items = (
        positive
        .groupby("user")["item"]
        .apply(set)
        .to_dict()
    )

    users = []
    items = []
    labels = []

    print(
        f"\nPositive interactions: {len(positive):,}"
    )

    print(
        f"Creating {num_negatives} negative samples per positive..."
    )

    for row in positive.itertuples(index=False):

        user = int(row.user)
        item = int(row.item)

        # Positive example.
        users.append(user)
        items.append(item)
        labels.append(1.0)

        # Negative examples.
        positive_items = user_positive_items[user]

        for _ in range(num_negatives):

            negative_item = random.randrange(num_items)

            while negative_item in positive_items:
                negative_item = random.randrange(num_items)

            users.append(user)
            items.append(negative_item)
            labels.append(0.0)

    return (
        np.array(users, dtype=np.int64),
        np.array(items, dtype=np.int64),
        np.array(labels, dtype=np.float32)
    )


# ============================================================
# PyTorch Dataset
# ============================================================

class InteractionDataset(Dataset):

    def __init__(self, users, items, labels):
        self.users = torch.tensor(
            users,
            dtype=torch.long
        )

        self.items = torch.tensor(
            items,
            dtype=torch.long
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.users[index],
            self.items[index],
            self.labels[index]
        )


# ============================================================
# NeuMF Model
# ============================================================

class NeuMF(nn.Module):
    """
    Neural Matrix Factorization.

    GMF branch:
        user_embedding * item_embedding

    MLP branch:
        concatenate user/item embeddings -> MLP

    Final:
        concatenate GMF + MLP -> prediction
    """

    def __init__(
        self,
        num_users,
        num_items,
        mf_dim=8,
        mlp_layers=(64, 32, 16, 8)
    ):
        super().__init__()

        # -------------------------
        # GMF embeddings
        # -------------------------

        self.user_gmf = nn.Embedding(
            num_users,
            mf_dim
        )

        self.item_gmf = nn.Embedding(
            num_items,
            mf_dim
        )

        # -------------------------
        # MLP embeddings
        # -------------------------

        mlp_embedding_dim = mlp_layers[0] // 2

        self.user_mlp = nn.Embedding(
            num_users,
            mlp_embedding_dim
        )

        self.item_mlp = nn.Embedding(
            num_items,
            mlp_embedding_dim
        )

        # -------------------------
        # MLP
        # -------------------------

        layers = []

        for i in range(len(mlp_layers) - 1):

            layers.append(
                nn.Linear(
                    mlp_layers[i],
                    mlp_layers[i + 1]
                )
            )

            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

        # -------------------------
        # Final prediction layer
        # -------------------------

        self.output = nn.Linear(
            mf_dim + mlp_layers[-1],
            1
        )

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.normal_(
            self.user_gmf.weight,
            std=0.01
        )

        nn.init.normal_(
            self.item_gmf.weight,
            std=0.01
        )

        nn.init.normal_(
            self.user_mlp.weight,
            std=0.01
        )

        nn.init.normal_(
            self.item_mlp.weight,
            std=0.01
        )

    def forward(self, user, item):

        # -------------------------
        # GMF branch
        # -------------------------

        user_gmf = self.user_gmf(user)
        item_gmf = self.item_gmf(item)

        gmf = user_gmf * item_gmf

        # -------------------------
        # MLP branch
        # -------------------------

        user_mlp = self.user_mlp(user)
        item_mlp = self.item_mlp(item)

        mlp = torch.cat(
            [user_mlp, item_mlp],
            dim=-1
        )

        mlp = self.mlp(mlp)

        # -------------------------
        # Combine
        # -------------------------

        neu_mf = torch.cat(
            [gmf, mlp],
            dim=-1
        )

        prediction = self.output(neu_mf)

        return prediction.squeeze(-1)


# ============================================================
# Training
# ============================================================

def train_model(
    model,
    loader,
    device,
    epochs=10,
    learning_rate=0.001
):

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    model.to(device)

    print(f"\nTraining on: {device}")

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for users, items, labels in loader:

            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(
                users,
                items
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"- loss: {average_loss:.4f}"
        )

    return model


# ============================================================
# Recommendations
# ============================================================

def recommend_movies(
    model,
    user_id,
    ratings,
    movies,
    user_to_idx,
    idx_to_movie,
    num_items,
    device,
    top_k=10
):
    """
    Recommend movies for an original MovieLens user ID.

    Already positively interacted movies are excluded.
    """

    if user_id not in user_to_idx:
        raise ValueError(
            f"User {user_id} does not exist."
        )

    user_idx = user_to_idx[user_id]

    # Find movies the user already positively interacted with.
    user_positive_items = set(
        ratings[
            (ratings["user"] == user_idx)
            & (ratings["rating"] >= POSITIVE_RATING_THRESHOLD)
        ]["item"].tolist()
    )

    # Score every movie.
    all_items = torch.arange(
        num_items,
        dtype=torch.long,
        device=device
    )

    all_users = torch.full(
        (num_items,),
        user_idx,
        dtype=torch.long,
        device=device
    )

    model.eval()

    with torch.no_grad():

        scores = model(
            all_users,
            all_items
        )

    scores = scores.cpu()

    # Do not recommend movies already positively interacted with.
    for item in user_positive_items:
        scores[item] = -float("inf")

    k = min(top_k, num_items - len(user_positive_items))

    top_scores, top_indices = torch.topk(
        scores,
        k=k
    )

    results = []

    for score, item_idx in zip(
        top_scores.tolist(),
        top_indices.tolist()
    ):

        movie_id = idx_to_movie[item_idx]

        movie_row = movies[
            movies["movie_id"] == movie_id
        ]

        if len(movie_row) > 0:
            title = movie_row.iloc[0]["title"]
        else:
            title = f"Movie {movie_id}"

        results.append(
            {
                "movie_id": movie_id,
                "title": title,
                "score": score
            }
        )

    return pd.DataFrame(results)

# 
# Create Model
#

def create_model():
    set_seed(SEED)

    print("=" * 60)
    print("NeuMF + MovieLens 100K")
    print("=" * 60)

    # -------------------------
    # Choose CPU or GPU
    # -------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # -------------------------
    # Download data
    # -------------------------

    download_movielens()

    # -------------------------
    # Load data
    # -------------------------

    ratings = load_ratings()
    movies = load_movies()

    # -------------------------
    # Encode IDs
    # -------------------------

    (
        ratings,
        user_to_idx,
        movie_to_idx,
        idx_to_movie
    ) = encode_ids(ratings)

    num_users = len(user_to_idx)
    num_items = len(movie_to_idx)

    print("\nEncoded dataset:")
    print(f"Users: {num_users}")
    print(f"Items: {num_items}")

    # -------------------------
    # Create training examples
    # -------------------------

    (
        users,
        items,
        labels
    ) = create_training_examples(
        ratings,
        num_items=num_items,
        num_negatives=NUM_NEGATIVES,
        positive_threshold=POSITIVE_RATING_THRESHOLD
    )

    print(f"Training examples: {len(labels):,}")

    # -------------------------
    # PyTorch Dataset
    # -------------------------

    dataset = InteractionDataset(
        users,
        items,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # -------------------------
    # Create NeuMF
    # -------------------------

    model = NeuMF(
        num_users=num_users,
        num_items=num_items,
        mf_dim=MF_DIM,
        mlp_layers=MLP_LAYERS
    )

    print("\nModel:")
    print(model)

    # -------------------------
    # Train
    # -------------------------

    model = train_model(
        model,
        loader,
        device=device,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # -------------------------
    # Save model
    # -------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
            "mf_dim": MF_DIM,
            "mlp_layers": MLP_LAYERS,
            "user_to_idx": user_to_idx,
            "movie_to_idx": movie_to_idx
        },
        MODEL_PATH
    )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )

    return(
        model,
        ratings,
        movies,
        user_to_idx,
        idx_to_movie,
        num_items,
        device
        )

def load_model():
    ratings = load_ratings()
    movies = load_movies()

    # -------------------------
    # Encode IDs
    # -------------------------

    (
        ratings,
        user_to_idx,
        movie_to_idx,
        idx_to_movie
    ) = encode_ids(ratings)

    num_users = len(user_to_idx)
    num_items = len(movie_to_idx)

    print("\nEncoded dataset:")
    print(f"Users: {num_users}")
    print(f"Items: {num_items}")

    # -------------------------
    # Create training examples
    # -------------------------

    (
        users,
        items,
        labels
    ) = create_training_examples(
        ratings,
        num_items=num_items,
        num_negatives=NUM_NEGATIVES,
        positive_threshold=POSITIVE_RATING_THRESHOLD
    )

    print(f"Training examples: {len(labels):,}")

    # -------------------------
    # PyTorch Dataset
    # -------------------------

    dataset = InteractionDataset(
        users,
        items,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    
    checkpoint = torch.load(
    "neumf_movielens.pt",
    weights_only=False
    )
    model = NeuMF(
    checkpoint["num_users"],
    checkpoint["num_items"],
    )

    return (
                model,
                ratings,
                movies,
                user_to_idx,
                idx_to_movie,
                num_items,
                device
            )
def get_recommendations(model, ratings, movies, user_to_idx, idx_to_movie, num_items, device):
    example_user = int(input("What User are the Recommendations for: "))

    print(
        f"\nTop 10 recommendations for "
        f"MovieLens user {example_user}:"
        )

    recommendations = recommend_movies(
        model=model,
        user_id=example_user,
        ratings=ratings,
        movies=movies,
        user_to_idx=user_to_idx,
        idx_to_movie=idx_to_movie,
        num_items=num_items,
        device=device,
        top_k=10
    )

    print(
        recommendations.to_string(
            index=False
        )
    )

    print("\nDone!")

def train_old_model(model, device, loader):
    ratings = load_ratings()
    movies = load_movies()

    # -------------------------
    # Encode IDs
    # -------------------------

    (
        ratings,
        user_to_idx,
        movie_to_idx,
        idx_to_movie
    ) = encode_ids(ratings)

    num_users = len(user_to_idx)
    num_items = len(movie_to_idx)

    print("\nEncoded dataset:")
    print(f"Users: {num_users}")
    print(f"Items: {num_items}")

    # -------------------------
    # Create training examples
    # -------------------------

    (
        users,
        items,
        labels
    ) = create_training_examples(
        ratings,
        num_items=num_items,
        num_negatives=NUM_NEGATIVES,
        positive_threshold=POSITIVE_RATING_THRESHOLD
    )

    print(f"Training examples: {len(labels):,}")

    # -------------------------
    # PyTorch Dataset
    # -------------------------

    dataset = InteractionDataset(
        users,
        items,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    
    checkpoint = torch.load(
    "neumf_movielens.pt",
    weights_only=False
    )
    model = NeuMF(
    checkpoint["num_users"],
    checkpoint["num_items"],
    )


    model = train_model(
                        model,
                        loader,
                        device=device,
                        epochs=EPOCHS,
                        learning_rate=LEARNING_RATE
                    )
    return (
                model,
                ratings,
                movies,
                user_to_idx,
                idx_to_movie,
                num_items,
                device
            )


# ============================================================
# Main
# ============================================================

def main():

    controller = input("Create New Model Type 1 \n" \
    "Train Model on New Data 2\n"
    "Get Recommendations Type 3 \n" \
    "Exit Tpye 0\n"
    "Entry: ")
    (
        model,
        ratings,
        movies,
        user_to_idx,
        idx_to_movie,
        num_items,
        device
    ) = load_model()
    
    while controller != "0":
        if controller == "1":
            (
                model,
                ratings,
                movies,
                user_to_idx,
                idx_to_movie,
                num_items,
                device
            ) = create_model()
        elif controller == "2":
            (
                model,
                ratings,
                movies,
                user_to_idx,
                idx_to_movie,
                num_items,
                device
            ) = train_old_model()
        elif controller == "3":
                get_recommendations(model, ratings,movies,
                                user_to_idx,
                                idx_to_movie,
                                num_items,
                                device)
        else:
            print("This option is not available")
        controller = input("Create New Model Type 1 \n" \
    "Train Model on New Data 2\n" \
    "Get Recommendations Type 3 \n" \
    "Exit Tpye 0\n" \
    "Entry: ")

     


    # -------------------------
    # Recommend movies
    # -------------------------



if __name__ == "__main__":
    main()
