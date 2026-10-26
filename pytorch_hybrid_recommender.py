"""PyTorch hybrid entertainment recommender for MovieLens Latest Small.
Components
----------
1. PyTorch explicit-feedback matrix factorization.
2. MiniLM semantic embeddings (Sentence Transformers/PyTorch).
3. Genre-profile similarity.
4. Bayesian popularity fallback.

The program evaluates on a chronological 70/15/15 split, then retrains a
deployment model on all ratings and saves everything needed for inference.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingConfig:
    factors: int = 32
    learning_rate: float = 0.005
    weight_decay: float = 1e-5
    batch_size: int = 1024
    epochs: int = 50
    patience: int = 5


def discover_movielens(start: str | Path = ".") -> Path:
    """Find the extracted MovieLens dataset directory."""
    start = Path(start).resolve()
    candidates = [start / "ml-latest", start / "ml-latest-small", start]
    candidates.extend(p.parent for p in start.rglob("ratings.csv"))
    for folder in candidates:
        if all((folder / name).exists() for name in (
            "ratings.csv", "movies.csv", "tags.csv", "links.csv"
        )):
            return folder
    raise FileNotFoundError(
        "Could not find ratings.csv, movies.csv, tags.csv, and links.csv. "
        "Place the extracted ml-latest folder beside this script."
    )


def load_movielens(folder: str | Path) -> dict[str, pd.DataFrame]:
    folder = Path(folder)
    ratings = pd.read_csv(folder / "ratings.csv")
    movies = pd.read_csv(folder / "movies.csv")
    tags = pd.read_csv(folder / "tags.csv")
    links = pd.read_csv(folder / "links.csv")

    required_ratings = {"userId", "movieId", "rating", "timestamp"}
    required_movies = {"movieId", "title", "genres"}
    if not required_ratings.issubset(ratings.columns):
        raise ValueError(f"ratings.csv must contain {sorted(required_ratings)}")
    if not required_movies.issubset(movies.columns):
        raise ValueError(f"movies.csv must contain {sorted(required_movies)}")
    if ratings[list(required_ratings)].isna().any().any():
        raise ValueError("ratings.csv contains missing required values.")
    if not ratings["rating"].between(0.5, 5.0).all():
        raise ValueError("Ratings must be between 0.5 and 5.0.")

    ratings = ratings.drop_duplicates(["userId", "movieId"]).copy()
    ratings["at"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    tags = tags.copy()
    tags["at"] = pd.to_datetime(tags["timestamp"], unit="s", utc=True)
    movies = movies.drop_duplicates("movieId").copy()

    return {"ratings": ratings, "movies": movies, "tags": tags, "links": links}


def chronological_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Global chronological split matching the selected group design."""
    ordered = ratings.sort_values(["at", "userId", "movieId"]).reset_index(drop=True)
    first = int(len(ordered) * 0.70)
    second = int(len(ordered) * 0.85)
    return (
        ordered.iloc[:first].copy(),
        ordered.iloc[first:second].copy(),
        ordered.iloc[second:].copy(),
    )


class MatrixFactorization(nn.Module):
    """Explicit-feedback matrix factorization with learned biases."""

    def __init__(self, users: int, items: int, factors: int, global_mean: float):
        super().__init__()
        self.user_factors = nn.Embedding(users, factors)
        self.item_factors = nn.Embedding(items, factors)
        self.user_bias = nn.Embedding(users, 1)
        self.item_bias = nn.Embedding(items, 1)
        self.register_buffer("global_mean", torch.tensor(float(global_mean), dtype=torch.float32))
        nn.init.normal_(self.user_factors.weight, std=0.08)
        nn.init.normal_(self.item_factors.weight, std=0.08)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        interaction = (self.user_factors(users) * self.item_factors(items)).sum(dim=1)
        return (
            self.global_mean
            + self.user_bias(users).squeeze(1)
            + self.item_bias(items).squeeze(1)
            + interaction
        )


def frame_to_tensors(
    frame: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    usable = frame[
        frame["userId"].isin(user_to_index) & frame["movieId"].isin(movie_to_index)
    ]
    users = torch.tensor(
        usable["userId"].map(user_to_index).to_numpy(), dtype=torch.long
    )
    items = torch.tensor(
        usable["movieId"].map(movie_to_index).to_numpy(), dtype=torch.long
    )
    ratings = torch.tensor(usable["rating"].to_numpy(), dtype=torch.float32)
    return users, items, ratings


@torch.no_grad()
def rating_metrics(
    model: MatrixFactorization,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    users, items, actual = tensors
    if len(actual) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "rows": 0}
    model.eval()
    predicted = model(users.to(device), items.to(device)).clamp(0.5, 5.0).cpu()
    difference = predicted - actual
    return {
        "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
        "mae": float(torch.mean(torch.abs(difference)).item()),
        "rows": int(len(actual)),
    }


def train_matrix_factorization(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
    config: TrainingConfig,
    device: torch.device,
) -> tuple[MatrixFactorization, list[dict[str, float]], int]:
    """Train with validation-based early stopping."""
    set_seed()
    train_tensors = frame_to_tensors(train, user_to_index, movie_to_index)
    validation_tensors = frame_to_tensors(validation, user_to_index, movie_to_index)
    loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    model = MatrixFactorization(
        len(user_to_index), len(movie_to_index), config.factors, train["rating"].mean()
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.MSELoss()
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_rmse = math.inf
    best_epoch = 1
    remaining_patience = config.patience

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_squared_error = 0.0
        total_rows = 0
        for users, items, actual in loader:
            users, items, actual = users.to(device), items.to(device), actual.to(device)
            optimizer.zero_grad()
            predicted = model(users, items)
            loss = loss_function(predicted, actual)
            loss.backward()
            optimizer.step()
            total_squared_error += float(loss.item()) * len(actual)
            total_rows += len(actual)

        train_rmse = math.sqrt(total_squared_error / max(total_rows, 1))
        validation_result = rating_metrics(model, validation_tensors, device)
        validation_rmse = validation_result["rmse"]
        history.append({
            "epoch": epoch,
            "train_rmse": train_rmse,
            "validation_rmse": validation_rmse,
            "validation_mae": validation_result["mae"],
        })
        print(
            f"Epoch {epoch:02d} | train RMSE {train_rmse:.4f} | "
            f"validation RMSE {validation_rmse:.4f}"
        )

        if validation_rmse < best_rmse - 1e-4:
            best_rmse = validation_rmse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            remaining_patience = config.patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model.")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def train_final_model(
    ratings: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
    config: TrainingConfig,
    epochs: int,
    device: torch.device,
) -> MatrixFactorization:
    """Retrain the deployment model on all ratings after evaluation."""
    set_seed()
    tensors = frame_to_tensors(ratings, user_to_index, movie_to_index)
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    model = MatrixFactorization(
        len(user_to_index), len(movie_to_index), config.factors, ratings["rating"].mean()
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        for users, items, actual in loader:
            users, items, actual = users.to(device), items.to(device), actual.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(users, items), actual)
            loss.backward()
            optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Final-model epoch {epoch:02d}/{epochs}")
    return model


class HybridRecommender:
    """Reusable inference engine for the trained hybrid model."""

    def __init__(
        self,
        model: MatrixFactorization,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        embeddings: np.ndarray,
        user_to_index: dict[int, int],
        movie_to_index: dict[int, int],
        device: torch.device,
    ):
        self.model = model.to(device).eval()
        self.movies = movies.reset_index(drop=True).copy()
        self.ratings = ratings.copy()
        self.embeddings = embeddings.astype(np.float32)
        self.user_to_index = user_to_index
        self.movie_to_index = movie_to_index
        self.device = device
        self.catalog_index = {
            int(movie_id): index for index, movie_id in enumerate(self.movies["movieId"])
        }
        self.seen = self.ratings.groupby("userId")["movieId"].apply(set).to_dict()
        self._prepare_genres()
        self._prepare_popularity()

    def _prepare_genres(self) -> None:
        genre_sets = [
            set(str(value).split("|")) - {"", "(no genres listed)"}
            for value in self.movies["genres"]
        ]
        self.genre_encoder = MultiLabelBinarizer()
        self.genre_matrix = self.genre_encoder.fit_transform(genre_sets).astype(np.float32)

    def _prepare_popularity(self) -> None:
        statistics = self.ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        global_mean = float(self.ratings["rating"].mean())
        counts = self.movies["movieId"].map(statistics["count"]).fillna(0).to_numpy()
        means = self.movies["movieId"].map(statistics["mean"]).fillna(global_mean).to_numpy()
        bayesian_rating = (counts * means + 20 * global_mean) / (counts + 20)
        self.popularity = (bayesian_rating / 5.0).astype(np.float32)

    def _matrix_scores(self, user_id: int) -> np.ndarray:
        scores = self.popularity.copy()
        if user_id not in self.user_to_index:
            return scores
        movie_ids = self.movies["movieId"].to_numpy()
        known_positions = [i for i, mid in enumerate(movie_ids) if int(mid) in self.movie_to_index]
        if not known_positions:
            return scores
        item_indices = torch.tensor(
            [self.movie_to_index[int(movie_ids[i])] for i in known_positions],
            dtype=torch.long,
            device=self.device,
        )
        user_indices = torch.full(
            (len(item_indices),), self.user_to_index[user_id], dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            predicted = self.model(user_indices, item_indices).clamp(0.5, 5.0).cpu().numpy() / 5.0
        scores[np.asarray(known_positions)] = predicted
        return scores

    def _profiles(self, user_id: int) -> tuple[np.ndarray, np.ndarray, bool]:
        history = self.ratings[
            (self.ratings["userId"] == user_id) & (self.ratings["rating"] >= 3.5)
        ]
        positions = [
            self.catalog_index[int(mid)] for mid in history["movieId"] if int(mid) in self.catalog_index
        ]
        if not positions:
            return (
                np.zeros(self.embeddings.shape[1], dtype=np.float32),
                np.zeros(self.genre_matrix.shape[1], dtype=np.float32),
                False,
            )
        weights = history.loc[
            history["movieId"].isin([int(self.movies.iloc[p]["movieId"]) for p in positions]),
            "rating",
        ].to_numpy(dtype=np.float32)
        weights = np.maximum(weights - 3.0, 0.5)
        semantic = np.average(self.embeddings[positions], axis=0, weights=weights)
        semantic /= max(float(np.linalg.norm(semantic)), 1e-8)
        genres = np.average(self.genre_matrix[positions], axis=0, weights=weights)
        genres /= max(float(np.linalg.norm(genres)), 1e-8)
        return semantic.astype(np.float32), genres.astype(np.float32), True

    def recommend(self, user_id: int, number: int = 10) -> pd.DataFrame:
        """Return unseen hybrid recommendations for one user."""
        matrix_scores = self._matrix_scores(user_id)
        semantic_profile, genre_profile, has_profile = self._profiles(user_id)
        if has_profile:
            semantic_scores = np.clip(self.embeddings @ semantic_profile, 0.0, 1.0)
            genre_norms = np.linalg.norm(self.genre_matrix, axis=1)
            genre_scores = np.divide(
                self.genre_matrix @ genre_profile,
                genre_norms,
                out=np.zeros(len(self.movies), dtype=np.float32),
                where=genre_norms > 0,
            )
        else:
            semantic_scores = np.zeros(len(self.movies), dtype=np.float32)
            genre_scores = np.zeros(len(self.movies), dtype=np.float32)

        # Fixed, explainable group-model blend.
        hybrid = (
            0.70 * matrix_scores
            + 0.20 * semantic_scores
            + 0.05 * genre_scores
            + 0.05 * self.popularity
        )
        excluded = self.seen.get(user_id, set())
        allowed = ~self.movies["movieId"].isin(excluded).to_numpy()
        ranked = np.flatnonzero(allowed)
        ranked = ranked[np.argsort(-hybrid[ranked], kind="stable")][:number]
        result = self.movies.iloc[ranked][["movieId", "title", "genres"]].copy()
        result["Predicted Rating"] = np.round(matrix_scores[ranked] * 5.0, 2)
        result["Hybrid Score"] = np.round(hybrid[ranked], 4)
        result["Reason"] = "PyTorch MF + MiniLM + genres + popularity"
        return result.reset_index(drop=True)


def build_catalog_embeddings(
    movies: pd.DataFrame,
    tags: pd.DataFrame,
    asof: pd.Timestamp,
    output: Path,
    device: torch.device,
) -> np.ndarray:
    """Encode title, genres, and historically available tags with MiniLM."""
    available_tags = tags[tags["at"] <= asof].copy()
    available_tags["tag"] = available_tags["tag"].fillna("").astype(str).str.strip()
    grouped_tags = available_tags.groupby("movieId")["tag"].apply(
        lambda values: "; ".join(sorted(set(values))[:12])
    )
    catalog = movies.copy()
    catalog["tags"] = catalog["movieId"].map(grouped_tags).fillna("")
    text = (
        "Title: " + catalog["title"].fillna("")
        + ". Genres: " + catalog["genres"].fillna("").str.replace("|", ", ", regex=False)
        + ". Tags: " + catalog["tags"]
    ).tolist()
    encoder = SentenceTransformer(MINILM_MODEL, device=str(device))
    embeddings = encoder.encode(
        text,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(output / "catalog_embeddings.npy", embeddings)
    return embeddings


def save_artifacts(
    output: Path,
    model: MatrixFactorization,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
    config: TrainingConfig,
    metrics: dict,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "number_of_users": len(user_to_index),
            "number_of_items": len(movie_to_index),
            "factors": config.factors,
            "global_mean": float(model.global_mean.item()),
            "user_to_index": user_to_index,
            "movie_to_index": movie_to_index,
        },
        output / "pytorch_hybrid_model.pth",
    )
    movies.to_csv(output / "catalog.csv", index=False)
    ratings.to_csv(output / "ratings_used.csv", index=False)
    (output / "model_information.json").write_text(
        json.dumps({"config": asdict(config), "metrics": metrics}, indent=2),
        encoding="utf-8",
    )


def load_engine(output: str | Path, device: torch.device | None = None) -> HybridRecommender:
    output = Path(output)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(output / "pytorch_hybrid_model.pth", map_location=device, weights_only=False)
    user_to_index = {int(key): int(value) for key, value in checkpoint["user_to_index"].items()}
    movie_to_index = {int(key): int(value) for key, value in checkpoint["movie_to_index"].items()}
    model = MatrixFactorization(
        checkpoint["number_of_users"],
        checkpoint["number_of_items"],
        checkpoint["factors"],
        checkpoint["global_mean"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    movies = pd.read_csv(output / "catalog.csv")
    ratings = pd.read_csv(output / "ratings_used.csv")
    embeddings = np.load(output / "catalog_embeddings.npy")
    return HybridRecommender(
        model, movies, ratings, embeddings, user_to_index, movie_to_index, device
    )


def train_command(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.data) if args.data else discover_movielens(Path.cwd())
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Dataset: {source}")
    print(f"Training device: {device}")
    data = load_movielens(source)
    ratings, movies, tags = data["ratings"], data["movies"], data["tags"]
    users = sorted(int(value) for value in ratings["userId"].unique())
    # Only rated titles receive collaborative embeddings. Unrated catalog
    # titles correctly fall back to content and popularity at inference time.
    movie_ids = sorted(int(value) for value in ratings["movieId"].unique())
    user_to_index = {value: index for index, value in enumerate(users)}
    movie_to_index = {value: index for index, value in enumerate(movie_ids)}
    train, validation, test = chronological_split(ratings)
    config = TrainingConfig(
        factors=args.factors,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )
    evaluation_model, history, best_epoch = train_matrix_factorization(
        train, validation, user_to_index, movie_to_index, config, device
    )
    validation_metrics = rating_metrics(
        evaluation_model, frame_to_tensors(validation, user_to_index, movie_to_index), device
    )
    test_metrics = rating_metrics(
        evaluation_model, frame_to_tensors(test, user_to_index, movie_to_index), device
    )
    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
        "best_epoch": best_epoch,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    print(json.dumps(metrics, indent=2))

    print("Retraining deployment model on all ratings...")
    final_model = train_final_model(
        ratings, user_to_index, movie_to_index, config, best_epoch, device
    )
    print("Encoding the catalog with MiniLM...")
    embeddings = build_catalog_embeddings(
        movies, tags, ratings["at"].max(), output, device
    )
    save_artifacts(
        output, final_model, movies, ratings, user_to_index, movie_to_index, config, metrics
    )
    engine = HybridRecommender(
        final_model, movies, ratings, embeddings, user_to_index, movie_to_index, device
    )
    sample = engine.recommend(args.sample_user, 10)
    sample.to_csv(output / "sample_recommendations.csv", index=False)
    print(f"\nSample recommendations for user {args.sample_user}:")
    print(sample.to_string(index=False))
    print(f"\nSaved trained artifacts in: {output.resolve()}")


def recommend_command(args: argparse.Namespace) -> None:
    engine = load_engine(args.output)
    print(engine.recommend(args.user, args.number).to_string(index=False))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or use the PyTorch hybrid recommender.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Train, evaluate, and save the model.")
    train_parser.add_argument(
    "--data",
    help="Path to the extracted MovieLens folder, such as ml-latest."
)
    train_parser.add_argument("--output", default="pytorch_hybrid_outputs_full")
    train_parser.add_argument("--factors", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=0.005)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_parser.add_argument("--batch-size", type=int, default=1024)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--patience", type=int, default=5)
    train_parser.add_argument("--sample-user", type=int, default=1)
    train_parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    train_parser.set_defaults(function=train_command)

    recommend_parser = subparsers.add_parser("recommend", help="Load the saved model.")
    recommend_parser.add_argument("--output", default="pytorch_hybrid_outputs_full")
    recommend_parser.add_argument("--user", type=int, default=1)
    recommend_parser.add_argument("--number", type=int, default=10)
    recommend_parser.set_defaults(function=recommend_command)
    return parser


if __name__ == "__main__":
    arguments = make_parser().parse_args()
    arguments.function(arguments)
