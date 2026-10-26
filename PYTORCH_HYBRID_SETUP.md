# PyTorch Hybrid Recommender Setup

## 1. Put these files in the project root

- `pytorch_hybrid_recommender.py`
- `requirements_pytorch_hybrid.txt`
- The extracted `ml-latest-small` folder

Expected layout:

```text
AI-Entertainment-Recommender-main/
├── ml-latest-small/
│   ├── links.csv
│   ├── movies.csv
│   ├── ratings.csv
│   └── tags.csv
├── pytorch_hybrid_recommender.py
└── requirements_pytorch_hybrid.txt
```

## 2. Install dependencies

Open the VS Code terminal in the project root and run:

```powershell
python -m pip install -r requirements_pytorch_hybrid.txt
```

## 3. Verify PyTorch

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

CPU mode is sufficient for MovieLens Latest Small.

## 4. Train and evaluate

```powershell
python pytorch_hybrid_recommender.py train --data ml-latest-small
```

The first run downloads the MiniLM model. Training prints the training and
validation RMSE for every epoch and uses early stopping. It then records test
RMSE/MAE, retrains the deployment model on all ratings, generates MiniLM
catalog embeddings, and saves the complete model.

## 5. Expected output folder

```text
pytorch_hybrid_outputs/
├── catalog.csv
├── catalog_embeddings.npy
├── model_information.json
├── pytorch_hybrid_model.pth
├── ratings_used.csv
├── sample_recommendations.csv
└── training_history.csv
```

## 6. Request recommendations without retraining

```powershell
python pytorch_hybrid_recommender.py recommend --user 1 --number 10
```

Change `1` to another MovieLens user ID and change `10` to the desired number
of recommendations.

## Final model description

The system combines:

- 70% PyTorch Matrix Factorization
- 20% MiniLM semantic similarity
- 5% genre similarity
- 5% Bayesian popularity

Movies already rated by the selected user are excluded. Unrated catalog titles
use content and popularity fallback signals.
