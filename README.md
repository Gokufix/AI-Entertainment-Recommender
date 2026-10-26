# MovieLens PyTorch Hybrid Recommendation System

## Project Description

This project is a movie recommendation system trained using the MovieLens Latest Small dataset.

The original group model was converted into a PyTorch implementation. The system uses several recommendation methods together to provide personalized movie recommendations.

## Recommendation Methods

The hybrid model combines:

- PyTorch matrix factorization
- MiniLM semantic similarity
- Genre similarity
- Movie popularity

The final recommendation score uses these weights:

- Matrix factorization: 70%
- MiniLM similarity: 20%
- Genre similarity: 5%
- Popularity: 5%

## Dataset

This project uses the MovieLens Latest Small dataset.

The `ml-latest-small` folder contains:

- `ratings.csv`
- `movies.csv`
- `tags.csv`
- `links.csv`

The ratings data is separated into training, validation, and testing sets.

## PyTorch Training

The matrix-factorization model learns:

- User embeddings
- Movie embeddings
- User rating biases
- Movie rating biases

The model is trained using PyTorch backpropagation. Validation data and early stopping are used to retain the best model.

## Current Model Results

- Validation RMSE: 0.9784
- Validation MAE: 0.7548
- Test RMSE: 1.0722
- Test MAE: 0.8486
- Best epoch: 3

## Installation

Install the required packages:

```powershell
python -m pip install -r requirements_pytorch_hybrid.txt