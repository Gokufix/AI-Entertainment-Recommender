# MovieMatch recommendation system

## Live website

[Open MovieMatch](https://moviematch-group7.onrender.com/)

## Important data conventions

- `movies_tmdb_ids_partially_completed_no_blanks.csv` uses TMDB IDs in `movieId`.
- Both extracted ratings CSV files use TMDB IDs in `movieId`.
- `tags.csv` still uses original MovieLens IDs. The full MovieLens
  `ml-latest/links.csv` file is therefore required when importing tags.
- Never commit `.env` or `system.db`.

## 1. Extract the ratings files

Install 7-Zip or WinRAR. Keep these two multipart files together:

- `ratings_1_tmdb_ids_no_blanks.part1.rar`
- `ratings_1_tmdb_ids_no_blanks.part2.rar`

Extract **part1 only**. The extractor will read part2 automatically and produce:

```text
ratings_1_tmdb_ids_no_blanks.csv
```

Then extract:

```text
ratings_2_tmdb_ids_no_blanks.rar
```

The final directory must contain both CSV files beside the Python scripts.

## 2. Install Python dependencies

From the `Recommend System` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Rebuild the database

Use a new database. Initial data is imported before change-tracking triggers are
enabled so the baseline import does not create tens of millions of unnecessary
`DataChanges` records.

```powershell
python database.py
python insert_users_sqlite3.py
python import_csv_updated_schema.py
python import_tags_csv.py --links "C:\path\to\ml-latest\links.csv"
python database.py --install-triggers
```

The required execution order is:

1. Create tables with `database.py`.
2. Insert placeholder MovieLens users.
3. Import TMDB-mapped movies and ratings.
4. Map MovieLens tag IDs through `links.csv` and import tags.
5. Enable change tracking for future application activity.

Verify the database:

```powershell
python -c "import sqlite3; c=sqlite3.connect('system.db'); print({t:c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in ('Users','Movies','Watched','Tags','DataChanges')})"
```

`DataChanges` should initially be `0` because it tracks changes made after the
baseline model data is loaded.

## 4. Train the database-backed hybrid model

The full database is large. GPU training is strongly recommended.

```powershell
python pytorch_hybrid_recommender.py train --data system.db --output pytorch_hybrid_outputs_full --sample-user 1
```

The model combines:

- PyTorch matrix factorization
- MiniLM semantic embeddings
- Genre similarity
- Bayesian popularity

## 5. Test the saved model

```powershell
python pytorch_hybrid_recommender.py recommend --output pytorch_hybrid_outputs_full --user 1 --number 20
```

The accessor returns a JSON-compatible dictionary. The command-line function
prints that returned dictionary only for testing.

## 6. Package the artifacts

Create one ZIP after training or after an incremental update:

```powershell
python pytorch_hybrid_recommender.py package --output pytorch_hybrid_outputs_full
```

This creates:

```text
pytorch_hybrid_outputs_full.zip
```

The ZIP is configured for Git LFS in the repository `.gitattributes` because it
will exceed GitHub's normal file-size limit.

At runtime, extract the ZIP so this directory exists beside `app.py`:

```text
Recommend System/pytorch_hybrid_outputs_full/
```

## 7. Configure TMDB without committing secrets

Copy the repository's `.env.example` to `.env` and enter either:

- `TMDB_READ_TOKEN`, or
- `TMDB_API_KEY`

The application supports both credential types. `.env` is intentionally ignored
by Git.

## 8. Run the website

```powershell
python app.py
```

Open `http://127.0.0.1:5000`.

## Incremental model updates

After users add or change ratings/tags and the initial model exists:

```powershell
python pytorch_hybrid_recommender.py update --data system.db --output pytorch_hybrid_outputs_full
python pytorch_hybrid_recommender.py package --output pytorch_hybrid_outputs_full
```
