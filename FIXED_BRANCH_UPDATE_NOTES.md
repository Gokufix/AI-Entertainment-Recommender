# Fixed-Branch integration updates

This package preserves the database-backed hybrid recommender design and fixes
the integration points required to rebuild it reliably.

## Changes

- Maps MovieLens tag IDs to TMDB IDs during tag import using full
  `ml-latest/links.csv`.
- Creates the baseline database before enabling change-tracking triggers, which
  prevents tens of millions of initial import events from filling
  `DataChanges`.
- Uses one `Users` schema for database creation, signup, and login.
- Hashes new application passwords and handles failed logins safely.
- Inserts newly rated TMDB titles into `Movies` before inserting a rating.
- Supports either `TMDB_READ_TOKEN` or `TMDB_API_KEY`.
- Removes the committed `.env` from this package and provides `.env.example`.
- Adds a `package` model command that creates
  `pytorch_hybrid_outputs_full.zip`.
- Configures the artifact ZIP for Git LFS.
- Documents the full extraction, database rebuild, training, testing,
  packaging, and runtime process.

## Required external input

The full MovieLens `ml-latest/links.csv` is required to import `tags.csv`
correctly. The small test dataset's links file is not sufficient.

## Security action

The TMDB credential previously committed in `.env` should be revoked and
replaced because the repository is public.
