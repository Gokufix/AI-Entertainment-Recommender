"""Local SQLite works immediately; DATABASE_URL enables persistent Postgres."""
from pathlib import Path
import os
from sqlalchemy import create_engine, text, inspect

FOLDER = Path(__file__).resolve().parent
# Keep website accounts separate from the team's movie/watch-history database.
local_file = FOLDER / 'users.db'
url = os.environ.get('DATABASE_URL') or 'sqlite:///' + str(local_file)
if url.startswith(('postgres://', 'postgresql://')):
    url = 'postgresql+psycopg://' + url.split('://', 1)[1]
engine = create_engine(url, pool_pre_ping=True)

def initialize():
    """Migrate existing accounts without deleting their passwords."""
    identity = 'INTEGER PRIMARY KEY' if engine.dialect.name == 'sqlite' else 'SERIAL PRIMARY KEY'
    with engine.begin() as db:
        db.execute(text(f'CREATE TABLE IF NOT EXISTS users (id {identity}, username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL)'))
        columns = {c['name'] for c in inspect(db).get_columns('users')}
        for name, kind in [('email', 'TEXT'), ('session_version', 'INTEGER NOT NULL DEFAULT 0')]:
            if name not in columns:
                db.execute(text(f'ALTER TABLE users ADD COLUMN {name} {kind}'))
        db.execute(text("UPDATE users SET email=lower(username) WHERE email IS NULL AND username LIKE '%@%.%'"))
        db.execute(text('CREATE UNIQUE INDEX IF NOT EXISTS users_username_lower ON users(lower(username))'))
        db.execute(text('CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower ON users(lower(email))'))
        db.execute(text('CREATE TABLE IF NOT EXISTS reset_tokens (token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), expires_at BIGINT NOT NULL, used INTEGER NOT NULL DEFAULT 0)'))
        db.execute(text('CREATE TABLE IF NOT EXISTS rate_limits (bucket TEXT PRIMARY KEY, hits INTEGER NOT NULL, expires_at BIGINT NOT NULL)'))
        db.execute(text('CREATE TABLE IF NOT EXISTS search_history (user_id INTEGER NOT NULL REFERENCES users(id), movie_id INTEGER NOT NULL, searched_at BIGINT NOT NULL, PRIMARY KEY(user_id,movie_id))'))

initialize()
