"""Fast import of synthetic users for a NEW, empty SQLite Users table.

Run database_corrected.py first, then this script, then the ratings importer.
The passwords are non-authenticating placeholders, NOT suitable for login.
"""
import csv
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = HERE / 'system.db'
RATINGS = [HERE / 'ratings_1_tmdb_ids_no_blanks.csv',
           HERE / 'ratings_2_tmdb_ids_no_blanks.csv']
BATCH_SIZE = 10_000


def main():
    if not DB.is_file():
        raise SystemExit('system.db not found. Run database_corrected.py first.')

    ids = set()
    for path in RATINGS:
        if not path.is_file():
            raise SystemExit(f'Missing CSV: {path.name}')
        print(f'Reading {path.name}...', flush=True)
        with path.open(newline='', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            if 'userId' not in (reader.fieldnames or []):
                raise SystemExit(f'Missing userId in {path.name}')
            for row in reader:
                if row['userId'].strip():
                    ids.add(int(row['userId']))

    print(f'Found {len(ids):,} unique users.', flush=True)
    with sqlite3.connect(DB) as conn:
        conn.execute('PRAGMA foreign_keys = ON')
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='Users'").fetchone():
            raise SystemExit('Users table missing. Run database_corrected.py first.')
        if conn.execute('SELECT EXISTS(SELECT 1 FROM Users)').fetchone()[0]:
            raise SystemExit('Users table is not empty; refusing to mix synthetic users with existing accounts.')
        statement = 'INSERT INTO Users (user_id, username, email, password) VALUES (?, ?, ?, ?)'
        batch = []
        inserted = 0
        for uid in sorted(ids):
            batch.append((uid, f'user_{uid}', f'user_{uid}@example.com', 'DISABLED_TEST_ACCOUNT'))
            if len(batch) >= BATCH_SIZE:
                conn.executemany(statement, batch)
                inserted += len(batch)
                batch.clear()
                if inserted % 50_000 == 0:
                    print(f'Inserted {inserted:,} users...', flush=True)
        if batch:
            conn.executemany(statement, batch)
            inserted += len(batch)
    print(f'Done: inserted {inserted:,} users into {DB.name}.')


if __name__ == '__main__':
    main()
