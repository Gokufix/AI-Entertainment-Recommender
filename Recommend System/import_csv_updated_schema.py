"""Import TMDB movies and MovieLens ratings into corrected database schema.

Usage:
  python database_corrected.py
  python import_csv_updated_schema.py --create-placeholder-users

WARNING: --create-placeholder-users creates non-login placeholder Users records
for MovieLens user IDs. Do not use on an existing application database where
user IDs may belong to real accounts. Use a NEW, EMPTY system.db instead.
Without that flag, ratings import only if every referenced user already exists;
otherwise the script stops before inserting ratings.
"""
import argparse
import csv
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
MOVIES = ROOT / 'movies_tmdb_ids_partially_completed_no_blanks.csv'
RATINGS = [ROOT / 'ratings_1_tmdb_ids_no_blanks.csv', ROOT / 'ratings_2_tmdb_ids_no_blanks.csv']
BATCH = 50_000

def insert_batch(conn, sql, batch):
    before = conn.total_changes
    conn.executemany(sql, batch)
    return conn.total_changes - before

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, default=ROOT / 'system.db')
    parser.add_argument('--create-placeholder-users', action='store_true')
    args = parser.parse_args()
    if not all(p.is_file() for p in [MOVIES, *RATINGS]):
        parser.error('Cleaned CSV files must be in the same directory as this script')
    with sqlite3.connect(args.db) as conn:
        conn.execute('PRAGMA foreign_keys=ON')
        for name in ('Users', 'Movies', 'Watched'):
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone():
                parser.error(f'{name} table missing: run database_corrected.py first')
        cols = {r[1] for r in conn.execute('PRAGMA table_info(Watched)')}
        if not {'user_id', 'movie_id', 'review', 'watched_at'} <= cols:
            parser.error('Watched table does not match corrected schema. Use a fresh database.')
        if args.create_placeholder_users and conn.execute('SELECT 1 FROM Users LIMIT 1').fetchone():
            parser.error('Refusing placeholder user creation: Users is not empty. Use a new database.')

        with MOVIES.open(newline='', encoding='utf-8-sig') as f:
            batch = []; added = 0
            for r in csv.DictReader(f):
                batch.append((int(r['movieId']), r['title'], r['genres']))
                if len(batch) == BATCH:
                    added += insert_batch(conn, 'INSERT OR IGNORE INTO Movies (movie_id,title,genres) VALUES (?,?,?)', batch)
                    batch.clear()
            if batch:
                added += insert_batch(conn, 'INSERT OR IGNORE INTO Movies (movie_id,title,genres) VALUES (?,?,?)', batch)
        conn.commit()
        print(f'Movies inserted: {added:,}')
        movie_ids = {r[0] for r in conn.execute('SELECT movie_id FROM Movies')}
        if args.create_placeholder_users:
            # Determine MovieLens user IDs without loading all ratings into memory.
            user_ids = set()
            for path in RATINGS:
                with path.open(newline='', encoding='utf-8-sig') as f:
                    for r in csv.DictReader(f):
                        user_ids.add(int(r['userId']))
            # Deliberately non-authenticatable placeholder credentials.
            import secrets
            ids = sorted(user_ids)
            for i in range(0, len(ids), BATCH):
                conn.executemany('INSERT INTO Users (user_id,username,email,password) VALUES (?,?,?,?)',
                    [(uid, f'movielens_import_{uid}', f'movielens_import_{uid}@invalid.example', secrets.token_hex(32)) for uid in ids[i:i+BATCH]])
            conn.commit()
            print(f'Placeholder MovieLens users inserted: {len(ids):,}')
        else:
            # Never silently drop ratings due to missing Users FK rows.
            existing_users = {r[0] for r in conn.execute('SELECT user_id FROM Users')}
            for path in RATINGS:
                with path.open(newline='', encoding='utf-8-sig') as f:
                    for r in csv.DictReader(f):
                        if int(r['userId']) not in existing_users:
                            parser.error('MovieLens user IDs are not present in Users. Use a fresh DB and --create-placeholder-users, or map users explicitly.')

        sql = 'INSERT OR IGNORE INTO Watched (user_id,movie_id,review,watched_at) VALUES (?,?,?,?)'
        for path in RATINGS:
            added = skipped_movie = skipped_duplicate = 0
            batch = []
            with path.open(newline='', encoding='utf-8-sig') as f:
                for r in csv.DictReader(f):
                    movie_id = int(r['movieId'])
                    if movie_id not in movie_ids:
                        skipped_movie += 1
                        continue
                    watched_at = datetime.fromtimestamp(int(r['timestamp']), timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    batch.append((int(r['userId']), movie_id, float(r['rating']), watched_at))
                    if len(batch) == BATCH:
                        n = insert_batch(conn, sql, batch)
                        added += n; skipped_duplicate += len(batch)-n; batch.clear()
                if batch:
                    n = insert_batch(conn, sql, batch)
                    added += n; skipped_duplicate += len(batch)-n
            conn.commit()
            print(f'{path.name}: inserted {added:,}; skipped missing movie {skipped_movie:,}; skipped duplicate/existing {skipped_duplicate:,}')

if __name__ == '__main__':
    main()
