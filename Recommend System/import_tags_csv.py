"""Import MovieLens tags.csv into an existing system.db Tags table.

Requires Users, Movies and Tags to exist. Does not create missing users/movies.
Run: python import_tags_csv.py --csv tags.csv --db system.db
"""

import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def import_tags(csv_path: Path, db_path: Path, batch_size: int = 5000) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if batch_size < 1:
        raise ValueError("batch-size must be positive")

    connection = sqlite3.connect(str(db_path), timeout=60)
    try:
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA foreign_keys=ON")
        for table in ("Users", "Movies", "Tags"):
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone() is None:
                raise RuntimeError(f"Missing required table: {table}")

        # Only stage rows with valid parent records; do not disable FK constraints.
        insert_sql = """
            INSERT OR IGNORE INTO Tags (user_id, movie_id, tag, added)
            SELECT ?, ?, ?, ?
            WHERE EXISTS (SELECT 1 FROM Users WHERE user_id = ?)
              AND EXISTS (SELECT 1 FROM Movies WHERE movie_id = ?)
        """
        total = inserted = skipped_invalid = skipped_missing = skipped_duplicate = 0
        batch = []

        def flush():
            nonlocal inserted, skipped_missing, skipped_duplicate
            if not batch:
                return
            with connection:
                # Per-row changes() distinguishes inserted from duplicate/missing.
                for user_id, movie_id, tag, added in batch:
                    cursor = connection.execute(
                        insert_sql, (user_id, movie_id, tag, added, user_id, movie_id)
                    )
                    if cursor.rowcount == 1:
                        inserted += 1
                    else:
                        parents_exist = connection.execute(
                            "SELECT EXISTS(SELECT 1 FROM Users WHERE user_id=?) "
                            "AND EXISTS(SELECT 1 FROM Movies WHERE movie_id=?)",
                            (user_id, movie_id),
                        ).fetchone()[0]
                        if parents_exist:
                            skipped_duplicate += 1
                        else:
                            skipped_missing += 1
            batch.clear()

        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"userId", "movieId", "tag", "timestamp"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"CSV must contain: {', '.join(sorted(required))}")

            for row in reader:
                total += 1
                try:
                    user_id = int(row["userId"])
                    movie_id = int(row["movieId"])
                    tag = row["tag"]
                    if tag is None or not tag.strip() or len(tag) > 200:
                        raise ValueError("Invalid tag")
                    added = datetime.fromtimestamp(
                        int(row["timestamp"]), tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError, OverflowError, OSError):
                    skipped_invalid += 1
                    continue
                batch.append((user_id, movie_id, tag, added))
                if len(batch) >= batch_size:
                    flush()
                    if total % 100000 < batch_size:
                        print(f"Processed {total:,} rows; inserted {inserted:,}...", flush=True)
            flush()

        print("\nImport complete")
        print(f"CSV rows read:             {total:,}")
        print(f"Tags inserted:             {inserted:,}")
        print(f"Skipped (missing parents): {skipped_missing:,}")
        print(f"Skipped (duplicates):      {skipped_duplicate:,}")
        print(f"Skipped (invalid rows):    {skipped_invalid:,}")
        print(f"Database: {db_path.resolve()}")
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description="Import MovieLens tags.csv into system.db")
    parser.add_argument("--csv", type=Path, default=Path("tags.csv"))
    parser.add_argument("--db", type=Path, default=Path("system.db"))
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()
    import_tags(args.csv, args.db, args.batch_size)


if __name__ == "__main__":
    main()
