"""Import MovieLens tags into the TMDB-ID MovieMatch database.

The included tags.csv uses MovieLens movieId values. The Movies and Watched
tables use TMDB IDs, so the full MovieLens ``links.csv`` file is required to
translate IDs before insertion.

Example::

    python import_tags_csv.py --links "C:\\path\\to\\ml-latest\\links.csv"
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_tmdb_mapping(links_path: Path) -> dict[int, int]:
    """Return a MovieLens movieId -> TMDB ID mapping."""
    if not links_path.is_file():
        raise FileNotFoundError(f"MovieLens links.csv not found: {links_path}")

    # tags.csv uses MovieLens IDs, but the new database uses TMDB IDs. Build
    # one lookup table so every tag is translated before database insertion.
    mapping: dict[int, int] = {}
    with links_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"movieId", "tmdbId"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("links.csv must contain movieId and tmdbId columns")

        for row in reader:
            movie_lens_id = (row.get("movieId") or "").strip()
            tmdb_id = (row.get("tmdbId") or "").strip()
            if not movie_lens_id or not tmdb_id:
                continue
            try:
                mapping[int(movie_lens_id)] = int(float(tmdb_id))
            except ValueError:
                continue

    if not mapping:
        raise ValueError(f"No usable MovieLens-to-TMDB mappings found in {links_path}")
    return mapping


def import_tags(
    csv_path: Path,
    links_path: Path,
    db_path: Path,
    batch_size: int = 5000,
) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if batch_size < 1:
        raise ValueError("batch-size must be positive")

    movie_id_map = load_tmdb_mapping(links_path)
    print(f"Loaded {len(movie_id_map):,} MovieLens-to-TMDB ID mappings.")

    connection = sqlite3.connect(str(db_path), timeout=60)
    try:
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA foreign_keys=ON")
        for table in ("Users", "Movies", "Tags"):
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone() is None:
                raise RuntimeError(f"Missing required table: {table}")

        # Loading parent IDs once is much faster than issuing two SELECT
        # queries for each of the more than two million tag rows.
        valid_users = {row[0] for row in connection.execute("SELECT user_id FROM Users")}
        valid_movies = {row[0] for row in connection.execute("SELECT movie_id FROM Movies")}
        insert_sql = """
            INSERT OR IGNORE INTO Tags (user_id, movie_id, tag, added)
            VALUES (?, ?, ?, ?)
        """

        total = inserted = skipped_invalid = 0
        skipped_unmapped = skipped_missing_parent = skipped_duplicate = 0
        batch: list[tuple[int, int, str, str]] = []

        def flush() -> None:
            nonlocal inserted, skipped_duplicate
            if not batch:
                return
            # Write tags in batches to keep the import fast and memory usage
            # predictable. INSERT OR IGNORE safely skips duplicate tags.
            before = connection.total_changes
            with connection:
                connection.executemany(insert_sql, batch)
            changed = connection.total_changes - before
            inserted += changed
            skipped_duplicate += len(batch) - changed
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
                    movie_lens_id = int(row["movieId"])
                    tag = (row.get("tag") or "").strip()
                    if not tag or len(tag) > 200:
                        raise ValueError("Invalid tag")
                    added = datetime.fromtimestamp(
                        int(row["timestamp"]), tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError, OverflowError, OSError):
                    skipped_invalid += 1
                    continue

                tmdb_id = movie_id_map.get(movie_lens_id)
                if tmdb_id is None:
                    skipped_unmapped += 1
                    continue
                if user_id not in valid_users or tmdb_id not in valid_movies:
                    skipped_missing_parent += 1
                    continue

                batch.append((user_id, tmdb_id, tag, added))
                if len(batch) >= batch_size:
                    flush()
                    if total % 100000 < batch_size:
                        print(f"Processed {total:,} rows; inserted {inserted:,}...", flush=True)
            flush()

        print("\nImport complete")
        print(f"CSV rows read:                 {total:,}")
        print(f"Tags inserted:                 {inserted:,}")
        print(f"Skipped (no TMDB mapping):     {skipped_unmapped:,}")
        print(f"Skipped (missing DB parents):  {skipped_missing_parent:,}")
        print(f"Skipped (duplicates):          {skipped_duplicate:,}")
        print(f"Skipped (invalid rows):        {skipped_invalid:,}")
        print(f"Database: {db_path.resolve()}")
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map MovieLens tag IDs to TMDB IDs and import them into system.db"
    )
    parser.add_argument("--csv", type=Path, default=HERE / "tags.csv")
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=HERE / "system.db")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()
    import_tags(args.csv, args.links, args.db, args.batch_size)


if __name__ == "__main__":
    main()
