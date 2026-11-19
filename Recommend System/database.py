"""Create the MovieMatch SQLite schema and optionally enable change tracking.

Initial bulk imports should run before change-tracking triggers are installed.
Otherwise every baseline MovieLens row would be copied into DataChanges and the
database would become unnecessarily large.

Examples
--------
Create the empty schema::

    python database.py

After importing users, movies, ratings, and tags, enable incremental tracking::

    python database.py --install-triggers
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


HERE = Path(__file__).resolve().parent


def create_schema(connection: sqlite3.Connection) -> None:
    """Create all application and model-support tables."""
    # Foreign keys protect the relationships between users, movies, ratings,
    # and tags. SQLite requires this setting for every new connection.
    connection.execute("PRAGMA foreign_keys = ON")

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL COLLATE NOCASE,
            email TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS Movies (
            movie_id INTEGER PRIMARY KEY,
            title TEXT,
            genres TEXT
        );

        CREATE TABLE IF NOT EXISTS Watched (
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            review REAL,
            watched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, movie_id),
            FOREIGN KEY (user_id) REFERENCES Users(user_id),
            FOREIGN KEY (movie_id) REFERENCES Movies(movie_id)
        );

        CREATE TABLE IF NOT EXISTS Tags (
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            added TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, movie_id, tag),
            FOREIGN KEY (user_id) REFERENCES Users(user_id),
            FOREIGN KEY (movie_id) REFERENCES Movies(movie_id)
        );

        CREATE TABLE IF NOT EXISTS DataChanges (
            change_id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT NOT NULL,
            record_id TEXT NOT NULL,
            action TEXT NOT NULL,
            changed_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS ModelTrainingState (
            model_name TEXT PRIMARY KEY,
            last_change_id INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_watched_movie_id
            ON Watched(movie_id);
        CREATE INDEX IF NOT EXISTS idx_tags_movie_id
            ON Tags(movie_id);
        CREATE INDEX IF NOT EXISTS idx_datachanges_change_id
            ON DataChanges(change_id);
        """
    )
    connection.commit()


def install_change_tracking(connection: sqlite3.Connection) -> None:
    """Install triggers used by the model's incremental update command."""
    connection.execute("PRAGMA foreign_keys = ON")

    # DataChanges stores JSON-formatted primary keys. The AI update command
    # reads this table to retrain only records changed since the last model.
    definitions = {
        "Users": ("json(NEW.user_id)", "json(OLD.user_id)"),
        "Movies": ("json(NEW.movie_id)", "json(OLD.movie_id)"),
        "Watched": (
            "json_array(NEW.user_id, NEW.movie_id)",
            "json_array(OLD.user_id, OLD.movie_id)",
        ),
        "Tags": (
            "json_array(NEW.user_id, NEW.movie_id, NEW.tag)",
            "json_array(OLD.user_id, OLD.movie_id, OLD.tag)",
        ),
    }

    for table, (new_key, old_key) in definitions.items():
        for action in ("INSERT", "UPDATE", "DELETE"):
            key = old_key if action == "DELETE" else new_key
            connection.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS
                    log_{table.lower()}_{action.lower()}
                AFTER {action} ON {table}
                BEGIN
                    INSERT INTO DataChanges
                        (table_name, record_id, action)
                    VALUES ('{table}', {key}, '{action}');
                END;
                """
            )

    connection.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the MovieMatch database schema.")
    parser.add_argument("--db", type=Path, default=HERE / "system.db")
    parser.add_argument(
        "--install-triggers",
        action="store_true",
        help="Enable change tracking after the initial bulk import is complete.",
    )
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as connection:
        # On the first run, create only the schema. Install triggers after the
        # large baseline CSV import so those rows are not logged as updates.
        create_schema(connection)
        if args.install_triggers:
            install_change_tracking(connection)

    action = "Schema and change tracking ready" if args.install_triggers else "Schema ready"
    print(f"{action}: {args.db.resolve()}")


if __name__ == "__main__":
    main()
