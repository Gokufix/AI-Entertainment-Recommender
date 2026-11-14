import sqlite3

conn = sqlite3.connect("system.db")
cursor = conn.cursor()

# -------------------------
# USERS TABLE
# -------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS Users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(200) UNIQUE NOT NULL,
    email VARCHAR(200) UNIQUE NOT NULL,
    password VARCHAR(200) NOT NULL
);
""")

# -------------------------
# MOVIES TABLE
# -------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS Movies (
    movie_id INTEGER PRIMARY KEY,
    title VARCHAR(100),
    genres VARCHAR(200)
);
""")

# -------------------------
# WATCHED TABLE
# -------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS Watched (
    user_id INTEGER,
    movie_id INTEGER,
    review REAL,
    watched_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (user_id, movie_id),

    FOREIGN KEY (user_id)
        REFERENCES Users(user_id),

    FOREIGN KEY (movie_id)
        REFERENCES Movies(movie_id)
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS Tags (
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    tag VARCHAR(200) NOT NULL,
    added DATETIME DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (user_id, movie_id, tag),

    FOREIGN KEY (user_id) REFERENCES Users(user_id),
    FOREIGN KEY (movie_id) REFERENCES Movies(movie_id)
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS DataChanges (
    change_id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    action TEXT NOT NULL,
    changed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS ModelTrainingState (
    model_name TEXT PRIMARY KEY,
    last_change_id INTEGER NOT NULL DEFAULT 0,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
""")

# Save changes
conn.commit()

import sqlite3

def install_change_tracking(conn):
    conn.execute("PRAGMA foreign_keys = ON")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS DataChanges (
            change_id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT NOT NULL,
            record_id TEXT NOT NULL,
            action TEXT NOT NULL,
            changed_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Use JSON arrays for composite keys, so tag text
    # containing punctuation does not break key parsing.
    definitions = {
        "Users": ("json(NEW.user_id)", "json(OLD.user_id)"),
        "Movies": ("json(NEW.movie_id)", "json(OLD.movie_id)"),
        "Watched": (
            "json_array(NEW.user_id, NEW.movie_id)",
            "json_array(OLD.user_id, OLD.movie_id)"
        ),
        "Tags": (
            "json_array(NEW.user_id, NEW.movie_id, NEW.tag)",
            "json_array(OLD.user_id, OLD.movie_id, OLD.tag)"
        ),
    }

    for table, (new_key, old_key) in definitions.items():
        for action in ("INSERT", "UPDATE", "DELETE"):
            key = old_key if action == "DELETE" else new_key

            conn.execute(f"""
                CREATE TRIGGER IF NOT EXISTS
                    log_{table.lower()}_{action.lower()}
                AFTER {action} ON {table}
                BEGIN
                    INSERT INTO DataChanges
                        (table_name, record_id, action)
                    VALUES (
                        '{table}',
                        {key},
                        '{action}'
                    );
                END;
            """)

    conn.commit()


with sqlite3.connect("system.db") as conn:
    install_change_tracking(conn)