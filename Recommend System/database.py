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
    show_id INTEGER PRIMARY KEY,
    movielens_id INTEGER UNIQUE,
    title VARCHAR(100),
    genres VARCHAR(200),
    description VARCHAR(200),
    type VARCHAR(5)
);
""")

# -------------------------
# WATCHED TABLE
# -------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS Watched (
    user_id INTEGER,
    show_id INTEGER,
    watched_at DATETIME,
    review VARCHAR(20),

    PRIMARY KEY (user_id, show_id),

    FOREIGN KEY (user_id)
        REFERENCES Users(user_id),

    FOREIGN KEY (show_id)
        REFERENCES Movies(show_id)
);
""")

# Save changes
conn.commit()