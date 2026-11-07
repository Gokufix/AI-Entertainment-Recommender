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
    watched_at DATETIME,

    PRIMARY KEY (user_id, movie_id),

    FOREIGN KEY (user_id)
        REFERENCES Users(user_id),

    FOREIGN KEY (movie_id)
        REFERENCES Movies(movie_id)
);
""")

# Save changes
conn.commit()