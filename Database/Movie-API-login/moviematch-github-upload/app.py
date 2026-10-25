"""A small Flask login example. Run this file in PyCharm to start the website."""

from pathlib import Path
from contextlib import closing
import secrets
import sqlite3

from flask import Flask, abort, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


app = Flask(__name__)
FOLDER = Path(__file__).resolve().parent
DATABASE = FOLDER / "users.db"

# Flask signs its session cookies with this secret. Keep this file private.
# Saving the secret means restarting the app does not log everyone out.
secret_file = FOLDER / ".session_secret"
if not secret_file.exists():
    secret_file.write_text(secrets.token_hex(32), encoding="utf-8")
app.secret_key = secret_file.read_text(encoding="utf-8").strip()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

# SQLite comes with Python. Create the table only if it does not exist.
with closing(sqlite3.connect(DATABASE)) as database:
    database.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL
        )
    """)
    database.commit()


@app.before_request
def protect_forms():
    """A hidden form token helps prevent another website submitting our forms."""
    if request.method == "POST":
        expected = session.get("form_token", "")
        submitted = request.form.get("form_token", "")
        if not expected or not secrets.compare_digest(expected.encode(), submitted.encode()):
            abort(400, description="Please reload the page and try again.")

    if "form_token" not in session:
        session["form_token"] = secrets.token_hex(32)


@app.route("/")
def welcome():
    # Someone who is already logged in goes straight to their home page.
    if "user_id" in session:
        return redirect(url_for("home"))
    return render_template("page.html", page="welcome")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    # GET displays the form. POST handles the form after the user submits it.
    error = None
    username = ""

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirmation = request.form.get("confirmation", "")

        # Check these rules on the server, even if the browser checks them too.
        if not 3 <= len(username) <= 100:
            error = "Your username or email must be between 3 and 100 characters."
        elif password != confirmation:
            error = "The passwords do not match. Please enter them again."
        elif not 8 <= len(password) <= 128:
            error = "Your password must be between 8 and 128 characters."
        else:
            try:
                with closing(sqlite3.connect(DATABASE)) as database:
                    # Store a password hash, never the original password.
                    # The ? placeholders keep user input separate from SQL code.
                    database.execute(
                        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                        (username, generate_password_hash(password)),
                    )
                    database.commit()
            except sqlite3.IntegrityError:
                error = "That username or email is already registered."
            else:
                return redirect(url_for("login", created="yes"))

    # The password fields are always left empty when showing the form again.
    return render_template("page.html", page="signup", error=error, username=username)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    username = ""

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        with closing(sqlite3.connect(DATABASE)) as database:
            database.row_factory = sqlite3.Row  # Access columns using their names.
            user = database.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

        if user and len(password) <= 128 and check_password_hash(user["password_hash"], password):
            # The session remembers who logged in. It never contains a password.
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("home"))

        error = "Incorrect username/email or password."

    return render_template(
        "page.html", page="login", error=error, username=username,
        created=request.args.get("created") == "yes",
    )


@app.route("/home")
def home():
    # Protect this page: visitors must log in before they can see it.
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("page.html", page="home")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("welcome"))


if __name__ == "__main__":
    # Local development server. Open http://127.0.0.1:5000 in your browser.
    app.run(host="127.0.0.1", port=5000, debug=False)
