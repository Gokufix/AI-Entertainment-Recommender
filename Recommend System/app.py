"""A small Flask login example. Run this file in PyCharm to start the website."""

from pathlib import Path
from contextlib import closing
import secrets
import sqlite3
import requests

from flask import Flask, abort, redirect, render_template, request, session, url_for, jsonify
from dotenv import load_dotenv

load_dotenv()
from werkzeug.security import check_password_hash, generate_password_hash
from pytorch_hybrid_recommender import load_engine
from tmdb_service import get_movie_details, get_watch_providers, search_movies

recommender = None

def get_recommender():
    global recommender

    if recommender is None:
        recommender = load_engine(
            "pytorch_hybrid_outputs_full"
        )

    return recommender

app = Flask(__name__)
FOLDER = Path(__file__).resolve().parent
DATABASE = FOLDER / "system.db"
API_KEY = "aasd"

# Flask signs its session cookies with this secret. Keep this file private.
# Saving the secret means restarting the app does not log everyone out.
secret_file = FOLDER / ".session_secret"
if not secret_file.exists():
    secret_file.write_text(secrets.token_hex(32), encoding="utf-8")
app.secret_key = secret_file.read_text(encoding="utf-8").strip()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")


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

@app.before_request
def protect_forms():
    """Protect form submissions and JSON API requests."""

    # Generate a token if the session doesn't have one
    if "form_token" not in session:
        session["form_token"] = secrets.token_hex(32)

    if request.method == "POST":
        expected = session.get("form_token", "")

        # Handle JSON requests
        if request.is_json:
            submitted = request.headers.get("X-Form-Token", "")

        # Handle regular HTML forms
        else:
            submitted = request.form.get("form_token", "")

        if not expected or not secrets.compare_digest(
            expected,
            submitted
        ):
            abort(
                400,
                description="Please reload the page and try again."
            )



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
        email = request.form.get("email", "")
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
                        "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                        (username, password, email),
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

        if password == user["password"]:
            # The session remembers who logged in. It never contains a password.
            session.clear()
            session["user_id"] = user["user_id"]
            session["username"] = user["username"]
            return redirect(url_for("discover"))

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


@app.route("/discover")
def discover():
    user_id = session.get("user_id")

    if user_id is None:
        return redirect("/login")

    return render_template("discover.html")

@app.route("/api/recommendations")
def api_recommendations():
    user_id = session.get("user_id")

    if user_id is None:
        return jsonify({"error": "Not logged in"}), 401

    try:
        ai_results = get_recommender().get_recommendations(
            user_id=user_id,
            number=20
        )

        movies = []

        for recommendation in ai_results["recommendations"].values():
            tmdb_id = int(recommendation["movieId"])

            try:
                details = get_movie_details(tmdb_id)

                if details is None:
                    continue

                details["movie_id"] = tmdb_id
                details["predicted_rating"] = recommendation["Predicted Rating"]
                details["hybrid_score"] = recommendation["Hybrid Score"]
                details["reason"] = recommendation["Reason"]

                movies.append(details)
                print(
                    "AI ID:", tmdb_id,
                    "| AI title:", recommendation.get("title"),
                    "| TMDB title:", details.get("title") if details else None,
                    "| TMDB release date:", details.get("release_date") if details else None,
                    "| TMDB media type:", details.get("media_type") if details else None
                    )

            except requests.RequestException:
                app.logger.exception(
                    "TMDB request failed for ID %s",
                    tmdb_id
                )

        return jsonify({"movies": movies})

    except Exception:
        app.logger.exception("Failed to generate recommendations")
        return jsonify({"error": "Could not load recommendations"}), 500


@app.route("/api/watch-providers/<media_type>/<int:tmdb_id>")
def api_watch_providers(media_type, tmdb_id):
    if session.get("user_id") is None:
        return jsonify({"error": "Not logged in"}), 401

    if media_type not in ("movie", "tv"):
        return jsonify({"error": "Invalid media type"}), 400

    try:
        providers = get_watch_providers(
            tmdb_id,
            media_type=media_type,
            region="US"
        )
        return jsonify(providers)

    except requests.RequestException:
        app.logger.exception(
            "Could not fetch watch providers for %s %s",
            media_type,
            tmdb_id
        )
        return jsonify({"error": "Watch providers unavailable"}), 502


@app.route("/rate")
def rate_page():
    if session.get("user_id") is None:
        return redirect("/login")

    return render_template("rate.html")


@app.route("/api/search-movies")
def api_search_movies():
    if session.get("user_id") is None:
        return jsonify({"error": "Not logged in"}), 401

    query = request.args.get("q", "").strip()

    if len(query) < 2:
        return jsonify({"movies": []})

    try:
        movies = search_movies(query)
        return jsonify({"movies": movies})

    except requests.RequestException:
        app.logger.exception("TMDB search failed")
        return jsonify({"error": "Search unavailable"}), 502


@app.route("/api/rate-movie", methods=["POST"])
def api_rate_movie():
    print("RATE ROUTE REACHED", flush=True)
    user_id = session.get("user_id")
    data = request.get_json(silent=True) or {}

    print("Received data:", data)
    print("User ID:", user_id)
    print("Movie ID:", data.get("movie_id"))
    print("Rating:", data.get("rating"))

    if user_id is None:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json(silent=True) or {}

    try:
        movie_id = int(data.get("movie_id"))
        rating = float(data.get("rating"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid rating data"}), 400

    if movie_id <= 0 or not (1 <= rating <= 5):
        return jsonify({"error": "Rating must be between 1 and 5"}), 400

    with sqlite3.connect(DATABASE) as conn:
        conn.execute("""
            INSERT INTO Watched (user_id, movie_id, review)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, movie_id)
            DO UPDATE SET review = ?
        """, (user_id, movie_id, rating, rating))


    return jsonify({"success": True})

    

if __name__ == "__main__":
    # Local development server. Open http://127.0.0.1:5000 in your browser.
    app.run(host="127.0.0.1", port=5000, debug=False)
