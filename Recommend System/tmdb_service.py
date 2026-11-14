import requests
import os
from functools import lru_cache
from datetime import date
TMDB_TOKEN = os.environ["TMDB_READ_TOKEN"]

@lru_cache(maxsize=2000)
def get_movie_details(tmdb_id):
    tmdb_id = int(tmdb_id)

    headers = {
        "Authorization": f"Bearer {TMDB_TOKEN}"
    }

    for media_type in ("movie", "tv"):
        response = requests.get(
            f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}",
            headers=headers,
            params={"language": "en-US"},
            timeout=10
        )

        if response.status_code == 404:
            continue

        response.raise_for_status()

        item = response.json()
        poster = item.get("poster_path")

        return {
            "movie_id": tmdb_id,
            "media_type": media_type,
            "title": item.get("title") or item.get("name"),
            "overview": item.get("overview"),
            "release_date": (
                item.get("release_date")
                or item.get("first_air_date")
            ),
            "rating": item.get("vote_average"),
            "genres": [
                genre["name"]
                for genre in item.get("genres", [])
            ],
            "poster_url": (
                f"https://image.tmdb.org/t/p/w500{poster}"
                if poster else None
            )
        }

    return None

@lru_cache(maxsize=2000)
def get_watch_providers(tmdb_id, media_type="movie", region="US"):
    if media_type not in ("movie", "tv"):
        raise ValueError("Invalid media type")

    response = requests.get(
        f"https://api.themoviedb.org/3/{media_type}/{int(tmdb_id)}/watch/providers",
        headers={
            "Authorization": f"Bearer {TMDB_TOKEN}"
        },
        timeout=10
    )

    response.raise_for_status()

    data = response.json()
    country = data.get("results", {}).get(region, {})

    def format_providers(category):
        return [
            {
                "name": provider["provider_name"],
                "logo": (
                    "https://image.tmdb.org/t/p/w92"
                    + provider["logo_path"]
                    if provider.get("logo_path")
                    else None
                )
            }
            for provider in country.get(category, [])
        ]

    return {
        "streaming": format_providers("flatrate"),
        "free": format_providers("free"),
        "ads": format_providers("ads"),
        "rent": format_providers("rent"),
        "buy": format_providers("buy"),
        "link": country.get("link")
    }

def search_movies(query):
    response = requests.get(
        "https://api.themoviedb.org/3/search/movie",
        headers={
            "Authorization": f"Bearer {TMDB_TOKEN}"
        },
        params={
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": 1
        },
        timeout=10
    )

    response.raise_for_status()
    data = response.json()

    movies = []
    today = date.today()

    for item in data.get("results", []):
        release_date_str = item.get("release_date")

        # Skip movies with missing or invalid release dates
        if not release_date_str:
            continue

        try:
            release_date = date.fromisoformat(release_date_str)
        except ValueError:
            continue

        # Skip movies that have not been released yet
        if release_date > today:
            continue

        poster_path = item.get("poster_path")

        # Use a default poster if TMDB has no image
        if poster_path:
            poster_url = (
                f"https://image.tmdb.org/t/p/w342{poster_path}"
            )
        else:
            poster_url = "/static/images/default-poster.png"

        movies.append({
            "movie_id": item["id"],
            "title": item.get("title", "Unknown"),
            "release_date": release_date_str,
            "overview": item.get("overview", ""),
            "poster_url": poster_url
        })

    return movies

