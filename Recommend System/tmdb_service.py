"""Small TMDB API client supporting either documented credential type."""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache

import requests


TMDB_READ_TOKEN = os.getenv("TMDB_READ_TOKEN", "").strip()
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

if not TMDB_READ_TOKEN and not TMDB_API_KEY:
    raise RuntimeError(
        "Set TMDB_READ_TOKEN (API read access token) or TMDB_API_KEY (v3 API key) in .env"
    )


def _auth_headers() -> dict[str, str]:
    # TMDB read-access tokens use Bearer authentication.
    if TMDB_READ_TOKEN:
        return {"Authorization": f"Bearer {TMDB_READ_TOKEN}"}
    return {}


def _auth_params(**params) -> dict:
    # Older v3 API keys are sent as a query parameter instead.
    if TMDB_API_KEY and not TMDB_READ_TOKEN:
        params["api_key"] = TMDB_API_KEY
    return params


def _get(url: str, *, params: dict | None = None) -> requests.Response:
    response = requests.get(
        url,
        headers=_auth_headers(),
        params=_auth_params(**(params or {})),
        timeout=10,
    )
    return response


@lru_cache(maxsize=2000)
def get_movie_details(tmdb_id):
    tmdb_id = int(tmdb_id)

    # TMDB uses separate ID namespaces for movies and television. Try both so
    # the website can return an accurate media_type value.
    for media_type in ("movie", "tv"):
        response = _get(
            f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}",
            params={"language": "en-US"},
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
            "release_date": item.get("release_date") or item.get("first_air_date"),
            "rating": item.get("vote_average"),
            "genres": [genre["name"] for genre in item.get("genres", [])],
            "poster_url": (
                f"https://image.tmdb.org/t/p/w500{poster}" if poster else None
            ),
        }

    return None


@lru_cache(maxsize=2000)
def get_watch_providers(tmdb_id, media_type="movie", region="US"):
    if media_type not in ("movie", "tv"):
        raise ValueError("Invalid media type")

    response = _get(
        f"https://api.themoviedb.org/3/{media_type}/{int(tmdb_id)}/watch/providers"
    )
    response.raise_for_status()

    data = response.json()
    country = data.get("results", {}).get(region, {})

    def format_providers(category):
        return [
            {
                "name": provider["provider_name"],
                "logo": (
                    "https://image.tmdb.org/t/p/w92" + provider["logo_path"]
                    if provider.get("logo_path")
                    else None
                ),
            }
            for provider in country.get(category, [])
        ]

    return {
        "streaming": format_providers("flatrate"),
        "free": format_providers("free"),
        "ads": format_providers("ads"),
        "rent": format_providers("rent"),
        "buy": format_providers("buy"),
        "link": country.get("link"),
    }


def search_movies(query):
    response = _get(
        "https://api.themoviedb.org/3/search/movie",
        params={
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": 1,
        },
    )
    response.raise_for_status()
    data = response.json()

    movies = []
    today = date.today()

    for item in data.get("results", []):
        release_date_str = item.get("release_date")
        if not release_date_str:
            continue

        try:
            release_date = date.fromisoformat(release_date_str)
        except ValueError:
            continue

        if release_date > today:
            continue

        poster_path = item.get("poster_path")
        poster_url = (
            f"https://image.tmdb.org/t/p/w342{poster_path}"
            if poster_path
            else "/static/images/default-poster.webp"
        )

        movies.append(
            {
                "movie_id": item["id"],
                "title": item.get("title", "Unknown"),
                "release_date": release_date_str,
                "overview": item.get("overview", ""),
                "poster_url": poster_url,
            }
        )

    return movies
