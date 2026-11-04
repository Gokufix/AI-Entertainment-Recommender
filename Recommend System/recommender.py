"""Title lookup and the notebook's MiniLM item-to-item ranking.

Embeddings are prepared once, not downloaded or trained on each web request.
MovieLens IDs are catalog IDs, never our website's account IDs.
"""
from pathlib import Path
import json
import random
import re
import unicodedata
import numpy as np

DATA = Path(__file__).resolve().parent / 'data'
movies = json.loads((DATA / 'catalog.json').read_text(encoding='utf-8'))
vectors = np.load(DATA / 'embeddings.npy', allow_pickle=False)
if vectors.shape != (len(movies), 384) or not np.isfinite(vectors).all():
    raise ValueError('Movie catalog and MiniLM vectors do not match. Run prepare_catalog.py.')
poster_map = json.loads((DATA / 'posters.json').read_text(encoding='utf-8'))
for movie in movies:
    movie['poster'] = poster_map.get(str(movie['id']))
    movie['genre_label'] = ' / '.join(movie['genres'][:3])
catalog = {m['id']: m for m in movies}
index = {m['id']: i for i, m in enumerate(movies)}
genre_sets = [set(m['genres']) - {'(no genres listed)', ''} for m in movies]

def normalize(title):
    title = re.sub(r'\(\d{4}\)$', '', title).strip().casefold()
    title = re.sub(r', (the|a|an)$', '', title)
    title = unicodedata.normalize('NFKD', title)
    return re.sub(r'[^a-z0-9]+', '', title)

def find_titles(query):
    normalized = normalize(query)
    if not normalized:
        return []
    exact = [m for m in movies if normalize(m['title']) == normalized]
    if exact:
        return exact
    return [m for m in movies if normalized in normalize(m['title'])][:30]

def random_picks():
    # Random showcase only. No claim of connected streaming/watch history.
    illustrated = [m for m in movies if m['poster']]
    return random.sample(illustrated, min(16, len(illustrated)))

def recommend(movie_id, count=20):
    seed = index[movie_id]
    semantic = np.clip(vectors @ vectors[seed], 0, 1)
    genres = np.array([len(g & genre_sets[seed]) / max(1, len(g | genre_sets[seed])) for g in genre_sets])
    scores = .8 * semantic + .2 * genres
    pool = [int(i) for i in np.argsort(-scores, kind='stable') if i != seed][:max(150, count*10)]
    selected = []
    # The same 0.08 diversity adjustment used by the supplied notebook.
    while pool and len(selected) < count:
        if not selected:
            pick = pool[0]
        else:
            similarity = (vectors[pool] @ vectors[selected].T).max(axis=1)
            pick = pool[int(np.argmax(.92*scores[pool] - .08*similarity))]
        selected.append(pick)
        pool.remove(pick)
    return [movies[i] for i in selected]
