"""Run locally once to prepare movie text vectors; Render only loads the result.

Usage: python prepare_catalog.py path/to/ml-latest-small
Build-only packages: numpy, onnxruntime, tokenizers.
Uses the official MiniLM ONNX export with attention-mask mean pooling.
"""
from pathlib import Path
import csv
import json
import shutil
import sys
import urllib.request
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
source = Path(sys.argv[1])
out = HERE / 'data'
out.mkdir(exist_ok=True)
cache = HERE / '.model-cache'
cache.mkdir(exist_ok=True)
base = 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/'
for remote, local in [('tokenizer.json', 'tokenizer.json'), ('onnx/model.onnx', 'model.onnx')]:
    dest = cache / local
    if not dest.exists():
        print('Downloading', local, flush=True)
        urllib.request.urlretrieve(base + remote, dest)

def rows(name):
    with (source / (name + '.csv')).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

ratings = rows('ratings')
users = sorted({int(r['userId']) for r in ratings})
selected = set(np.random.default_rng(42).choice(users, 600, replace=False).tolist())
tags = {}
for row in rows('tags'):
    if int(row['userId']) in selected:
        tags.setdefault(row['movieId'], set()).add(row['tag'].strip())
links = {r['movieId']: r for r in rows('links')}
movies = []
for r in sorted(rows('movies'), key=lambda r: int(r['movieId'])):
    movies.append({'id': int(r['movieId']), 'title': r['title'],
                   'genres': r['genres'].split('|'),
                   'tags': '; '.join(sorted(tags.get(r['movieId'], set()))[:12]),
                   'tmdb': links.get(r['movieId'], {}).get('tmdbId', '')})
texts = ['Title: '+m['title']+'. Genres: '+', '.join(m['genres'])+
         '. Description: . Tags: '+m['tags'] for m in movies]
tokenizer = Tokenizer.from_file(str(cache / 'tokenizer.json'))
tokenizer.enable_truncation(max_length=256)
tokenizer.enable_padding()
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
model = ort.InferenceSession(str(cache / 'model.onnx'), opts, providers=['CPUExecutionProvider'])
vectors = []
for start in range(0, len(texts), 32):
    encoded = tokenizer.encode_batch(texts[start:start+32])
    inputs = {'input_ids': np.array([e.ids for e in encoded], dtype=np.int64),
              'attention_mask': np.array([e.attention_mask for e in encoded], dtype=np.int64),
              'token_type_ids': np.array([e.type_ids for e in encoded], dtype=np.int64)}
    hidden = model.run(None, {i.name: inputs[i.name] for i in model.get_inputs()})[0]
    mask = inputs['attention_mask'][..., None]
    pooled = (hidden * mask).sum(1) / mask.sum(1).clip(1)
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(1e-8)
    vectors.append(pooled.astype(np.float32))
    if start % 640 == 0:
        print('Encoded', start, '/', len(texts), flush=True)
np.save(out / 'embeddings.npy', np.concatenate(vectors))
(out / 'catalog.json').write_text(json.dumps(movies, ensure_ascii=False), encoding='utf-8')
shutil.copy2(source / 'README.txt', out / 'MOVIELENS_LICENSE.txt')
(out / 'MODEL.md').write_text('MiniLM all-MiniLM-L6-v2, official ONNX float32 export.\n'
    '384-dimensional normalized mean-pooled vectors; maximum 256 tokens.\n'
    'Title + genres + tags from 600 sampled users (seed 42); no plots or watch history.\n'
    'Runtime ranking: 80% cosine similarity + 20% genre Jaccard, with notebook diversity reranking.\n'
    'Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0).\n'
    'MovieLens license and citation: MOVIELENS_LICENSE.txt.\n', encoding='utf-8')
print('Prepared', len(movies), 'movies.', flush=True)
