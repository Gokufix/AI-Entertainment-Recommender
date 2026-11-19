# AI Entertainment Recommender

MovieMatch is a Flask application backed by a SQLite ratings database and a
PyTorch hybrid recommendation model.

The model combines matrix factorization, MiniLM text embeddings, genre
similarity, and popularity. It returns 20 recommendations as a JSON-compatible
dictionary for the website API.

See [`Recommend System/README.md`](Recommend%20System/README.md) for the complete
database rebuild, training, testing, artifact packaging, and local run steps.
