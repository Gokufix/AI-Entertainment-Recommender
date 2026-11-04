# MovieMatch: password recovery and discovery

## What is ready

- Login and signup, including a recovery email and password confirmation.
- Forgot-password form, a branded email button, a new-password form, and a success screen.
- Reset links last 30 minutes, work once, and revoke previous login sessions.
- Silver gray and Tiffany blue Discover page with random For You picks.
- Left/right arrows, touch scrolling, keyboard navigation, and visible scrollbars.
- Title search and 20 similar movies, using 9,742 MovieLens titles.
- Search history and recovery-email settings. Search history is not watch history.

No streaming-service history is connected. Missing artwork is shown as a title card;
it is never substituted with an unrelated movie's poster.

## Run in PyCharm

Open this folder as a project. In PyCharm's **Terminal** (not inside app.py), run:

```sh
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. Register a test account and sign in.
The supplied movie vectors are ready; do not run model training to start the app.
Local SQLite is created automatically. Website accounts use users.db locally; the team's system.db and database.py
remain separate. Existing users.db accounts are migrated in place.

## Set up reset emails without buying a domain

Use Brevo's HTTPS API. Do not put API keys in GitHub, screenshots, or chat.

1. Create an account at https://www.brevo.com/ and verify your email.
2. In Brevo's sender settings, add a sender named **MovieMatch**, using an email
   address you control. Complete the verification email. Ensure transactional
   sending is activated; Brevo may request account review.
3. Open **SMTP & API â†’ API Keys** and create a dedicated API key for MovieMatch.
   This is an API key, not an SMTP password.
4. In Render, open the MovieMatch service â†’ **Environment**. Add these values:

| Variable | Value |
| --- | --- |
| BREVO_API_KEY | Your private Brevo API key |
| MAIL_FROM | MovieMatch <your-verified-email@example.com> |
| PUBLIC_BASE_URL | https://moviematch-group7.onrender.com |
| SECRET_KEY | A long, randomly generated secret; keep the same value between deploys |
| DATABASE_URL | Connection URL for a persistent PostgreSQL database |

For the SECRET_KEY, generate a value locally with:

```sh
python -c "import secrets; print(secrets.token_hex(32))"
```

Copy that output directly into Render, never into a code file or chat.

5. Save the variables and deploy the updated source.
6. Register with an email you can access. Choose **Forgot password?** on the
   login page, request the email, click **Press here to reset password**, and
   enter the same new password twice.
7. Confirm the success message, go home, and log in with the new password.
   The old password and used reset link must stop working.

Brevo may substitute a compliant sender domain when using a personal/free email
address. Deliverability and provider approval are controlled by Brevo.
If email configuration is missing, the website says email reset is unavailable;
it does not display a usable reset link in the browser.

Resend is also supported if a custom domain is added later: set RESEND_API_KEY
instead of BREVO_API_KEY and use a verified Resend MAIL_FROM sender.

Official guidance:
- https://developers.brevo.com/docs/send-a-transactional-email
- https://help.brevo.com/hc/en-us/articles/14925263522578-Comply-with-Gmail-Yahoo-and-Microsoft-s-requirements-for-email-senders

## Persistent accounts on Render

Render's local filesystem is temporary. SQLite accounts and local secrets can
disappear on redeploy or restart. Configure DATABASE_URL and SECRET_KEY before
relying on real accounts or password recovery. A paid plan is not purchased by
this project; choose a database provider/plan yourself.

DATABASE_URL supports PostgreSQL URLs beginning postgres:// or postgresql://.
Schema creation is automatic. Switching to a new database does not copy old
SQLite accounts: back them up and migrate them separately, or recreate demo
accounts intentionally. Never upload an account database to GitHub.

https://render.com/docs/free#local-files-lost-on-redeploy

## Upload and deploy

The current repository app folder was found at **Recommend System** on main.
Keep other group members' project files. Upload the extracted files from this
package into that existing folder, preserving data/, static/, and templates/.
Do not upload the ZIP as the website source. Do not delete the repository folder.

Render settings:
- Root Directory: `Recommend System`
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn --workers 1 --bind 0.0.0.0:$PORT app:app`

The /recommendations address redirects to the signed-in Discover page.
This package removes the unfinished startup reference to an undefined MOVIE_ID;
no placeholder TMDB API key is used.

## How recommendations work

The supplied notebook uses MiniLM text similarity and genre overlap. This web
version uses that item-similarity calculation: 80% clipped cosine similarity and
20% genre Jaccard, with the notebook's 0.08 diversity adjustment. The selected
movie is excluded. The movie vectors were generated using the official float32
ONNX export of all-MiniLM-L6-v2, attention-mask mean pooling, normalization, and
a maximum of 256 tokens. This is an inference-format adaptation of the notebook,
not an ALS model trained for website users.

Metadata consists of titles, genres, and tags from the notebook's seeded sample
of 600 MovieLens users. No plots or third-party watch histories are invented.
For You is a random sample of the illustrated showcase, not personalized yet.
For a title with several matches, choose the intended movie before recommendations.
The catalog ends in 2018; recent films may not be found.

To rebuild vectors locally (optional):

```sh
python -m pip install numpy onnxruntime tokenizers
python prepare_catalog.py path/to/ml-latest-small
```

Keep data/catalog.json and data/embeddings.npy together and in the same order.
The model cache is local build material and must not be uploaded. MovieLens
license/citation is included in data/MOVIELENS_LICENSE.txt; poster sources are
listed in POSTER_SOURCES.md. This is an educational, noncommercial project.

## Validation and limits

Automated tests cover old-account migration, CSRF, signup/login/logout, reset
expiry, reuse, confirmation, throttling, mail failure, session revocation,
settings, movie ranking, missing searches, HTML escaping, and private history.
Run `python -m pip install pytest`, then `python -m pytest tests -q`.

Email transport was tested with a mock provider. Actual inbox delivery requires
your verified email provider and a final live test. PostgreSQL support is
implemented but must be validated against your configured database before launch.
