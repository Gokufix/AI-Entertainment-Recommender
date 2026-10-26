# MovieMatch

Flask login app with five real movie posters and Tiffany blue/orange styling.

## Upload this complete app
Open Database/Movie-API-login/moviematch-github-upload on GitHub. Choose Add file > Upload files. Drag all contents of this folder into that page, including static and templates. Do not upload the ZIP or the enclosing folder. Commit to main. Existing matching files are replaced; old unused files can remain.

Keep your working Render settings. Expected root: Database/Movie-API-login/moviematch-github-upload
Build: pip install -r requirements.txt
Start: gunicorn --workers 1 --bind 0.0.0.0:$PORT app:app

## Local run
python -m pip install Flask
python app.py

The app creates users.db and .session_secret locally. They are intentionally excluded from this package. Movie recommendations are not connected yet.
