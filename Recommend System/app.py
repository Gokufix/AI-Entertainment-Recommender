"""MovieMatch accounts, email password reset, and movie discovery.
Run python app.py in PyCharm. See SETUP.md for Render configuration.
"""
from pathlib import Path
import hashlib
import hmac
import mimetypes
import os
import re
import secrets
import time
from urllib.parse import urlsplit
from flask import Flask, abort, redirect, render_template, request, session, url_for, g
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash
from account_store import engine
import mailer

app = Flask(__name__)
# Windows can register .js as text/plain; browsers reject that with nosniff.
mimetypes.add_type('application/javascript', '.js')
FOLDER = Path(__file__).resolve().parent
# Keep the same SECRET_KEY on Render. Never upload the local secret file.
secret = os.environ.get('SECRET_KEY')
if not secret:
    path = FOLDER / '.session_secret'
    if not path.exists():
        path.write_text(secrets.token_hex(32), encoding='utf-8')
    secret = path.read_text(encoding='utf-8').strip()
app.secret_key = secret
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax',
                  SESSION_COOKIE_SECURE=os.environ.get('PUBLIC_BASE_URL', '').startswith('https://'),
                  MAX_CONTENT_LENGTH=16384)
EMAIL = re.compile(r'^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$')

def valid_email(value):
    return len(value) <= 254 and EMAIL.fullmatch(value) is not None

def limited(category, identity, limit, seconds=3600):
    """Shared database counters; hashed keys avoid storing raw emails/IPs."""
    now = int(time.time())
    key = hmac.new(app.secret_key.encode(), (category + ':' + identity).encode(), hashlib.sha256).hexdigest()
    with engine.begin() as db:
        db.execute(text('DELETE FROM rate_limits WHERE expires_at <= :now'), {'now': now})
        row = db.execute(text('INSERT INTO rate_limits(bucket,hits,expires_at) VALUES (:key,1,:end) ON CONFLICT(bucket) DO UPDATE SET hits=rate_limits.hits+1 RETURNING hits'),
                         {'key': key, 'end': now+seconds}).first()
    return row[0] > limit

@app.before_request
def protect_forms():
    g.user = None
    if session.get('user_id'):
        with engine.connect() as db:
            user = db.execute(text('SELECT * FROM users WHERE id=:id'), {'id': session['user_id']}).mappings().first()
        # Resetting a password revokes every old signed-in session.
        if not user or user['session_version'] != session.get('session_version', 0):
            session.clear()
        else:
            g.user = user
    if request.method == 'POST':
        expected, submitted = session.get('form_token', ''), request.form.get('form_token', '')
        if not expected or not secrets.compare_digest(expected.encode(), submitted.encode()):
            abort(400, description='Please reload the page and try again.')
    session.setdefault('form_token', secrets.token_hex(32))

@app.after_request
def privacy_headers(response):
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    if request.endpoint != 'static':
        response.headers['Cache-Control'] = 'no-store'
    return response

@app.route('/')
def welcome():
    if g.user:
        return redirect(url_for('home'))
    return render_template('page.html', page='welcome')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip().lower()
    if request.method == 'POST':
        password, confirmation = request.form.get('password', ''), request.form.get('confirmation', '')
        if limited('signup', request.remote_addr or '', 15):
            error = 'Too many attempts. Please try again in an hour.'
        elif not 3 <= len(username) <= 100:
            error = 'Choose a username between 3 and 100 characters.'
        elif not valid_email(email):
            error = 'Enter a valid email address for password recovery.'
        elif '@' in username and username.lower() != email:
            error = 'If your username is an email address, use the same address below.'
        elif password != confirmation:
            error = 'The passwords do not match.'
        elif not 8 <= len(password) <= 128:
            error = 'Use a password between 8 and 128 characters.'
        else:
            try:
                with engine.begin() as db:
                    db.execute(text('INSERT INTO users(username,email,password_hash) VALUES (:username,:email,:password)'),
                               {'username': username, 'email': email, 'password': generate_password_hash(password)})
            except IntegrityError:
                error = 'That username or email is already registered.'
            else:
                return redirect(url_for('login', created='yes'))
    return render_template('page.html', page='signup', error=error, username=username, email=email)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    username = request.form.get('username', '').strip()
    if request.method == 'POST':
        password = request.form.get('password', '')
        if limited('login', (request.remote_addr or '') + ':' + username.lower(), 20, 900):
            error = 'Too many attempts. Try again in 15 minutes.'
        else:
            with engine.connect() as db:
                user = db.execute(text('SELECT * FROM users WHERE lower(username)=:name OR lower(email)=:name'), {'name': username.lower()}).mappings().first()
            if user and len(password) <= 128 and check_password_hash(user['password_hash'], password):
                session.clear()
                session.update(user_id=user['id'], username=user['username'], session_version=user['session_version'])
                return redirect(url_for('home'))
            error = 'Incorrect username/email or password.'
    return render_template('page.html', page='login', error=error, username=username,
                           created=request.args.get('created') == 'yes')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    error = None
    sent = False
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        if not valid_email(email):
            error = 'Enter a valid email address.'
        elif not mailer.configured():
            error = 'Password-reset email is not available yet. Please try again after the site administrator finishes email setup.'
        else:
            # Unknown addresses and throttled requests receive the same response.
            sent = True
            ip_limited = limited('reset-ip', request.remote_addr or '', 20)
            email_limited = limited('reset-email', email, 3)
            if not ip_limited and not email_limited:
                with engine.connect() as db:
                    user = db.execute(text('SELECT id FROM users WHERE lower(email)=:email'), {'email': email}).mappings().first()
                if user:
                    token = secrets.token_urlsafe(32)
                    digest = hashlib.sha256(token.encode()).hexdigest()
                    base = os.environ['PUBLIC_BASE_URL'].rstrip('/')
                    parsed = urlsplit(base)
                    if parsed.scheme != 'https' or not parsed.netloc:
                        app.logger.error('PUBLIC_BASE_URL must be an HTTPS origin.')
                    else:
                        with engine.begin() as db:
                            db.execute(text('DELETE FROM reset_tokens WHERE expires_at <= :now'), {'now': int(time.time())})
                            db.execute(text('INSERT INTO reset_tokens(token_hash,user_id,expires_at) VALUES (:hash,:id,:expiry)'),
                                       {'hash': digest, 'id': user['id'], 'expiry': int(time.time())+1800})
                        try:
                            mailer.send_reset(email, base + url_for('reset_password', token=token))
                        except Exception:
                            app.logger.error('Reset email delivery failed; check provider configuration.')
                            with engine.begin() as db:
                                db.execute(text('DELETE FROM reset_tokens WHERE token_hash=:hash'), {'hash': digest})
    return render_template('recovery.html', page='forgot', error=error, sent=sent)

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    digest = hashlib.sha256(token.encode()).hexdigest()
    with engine.connect() as db:
        entry = db.execute(text('SELECT * FROM reset_tokens WHERE token_hash=:hash AND used=0 AND expires_at>:now'), {'hash': digest, 'now': int(time.time())}).mappings().first()
    error = None
    if not entry:
        return render_template('recovery.html', page='invalid'), 400
    if request.method == 'POST':
        password, confirmation = request.form.get('password', ''), request.form.get('confirmation', '')
        if password != confirmation:
            error = 'The passwords do not match.'
        elif not 8 <= len(password) <= 128:
            error = 'Use a password between 8 and 128 characters.'
        else:
            hashed = generate_password_hash(password)
            with engine.begin() as db:
                # Lock the account first; simultaneous reset requests cannot both win.
                db.execute(text('UPDATE users SET session_version=session_version WHERE id=:id'), {'id': entry['user_id']})
                changed = db.execute(text('UPDATE reset_tokens SET used=1 WHERE token_hash=:hash AND used=0 AND expires_at>:now'), {'hash': digest, 'now': int(time.time())})
                if changed.rowcount != 1:
                    return render_template('recovery.html', page='invalid'), 400
                db.execute(text('UPDATE users SET password_hash=:password, session_version=session_version+1 WHERE id=:id'), {'password': hashed, 'id': entry['user_id']})
                db.execute(text('DELETE FROM reset_tokens WHERE user_id=:id'), {'id': entry['user_id']})
            session.clear()
            return redirect(url_for('reset_complete'))
    return render_template('recovery.html', page='reset', error=error)

@app.route('/reset-complete')
def reset_complete():
    return render_template('recovery.html', page='complete')

@app.route('/home')
def home():
    if not g.user:
        return redirect(url_for('login'))
    from recommender import catalog, random_picks, find_titles, recommend
    query = request.args.get('q', '').strip()[:150]
    selected = catalog.get(request.args.get('movie', type=int))
    matches = find_titles(query) if query and not selected else []
    if len(matches) == 1:
        selected = matches[0]
    results = recommend(selected['id']) if selected else []
    if selected:
        with engine.begin() as db:
            db.execute(text('INSERT INTO search_history(user_id,movie_id,searched_at) VALUES (:user,:movie,:now) ON CONFLICT(user_id,movie_id) DO UPDATE SET searched_at=:now'),
                       {'user': g.user['id'], 'movie': selected['id'], 'now': int(time.time())})
    return render_template('discover.html', page='discover', picks=random_picks(), query=query,
                           selected=selected, results=results, matches=matches)

@app.route('/history')
def history():
    if not g.user:
        return redirect(url_for('login'))
    from recommender import catalog
    with engine.connect() as db:
        ids = db.execute(text('SELECT movie_id FROM search_history WHERE user_id=:id ORDER BY searched_at DESC LIMIT 50'), {'id': g.user['id']}).scalars().all()
    return render_template('discover.html', page='history', results=[catalog[i] for i in ids if i in catalog])

@app.route('/recommendations')
def recommendations():
    # Preserve the existing team's recommendations URL.
    return redirect(url_for('home'))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if not g.user:
        return redirect(url_for('login'))
    error = success = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if limited('settings', str(g.user['id']), 10, 900):
            error = 'Too many attempts. Try again in 15 minutes.'
        elif len(password) > 128 or not check_password_hash(g.user['password_hash'], password):
            error = 'Your current password is incorrect.'
        elif not valid_email(email):
            error = 'Enter a valid email address.'
        elif '@' in g.user['username'] and email != g.user['username'].lower():
            error = 'This account uses its original email as the username. Please keep that recovery address.'
        else:
            try:
                with engine.begin() as db:
                    other = db.execute(text('SELECT id FROM users WHERE lower(username)=:email AND id<>:id'), {'email': email, 'id': g.user['id']}).first()
                    if other:
                        raise ValueError('Email already in use')
                    db.execute(text('UPDATE users SET email=:email WHERE id=:id'), {'email': email, 'id': g.user['id']})
                    db.execute(text('DELETE FROM reset_tokens WHERE user_id=:id'), {'id': g.user['id']})
                success = 'Recovery email saved.'
                g.user = dict(g.user, email=email)
            except (IntegrityError, ValueError):
                error = 'That email is already registered.'
    return render_template('recovery.html', page='settings', error=error, success=success)

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('welcome'))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', '5000')), debug=False)
