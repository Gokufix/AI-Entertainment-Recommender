"""Integration tests use an isolated database and a fake email transport."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import pytest
from werkzeug.security import generate_password_hash

temp = tempfile.TemporaryDirectory()
db_path = Path(temp.name) / 'test.db'
# Verify migration of the original three-column table too.
with sqlite3.connect(db_path) as db:
    db.execute('CREATE TABLE users(id INTEGER PRIMARY KEY, username TEXT UNIQUE COLLATE NOCASE, password_hash TEXT NOT NULL)')
    db.execute('INSERT INTO users(username,password_hash) VALUES (?,?)', ('legacy@example.test', generate_password_hash('legacy-password')))
db.close()
os.environ['DATABASE_URL'] = 'sqlite:///' + str(db_path)
os.environ['SECRET_KEY'] = 'isolated-automated-test-secret'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
module = importlib.import_module('app')
app = module.app
app.config.update(TESTING=True, SESSION_COOKIE_SECURE=False)
from account_store import engine
from sqlalchemy import text

@pytest.fixture(scope='session', autouse=True)
def close_database():
    yield
    engine.dispose()
    temp.cleanup()

def post(client, path, **data):
    client.get('/login')
    with client.session_transaction() as sess:
        data['form_token'] = sess['form_token']
    return client.post(path, data=data)

@pytest.fixture
def client():
    with engine.begin() as db:
        for table in ['reset_tokens', 'rate_limits', 'search_history']:
            db.execute(text('DELETE FROM '+table))
        db.execute(text("DELETE FROM users WHERE username <> 'legacy@example.test'"))
    return app.test_client()

def signup(client):
    assert post(client, '/signup', username='Tester', email='tester@example.test',
                password='original-password', confirmation='original-password').status_code == 302

@pytest.fixture
def mail(monkeypatch):
    messages = []
    monkeypatch.setenv('PUBLIC_BASE_URL', 'https://moviematch.example.test')
    monkeypatch.setenv('RESEND_API_KEY', 'fake-test-key')
    monkeypatch.setenv('MAIL_FROM', 'MovieMatch <reset@example.test>')
    monkeypatch.setattr(module.mailer, 'send_reset', lambda email, link: messages.append((email, link)))
    return messages

def test_legacy_migration(client):
    with engine.connect() as db:
        row = db.execute(text("SELECT * FROM users WHERE username='legacy@example.test'")).mappings().one()
        assert row['email'] == 'legacy@example.test'
        assert row['session_version'] == 0
    assert post(client, '/login', username='legacy@example.test', password='legacy-password').status_code == 302

def test_auth_and_csrf(client):
    assert client.get('/home').status_code == 302
    assert client.get('/history').status_code == 302
    assert client.get('/settings').status_code == 302
    assert client.post('/signup', data={}).status_code == 400
    assert b'Forgot password?' in client.get('/login').data
    signup(client)
    assert b'already registered' in post(client, '/signup', username='tester', email='other@example.test', password='long-password', confirmation='long-password').data
    assert post(client, '/login', username='TESTER@example.test', password='original-password').status_code == 302
    assert b'Discovery picks' in client.get('/home').data
    assert post(client, '/logout').status_code == 302
    assert client.get('/home').status_code == 302

def test_reset_expiry_reuse_and_session_revocation(client, mail):
    signup(client)
    post(client, '/login', username='Tester', password='original-password')
    other = app.test_client()
    known = post(other, '/forgot-password', email='tester@example.test')
    unknown = post(other, '/forgot-password', email='missing@example.test')
    assert known.data == unknown.data
    assert len(mail) == 1
    route = mail[0][1].split('https://moviematch.example.test')[1]
    token = route.rsplit('/', 1)[1]
    assert token.encode() not in known.data
    with engine.connect() as db:
        assert db.execute(text('SELECT token_hash FROM reset_tokens')).scalar() == hashlib.sha256(token.encode()).hexdigest()
    assert post(other, route, password='new-password', confirmation='different-password').status_code == 200
    response = post(other, route, password='new-password', confirmation='new-password')
    assert response.status_code == 302 and response.location.endswith('/reset-complete')
    assert b'You have reset your password' in other.get(response.location).data
    assert other.get(route).status_code == 400
    assert client.get('/home').status_code == 302  # Old session is revoked.
    assert b'Incorrect' in post(other, '/login', username='Tester', password='original-password').data
    assert post(other, '/login', username='Tester', password='new-password').status_code == 302
    post(other, '/forgot-password', email='tester@example.test')
    expired = mail[-1][1].split('https://moviematch.example.test')[1]
    with engine.begin() as db:
        db.execute(text('UPDATE reset_tokens SET expires_at=1'))
    assert other.get(expired).status_code == 400

def test_missing_provider_and_throttle(client, mail, monkeypatch):
    signup(client)
    monkeypatch.delenv('RESEND_API_KEY')
    assert b'not available yet' in post(client, '/forgot-password', email='tester@example.test').data
    assert not mail
    monkeypatch.setenv('RESEND_API_KEY', 'fake')
    for _ in range(5):
        post(client, '/forgot-password', email='tester@example.test')
    assert len(mail) == 3

def test_provider_failure_revokes_token(client, mail, monkeypatch):
    signup(client)
    def fail(*args):
        raise OSError('provider offline')
    monkeypatch.setattr(module.mailer, 'send_reset', fail)
    assert post(client, '/forgot-password', email='tester@example.test').status_code == 200
    with engine.connect() as db:
        assert db.execute(text('SELECT count(*) FROM reset_tokens')).scalar() == 0

def test_recommendations_and_history(client):
    from recommender import catalog, recommend, find_titles
    assert len(catalog) == 9742
    result = recommend(1)
    assert len(result) == 20 and len({m['id'] for m in result}) == 20
    assert 1 not in {m['id'] for m in result}
    assert 3114 in {m['id'] for m in result}  # Toy Story 2 is related.
    assert find_titles('Spider-Man')[0]['id'] == 5349
    signup(client)
    post(client, '/login', username='Tester', password='original-password')
    assert b'Toy Story 2' in client.get('/home?q=Toy+Story').data
    assert b'Toy Story (1995)' in client.get('/history').data
    assert b'No titles found' in client.get('/home?q=zzzzzznomovie').data
    assert b'<script>' not in client.get('/home?q=%3Cscript%3E').data
    assert b'not connected' in client.get('/history').data

def test_settings_requires_password(client):
    signup(client)
    post(client, '/login', username='Tester', password='original-password')
    assert b'incorrect' in post(client, '/settings', email='new@example.test', password='wrong').data
    assert b'Recovery email saved' in post(client, '/settings', email='new@example.test', password='original-password').data

def test_email_button(monkeypatch):
    import mailer
    monkeypatch.setenv('RESEND_API_KEY', 'fake')
    monkeypatch.setenv('MAIL_FROM', 'reset@example.test')
    payload = []
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self): return b'{"id":"test"}'
    def capture(req, **kwargs):
        payload.append(json.loads(req.data)); return Response()
    monkeypatch.setattr(mailer, 'urlopen', capture)
    mailer.send_reset('tester@example.test','https://example.test/reset-password/test')
    assert 'Press here to reset password' in payload[0]['html']
    assert 'https://example.test/reset-password/test' in payload[0]['text']


def test_static_javascript_mime(client):
    assert client.get('/static/discovery.js').mimetype == 'application/javascript'

def test_brevo_email(monkeypatch):
    import mailer
    monkeypatch.setenv('BREVO_API_KEY', 'fake-test-key')
    monkeypatch.setenv('MAIL_FROM', 'MovieMatch <sender@example.test>')
    captured=[]
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self): return b'{"messageId":"test-brevo"}'
    def capture(req, **kwargs):
        captured.append((req.full_url, json.loads(req.data)))
        return Response()
    monkeypatch.setattr(mailer,'urlopen',capture)
    assert mailer.send_reset('tester@example.test','https://example.test/reset-password/test') == 'test-brevo'
    assert captured[0][0] == 'https://api.brevo.com/v3/smtp/email'
    assert captured[0][1]['sender']['email'] == 'sender@example.test'
    assert 'Press here to reset password' in captured[0][1]['htmlContent']
