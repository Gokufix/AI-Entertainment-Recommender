"""Send reset links with Brevo (or Resend). Keep secrets in Render, never GitHub."""
import json
import os
from html import escape
from email.utils import parseaddr
from urllib.request import Request, urlopen

def configured():
    return bool((os.environ.get('BREVO_API_KEY') or os.environ.get('RESEND_API_KEY'))
                and os.environ.get('MAIL_FROM') and os.environ.get('PUBLIC_BASE_URL'))

def send_reset(address, link):
    message = {
        'from': os.environ['MAIL_FROM'], 'to': [address],
        'subject': 'Reset your MovieMatch password',
        'text': f'Press here to reset password: {link}\nThis link expires in 30 minutes and works once. If you did not request this, ignore this email.',
        'html': '<div style="font-family:Arial;background:#ededee;padding:32px;color:#28282c">'
                '<h1>Movie<span style="color:#28756e">Match</span></h1><h2>Reset your password</h2>'
                '<p>Use the button below to choose a new password.</p>'
                f'<p><a href="{escape(link, quote=True)}" style="display:inline-block;background:#81d8d0;color:#28282c;padding:16px;text-decoration:none;border-radius:8px">Press here to reset password</a></p>'
                '<p>This link expires in 30 minutes and works once.</p>'
                '<p>If you did not request this, you can ignore this email.</p></div>'}
    if os.environ.get('BREVO_API_KEY'):
        name, sender = parseaddr(os.environ['MAIL_FROM'])
        message = {'sender': {'name': name or 'MovieMatch', 'email': sender},
                   'to': [{'email': address}], 'subject': message['subject'],
                   'htmlContent': message['html'], 'textContent': message['text']}
        endpoint = 'https://api.brevo.com/v3/smtp/email'
        auth = {'api-key': os.environ['BREVO_API_KEY']}
        id_field = 'messageId'
    else:
        endpoint = 'https://api.resend.com/emails'
        auth = {'Authorization': 'Bearer ' + os.environ['RESEND_API_KEY']}
        id_field = 'id'
    req = Request(endpoint, data=json.dumps(message).encode(),
                  headers={**auth, 'Content-Type': 'application/json', 'User-Agent': 'MovieMatch/1.0'})
    with urlopen(req, timeout=15) as response:
        return json.load(response)[id_field]
