"""Minimal local web app for the TikTok app-review screencast.

TikTok's review asks to see the integration end to end: the user signing in
with TikTok, granting scopes, and content reaching the account. The pipeline
itself is a CLI, so this page exists purely to make that flow visible and
recordable. It uses the same token file the pipeline uses, so a login here
also authorises the pipeline.

Run:  venv/Scripts/python.exe tools/tiktok_demo_app.py
Then open http://localhost:8722/
"""
import html
import json
import os
import secrets
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from agents import tiktok_client  # noqa: E402
from config import CASES_DIR  # noqa: E402

PORT = 8722
# TikTok only accepts HTTPS redirect URIs on a real domain -- localhost is
# rejected outright. The GitHub Pages callback receives the code, shows it,
# and the operator pastes it back here; the token exchange still happens
# locally, so the client secret never leaves this machine.
REDIRECT_URI = "https://fundorin94.github.io/tiktok-app-pages/callback.html"
AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
# video.upload = send to drafts; the owner posts from the TikTok app.
SCOPES = "user.info.basic,video.upload"

CLIENT_KEY = os.environ.get("TIKTOK_CLIENT_KEY", "")
CLIENT_SECRET = os.environ.get("TIKTOK_CLIENT_SECRET", "")

_state = {"csrf": "", "creator": None, "last_result": ""}


def _videos() -> list:
    return sorted(CASES_DIR.glob("*/video/part*.mp4"))


def _granted_scopes() -> set:
    """Scopes the stored token actually carries. Refreshing a token never
    widens them, so a token issued before the switch to drafts still lacks
    video.upload and every upload fails with a bare "did not authorize the
    scope" error -- worth catching before the request instead."""
    try:
        data = json.loads(tiktok_client.TOKEN_PATH.read_text(encoding="utf-8"))
    except Exception:
        return set()
    return {s.strip() for s in (data.get("scope") or "").split(",") if s.strip()}


def _page(body: str) -> bytes:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>True Crime Pipeline -- TikTok upload</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto;
        padding: 0 20px; line-height: 1.55; color: #111; }}
 .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 20px; margin: 18px 0; }}
 .btn {{ display: inline-block; background: #fe2c55; color: #fff; padding: 11px 20px;
        border: 0; border-radius: 8px; font-size: 15px; cursor: pointer; text-decoration: none; }}
 .muted {{ color: #666; font-size: 14px; }}
 .ok {{ color: #0a7d28; }} .err {{ color: #b3261e; }}
 select {{ padding: 8px; font-size: 15px; width: 100%; }}
</style></head><body>
<h1>True Crime Pipeline</h1>
<p class="muted">Uploads pipeline-generated documentary videos to the owner's own
TikTok account as drafts. Nothing is published automatically: every video is
reviewed and posted by the account owner inside the TikTok app.</p>
<p class="muted">This tool runs locally on the developer's own machine. TikTok
redirects to the callback page at
<a href="https://fundorin94.github.io/tiktok-app-pages/">fundorin94.github.io/tiktok-app-pages</a>,
which also hosts the privacy policy and terms of service.</p>
{body}
</body></html>""".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the console clean for recording
        pass

    def _send(self, payload: bytes, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

        if path == "/":
            self._send(_page(self._home()))
        elif path == "/disconnect":
            tiktok_client.TOKEN_PATH.unlink(missing_ok=True)
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
        elif path == "/login":
            _state["csrf"] = secrets.token_urlsafe(16)
            params = {
                "client_key": CLIENT_KEY,
                "scope": SCOPES,
                "response_type": "code",
                "redirect_uri": REDIRECT_URI,
                "state": _state["csrf"],
            }
            self.send_response(302)
            self.send_header("Location", f"{AUTH_URL}?{urllib.parse.urlencode(params)}")
            self.end_headers()
        else:
            self._send(_page("<p>Not found.</p>"), 404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        form = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))

        if path == "/code":
            self._exchange_code(form.get("code", [""])[0].strip())
            return
        if path != "/upload":
            self._send(_page("<p>Not found.</p>"), 404)
            return
        # A pasted path wins over the dropdown, so any rendered file can be
        # sent without it having to match the pipeline's naming.
        typed = form.get("video_path", [""])[0].strip().strip('"')
        video = Path(typed or form.get("video", [""])[0])
        if not video.is_file():
            _state["last_result"] = (
                f'<p class="err">Not a file: {html.escape(str(video))}</p>')
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return
        try:
            token = tiktok_client.get_valid_access_token(CLIENT_KEY, CLIENT_SECRET)
            publish_id = tiktok_client.upload_to_inbox(token, video)
            _state["last_result"] = (
                f'<p class="ok"><b>Sent to your TikTok account.</b> publish_id: '
                f"{html.escape(publish_id)}<br>Open TikTok and go to <b>Inbox</b> "
                "(the notifications tab) &mdash; tap the notification about the uploaded "
                "video to add a caption and post it.</p>"
            )
        except Exception as exc:  # surface the real reason on screen
            _state["last_result"] = f'<p class="err">Upload failed: {html.escape(str(exc))}</p>'
        self.send_response(302)
        self.send_header("Location", "/")
        self.end_headers()

    def _exchange_code(self, code: str):
        if not code:
            _state["last_result"] = '<p class="err">No code entered.</p>'
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return
        resp = requests.post(
            tiktok_client.TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "client_key": CLIENT_KEY,
                "client_secret": CLIENT_SECRET,
                "code": urllib.parse.unquote(code),
                "grant_type": "authorization_code",
                "redirect_uri": REDIRECT_URI,
            },
            timeout=20,
        )
        data = resp.json()
        if "access_token" not in data:
            _state["last_result"] = (
                f'<p class="err">Token exchange failed: {html.escape(json.dumps(data))}</p>')
        else:
            tiktok_client._save_token(data)
        self.send_response(302)
        self.send_header("Location", "/")
        self.end_headers()

    def _home(self) -> str:
        if not tiktok_client.TOKEN_PATH.exists():
            result = _state.pop("last_result", "") or ""
            return (
                '<div class="card"><h2>1. Connect your TikTok account</h2>'
                "<p>Sign in with TikTok and grant access so the pipeline can place "
                "finished videos in your drafts.</p>"
                '<p><a class="btn" href="/login" target="_blank">Continue with TikTok</a></p>'
                f'<p class="muted">Scopes requested: {SCOPES}</p></div>'
                '<div class="card"><h2>2. Paste the authorization code</h2>'
                "<p>TikTok sends the code to this project's callback page. Copy it from "
                "there and paste it here to finish connecting.</p>"
                '<form method="post" action="/code">'
                '<input name="code" placeholder="authorization code" '
                'style="width:100%;padding:9px;font-size:15px">'
                '<p><button class="btn" type="submit">Connect</button></p></form>'
                f"{result}</div>"
            )

        missing = {"video.upload"} - _granted_scopes()
        if missing:
            return (
                '<div class="card"><h2>Reconnect required</h2>'
                f'<p class="err">The saved authorization is missing '
                f'<b>{html.escape(", ".join(sorted(missing)))}</b>, so uploads to your drafts '
                "will be refused. It was granted before this tool switched from posting "
                "directly to uploading drafts, and refreshing a token never adds new "
                "permissions.</p>"
                '<p><a class="btn" href="/disconnect">Disconnect and sign in again</a></p>'
                f'<p class="muted">Currently granted: {html.escape(", ".join(sorted(_granted_scopes())) or "none")}</p></div>'
            )

        who = ""
        try:
            token = tiktok_client.get_valid_access_token(CLIENT_KEY, CLIENT_SECRET)
            # user.info.basic -- confirm which account the upload will go to.
            user = tiktok_client.get_user_info(token)
            avatar = user.get("avatar_url", "")
            img = (f'<img src="{html.escape(avatar)}" alt="" width="48" height="48" '
                   'style="border-radius:50%;vertical-align:middle;margin-right:10px">') if avatar else ""
            who = (f'<p>{img}Uploads will go to <b>'
                   f'{html.escape(str(user.get("display_name", "")))}</b></p>')
            # Content Posting API -- the account's own posting limits.
            info = tiktok_client.get_creator_info(token)
            who += (f'<p class="muted">Maximum video length for this account: '
                    f'{info.get("max_video_post_duration_sec", "?")}s</p>')
        except Exception as exc:
            who = f'<p class="err">Could not read account info: {html.escape(str(exc))}</p>'

        options = "".join(
            f'<option value="{html.escape(str(v))}">{html.escape(v.parent.parent.name)} / {html.escape(v.name)}</option>'
            for v in _videos()
        ) or '<option value="">no rendered videos found</option>'

        result = _state.pop("last_result", "") or ""
        return (
            f'<div class="card"><h2>Connected</h2>{who}'
            f'<p class="muted">Granted: {html.escape(", ".join(sorted(_granted_scopes())))} '
            '&middot; <a href="/disconnect">disconnect</a></p></div>'
            f'<div class="card"><h2>2. Send a video to your drafts</h2>'
            '<form method="post" action="/upload">'
            f"<select name=\"video\">{options}</select>"
            '<p class="muted">or paste the full path to any other mp4:</p>'
            '<input name="video_path" placeholder="C:\\path\\to\\video.mp4" '
            'style="width:100%;padding:8px;font-size:14px">'
            '<p><button class="btn" type="submit">Send to TikTok drafts</button></p></form>'
            '<p class="muted">The video appears in your TikTok drafts. You add the caption, '
            "choose the audience and publish it yourself in the TikTok app.</p>"
            f"{result}</div>"
        )


def main():
    if not CLIENT_KEY or not CLIENT_SECRET:
        raise SystemExit("Set TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET in your environment first.")
    server = HTTPServer(("localhost", PORT), Handler)
    print(f"Demo app running at http://localhost:{PORT}/  (Ctrl+C to stop)")
    print(f"Add this exact redirect URI in the TikTok developer portal: {REDIRECT_URI}")
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}/")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
