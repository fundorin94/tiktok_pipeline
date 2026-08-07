"""One-off helper: exchange a TikTok OAuth authorization code for tokens.
Not part of the pipeline -- run manually once (or when re-authorizing).

Usage:
  venv\\Scripts\\python.exe tools\\tiktok_oauth_exchange.py <authorization_code>

Reads TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET from .env. Saves the result
to data/tiktok_token.json (gitignored). The actual token values are never
printed to the console -- only non-secret confirmation fields are.
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR  # noqa: E402  (import triggers load_dotenv())

REDIRECT_URI = "https://fundorin94.github.io/tiktok-app-pages/callback.html"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
TOKEN_PATH = DATA_DIR / "tiktok_token.json"


def main():
    if len(sys.argv) < 2:
        print("usage: tiktok_oauth_exchange.py <authorization_code>")
        sys.exit(1)
    code = sys.argv[1]

    client_key = os.environ.get("TIKTOK_CLIENT_KEY")
    client_secret = os.environ.get("TIKTOK_CLIENT_SECRET")
    if not client_key or not client_secret:
        print("Set TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET in .env first.")
        sys.exit(1)

    resp = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "client_key": client_key,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": REDIRECT_URI,
        },
        timeout=15,
    )

    data = resp.json()
    if resp.status_code != 200 or "access_token" not in data:
        print(f"Token exchange failed (status {resp.status_code}):")
        print(json.dumps(data, indent=2))
        sys.exit(1)

    data["obtained_at"] = time.time()  # agents/tiktok_client.py uses this to know when to refresh
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print("Success -- token saved locally (values not shown here, treat the file as a secret).")
    print(f"  open_id: {data.get('open_id')}")
    print(f"  scope: {data.get('scope')}")
    print(f"  expires_in: {data.get('expires_in')}s")
    print(f"  saved to: {TOKEN_PATH}")


if __name__ == "__main__":
    main()
