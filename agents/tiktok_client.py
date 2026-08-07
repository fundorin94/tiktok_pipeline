import json
import time
from pathlib import Path

import requests

from config import DATA_DIR

TOKEN_PATH = DATA_DIR / "tiktok_token.json"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
INIT_URL = "https://open.tiktokapis.com/v2/post/publish/video/init/"
# Draft upload: the video lands in the account's TikTok inbox and the owner
# finishes posting inside the TikTok app. Needs only the video.upload scope,
# and none of the Direct Post consent UI, which is why app review for this
# path is far lighter than for publishing straight to the feed.
INBOX_INIT_URL = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
STATUS_URL = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
CREATOR_INFO_URL = "https://open.tiktokapis.com/v2/post/publish/creator_info/query/"
USER_INFO_URL = "https://open.tiktokapis.com/v2/user/info/"

REFRESH_BUFFER_SECONDS = 300  # refresh a bit before actual expiry


def _load_token() -> dict:
    if not TOKEN_PATH.exists():
        raise RuntimeError(
            f"No TikTok token found at {TOKEN_PATH} -- run tools/tiktok_oauth_exchange.py first"
        )
    return json.loads(TOKEN_PATH.read_text(encoding="utf-8"))


def _save_token(data: dict) -> None:
    data["obtained_at"] = time.time()
    TOKEN_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _refresh(token: dict, client_key: str, client_secret: str) -> dict:
    resp = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "client_key": client_key,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": token["refresh_token"],
        },
        timeout=15,
    )
    data = resp.json()
    if resp.status_code != 200 or "access_token" not in data:
        raise RuntimeError(f"TikTok token refresh failed: {data}")
    _save_token(data)
    return data


def get_valid_access_token(client_key: str, client_secret: str) -> str:
    token = _load_token()
    obtained_at = token.get("obtained_at", 0)
    expires_at = obtained_at + token.get("expires_in", 0)
    if time.time() > expires_at - REFRESH_BUFFER_SECONDS:
        token = _refresh(token, client_key, client_secret)
    return token["access_token"]


def _upload_bytes(upload_url: str, video_path: Path, video_size: int) -> None:
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    resp = requests.put(
        upload_url,
        headers={
            "Content-Type": "video/mp4",
            "Content-Range": f"bytes 0-{video_size - 1}/{video_size}",
        },
        data=video_bytes,
        timeout=180,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"TikTok video upload failed: {resp.status_code} {resp.text}")


def get_user_info(access_token: str) -> dict:
    """Display name and avatar of the signed-in account (user.info.basic).
    Shown before an upload so the operator can confirm the video is going to
    the intended account -- the app holds a long-lived refresh token, so
    "which account am I actually connected to" is worth answering on screen."""
    resp = requests.get(
        USER_INFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        params={"fields": "open_id,display_name,avatar_url"},
        timeout=15,
    )
    data = resp.json()
    if resp.status_code != 200 or data.get("error", {}).get("code") not in (None, "ok"):
        raise RuntimeError(f"TikTok user info query failed: {data}")
    return data["data"]["user"]


def get_creator_info(access_token: str) -> dict:
    """Account settings TikTok requires an app to respect before posting
    (nickname, allowed privacy levels, duration cap, interaction toggles).
    Used by the consent page; also a cheap check that the token works."""
    resp = requests.post(
        CREATOR_INFO_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        },
        timeout=15,
    )
    data = resp.json()
    if resp.status_code != 200 or data.get("error", {}).get("code") not in (None, "ok"):
        raise RuntimeError(f"TikTok creator_info query failed: {data}")
    return data["data"]


def upload_to_inbox(access_token: str, video_path: Path) -> str:
    """Send the video to the account's TikTok inbox as a draft and return a
    publish_id. The owner reviews and posts it from the TikTok app, so no
    caption or privacy level is sent here -- both are chosen there."""
    video_size = video_path.stat().st_size

    init_resp = requests.post(
        INBOX_INIT_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        },
        json={
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_size,
                "chunk_size": video_size,
                "total_chunk_count": 1,
            },
        },
        timeout=30,
    )
    init_data = init_resp.json()
    error = init_data.get("error", {})
    if init_resp.status_code != 200 or error.get("code") not in (None, "ok"):
        raise RuntimeError(f"TikTok inbox init failed: {init_data}")

    _upload_bytes(init_data["data"]["upload_url"], video_path, video_size)
    return init_data["data"]["publish_id"]


def publish_video(access_token: str, video_path: Path, caption: str, privacy_level: str = "SELF_ONLY") -> str:
    """Runs init -> upload and returns a publish_id. Does not wait for
    moderation -- call check_status separately to see the final result.
    privacy_level defaults to SELF_ONLY (only visible to the poster) as a
    safety default; TikTok also force-restricts unaudited API clients to
    private visibility regardless of what's requested here."""
    video_size = video_path.stat().st_size

    init_resp = requests.post(
        INIT_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        },
        json={
            "post_info": {
                "title": caption,
                "privacy_level": privacy_level,
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_size,
                "chunk_size": video_size,
                "total_chunk_count": 1,
            },
        },
        timeout=30,
    )
    init_data = init_resp.json()
    error = init_data.get("error", {})
    if init_resp.status_code != 200 or error.get("code") not in (None, "ok"):
        raise RuntimeError(f"TikTok post init failed: {init_data}")

    publish_id = init_data["data"]["publish_id"]
    upload_url = init_data["data"]["upload_url"]

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    upload_resp = requests.put(
        upload_url,
        headers={
            "Content-Type": "video/mp4",
            "Content-Range": f"bytes 0-{video_size - 1}/{video_size}",
        },
        data=video_bytes,
        timeout=120,
    )
    if upload_resp.status_code not in (200, 201):
        raise RuntimeError(f"TikTok video upload failed: {upload_resp.status_code} {upload_resp.text}")

    return publish_id


def check_status(access_token: str, publish_id: str) -> dict:
    resp = requests.post(
        STATUS_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        },
        json={"publish_id": publish_id},
        timeout=15,
    )
    data = resp.json()
    if resp.status_code != 200:
        raise RuntimeError(f"TikTok status check failed: {data}")
    return data["data"]


def wait_for_status(access_token: str, publish_id: str, timeout_seconds: int = 90, poll_interval: int = 5) -> dict:
    """Poll status until it leaves the PROCESSING_UPLOAD state or the timeout
    is hit. TikTok rate-limits this endpoint to 30 requests/min/token."""
    deadline = time.time() + timeout_seconds
    status = check_status(access_token, publish_id)
    while status.get("status") == "PROCESSING_UPLOAD" and time.time() < deadline:
        time.sleep(poll_interval)
        status = check_status(access_token, publish_id)
    return status
