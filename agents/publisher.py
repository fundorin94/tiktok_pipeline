import json
import os
from datetime import datetime, timezone
from pathlib import Path

from config import CASES_DIR, PUBLISH_DRY_RUN, TIKTOK_POST_MODE


def _case_dir(case_id: str) -> Path:
    d = CASES_DIR / case_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise RuntimeError(f"{label} not found at {path} -- run the earlier stages first")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _build_caption(part: dict) -> str:
    return part["caption"] + " " + " ".join(f"#{h}" for h in part["hashtags"])


def _write_caption_file(case_dir: Path, part: dict, caption: str) -> Path:
    """A ready-to-paste caption for the TikTok app. Drafts uploaded to the
    inbox carry no title or caption -- the API has nowhere to put them -- so
    the text has to reach the phone some other way, and a file per part is
    the least error-prone."""
    captions_dir = case_dir / "captions"
    captions_dir.mkdir(parents=True, exist_ok=True)
    path = captions_dir / f"part{part['part_number']}.txt"
    path.write_text(
        f"{part['title']}\n\n{caption}\n",
        encoding="utf-8",
    )
    return path


def run(case_id: str, db) -> None:
    case_dir = _case_dir(case_id)
    metadata = _load_json(case_dir / "metadata.json", "metadata.json")

    review_queue = _load_json_if_exists(case_dir / "review_queue.json")
    manual_queue = _load_json_if_exists(case_dir / "manual_sourcing_queue.json")
    pending_review_count = len(review_queue) + len(manual_queue)

    video_dir = case_dir / "video"

    if not PUBLISH_DRY_RUN and pending_review_count:
        raise RuntimeError(
            f"{pending_review_count} scene(s) in this case are still awaiting manual visual "
            "review (review_queue.json / manual_sourcing_queue.json). Resolve those before "
            "posting for real -- refusing to publish with unreviewed visuals."
        )

    access_token = None
    if not PUBLISH_DRY_RUN:
        from agents import tiktok_client

        client_key = os.environ.get("TIKTOK_CLIENT_KEY")
        client_secret = os.environ.get("TIKTOK_CLIENT_SECRET")
        if not client_key or not client_secret:
            raise RuntimeError("TIKTOK_CLIENT_KEY / TIKTOK_CLIENT_SECRET not set in .env")
        access_token = tiktok_client.get_valid_access_token(client_key, client_secret)

    records = []
    for part in metadata["parts"]:
        part_number = part["part_number"]
        video_path = video_dir / f"part{part_number}.mp4"
        caption = _build_caption(part)

        caption_file = _write_caption_file(case_dir, part, caption)

        record = {
            "case_id": metadata.get("case_id", case_id),
            "part_number": part_number,
            "dry_run": PUBLISH_DRY_RUN,
            "platform": "tiktok",
            "video_path": str(video_path),
            "title": part["title"],
            "caption": caption,
            "caption_file": str(caption_file),
            "scheduled_for": None,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }

        if not video_path.exists():
            record["status"] = "missing_video"
            record["note"] = f"video file not found at {video_path} -- run the video stage for this case"
            records.append(record)
            continue

        if PUBLISH_DRY_RUN:
            record["status"] = "pending_review"
            record["note"] = "dry_run=True: nothing was sent to TikTok. Review video_path manually before posting."
        else:
            from agents import tiktok_client

            try:
                if TIKTOK_POST_MODE == "inbox":
                    publish_id = tiktok_client.upload_to_inbox(access_token, video_path)
                    note = ("delivered to the account's TikTok inbox -- open the app, tap the "
                            "upload notification, paste the caption and post.")
                else:
                    publish_id = tiktok_client.publish_video(access_token, video_path, caption)
                    note = "posted via TikTok Content Posting API"
                status_data = tiktok_client.wait_for_status(access_token, publish_id)
                record["publish_id"] = publish_id
                record["post_mode"] = TIKTOK_POST_MODE
                record["status"] = status_data.get("status", "UNKNOWN")
                record["note"] = status_data.get("fail_reason") or note
            except RuntimeError as exc:
                record["status"] = "error"
                record["note"] = str(exc)

        records.append(record)

    out_path = case_dir / "publish_log.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    db.update_case_status(case_id, "publish_dry_run_done" if PUBLISH_DRY_RUN else "publish_done")

    print(f"  publish log written: {out_path}")
    print(f"  captions ready to paste: {case_dir / 'captions'}")
    if PUBLISH_DRY_RUN:
        print(f"  {len(records)} part(s) logged as dry-run -- nothing was posted to TikTok")
    else:
        for r in records:
            print(f"    part {r['part_number']}: {r['status']}")
        if TIKTOK_POST_MODE == "inbox":
            print("  videos delivered to your TikTok inbox -- open the app, tap the upload "
                  "notification, paste the caption from captions/partN.txt, and post")
    if pending_review_count:
        print(f"  warning: {pending_review_count} scene(s) still need manual visual review")
