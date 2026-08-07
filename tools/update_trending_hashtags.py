"""Refresh the trending hashtags used in captions.

TikTok publishes no API for trending hashtags -- the Content Posting API
doesn't expose them and the Research API is limited to approved researchers.
So they're copied in by hand, and stamped with a date so the metadata stage
can tell when they've gone stale and stop using them.

Where to get them:
  TikTok Creative Center -> Trends -> Hashtags
  https://ads.tiktok.com/business/creativecenter/inspiration/popular/hashtag/
  Filter by your region and the Entertainment or News category, then take the
  handful that actually suit a true-crime documentary.

Usage:
  python tools/update_trending_hashtags.py truecrimestory casefile unsolved
  python tools/update_trending_hashtags.py --show
  python tools/update_trending_hashtags.py --clear
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

POOL_PATH = Path(__file__).resolve().parent.parent / "data" / "hashtag_pool.json"


def _load() -> dict:
    return json.loads(POOL_PATH.read_text(encoding="utf-8"))


def _save(pool: dict) -> None:
    POOL_PATH.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list) -> int:
    if not POOL_PATH.exists():
        print(f"hashtag pool not found at {POOL_PATH}")
        return 1
    pool = _load()

    if not argv or argv[0] == "--show":
        stamp = pool.get("trending_updated") or "never"
        print(f"trending ({stamp}): {', '.join(pool.get('trending') or []) or '(none)'}")
        print(f"evergreen: {', '.join(pool.get('evergreen') or [])}")
        print(f"banned: {', '.join(pool.get('banned') or [])}")
        return 0

    if argv[0] == "--clear":
        pool["trending"] = []
        pool["trending_updated"] = None
        _save(pool)
        print("trending hashtags cleared")
        return 0

    banned = {t.lower() for t in pool.get("banned", [])}
    tags, skipped = [], []
    for raw in argv:
        tag = raw.strip().lstrip("#").lower().replace(" ", "")
        if not tag:
            continue
        (skipped if tag in banned else tags).append(tag)

    if skipped:
        print(f"skipped banned tag(s): {', '.join(skipped)}")
    if not tags:
        print("nothing to save")
        return 1

    pool["trending"] = tags
    pool["trending_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save(pool)
    print(f"trending hashtags set ({len(tags)}): {', '.join(tags)}")
    print("these will be used in captions for the next 14 days, then skipped as stale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
