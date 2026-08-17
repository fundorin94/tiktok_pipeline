"""Lay metadata.json out as a text file you can copy-paste into TikTok.

The metadata stage writes JSON, which is the wrong shape for the actual
posting: uploading by hand means selecting one description, hashtags and all,
and pasting it into the caption box. Digging that out of nested JSON per part
is where typos come from.

Regenerate this after every metadata run -- the stage rewrites titles and
captions from scratch each time, so a file kept from an earlier run describes
videos that no longer exist.

Usage:
  python tools/export_metadata.py test3
  python tools/export_metadata.py test3 --print
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# TikTok's caption box holds 2200 characters, and hashtags are part of that
# count -- worth showing per part, since a long caption plus ten tags creeps up
# on it and the box truncates silently.
CAPTION_LIMIT = 2200
RULE = "=" * 64
THIN = "-" * 64


def _blocks(case_id: str, meta: dict, video_dir: Path):
    parts = sorted(meta["parts"], key=lambda p: p["part_number"])
    yield f"{case_id} -- {len(parts)} part(s)"
    yield "Copy everything between the dashed lines into the TikTok caption box."
    yield ""

    for part in parts:
        number = part["part_number"]
        tags = " ".join("#" + t for t in part["hashtags"])
        description = f"{part['caption']}\n\n{tags}"
        video = video_dir / f"part{number}.mp4"

        yield RULE
        yield f"PART {number}"
        yield f"video:  {video if video.is_file() else str(video) + '   *** MISSING ***'}"
        yield f"title:  {part['title']}   ({len(part['title'])} chars)"
        yield RULE
        yield THIN
        yield description
        yield THIN
        yield f"description: {len(description)} / {CAPTION_LIMIT} characters"
        yield ""


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit("usage: python tools/export_metadata.py <case_id> [--print]")
    case_id = args[0]

    case_dir = ROOT / "data" / "cases" / case_id
    meta_path = case_dir / "metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found -- run the metadata stage first")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    text = "\n".join(_blocks(case_id, meta, case_dir / "video")) + "\n"

    out_path = case_dir / "publish.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"written: {out_path}")
    if "--print" in sys.argv[1:]:
        print()
        print(text)


if __name__ == "__main__":
    main()
