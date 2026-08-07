import json
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from config import ANTHROPIC_API_KEY, CASES_DIR, DATA_DIR, METADATA_MODEL

HASHTAG_POOL_PATH = DATA_DIR / "hashtag_pool.json"
TRENDING_STALE_DAYS = 14
MAX_HASHTAGS = 10

SYSTEM_PROMPT = """You are a social media metadata writer for a true-crime short-form video \
series distributed as TikTok "Part 1, Part 2, ..." episodes. You receive the full multi-part \
script (already written) and produce TikTok metadata for each part.

For every part, write:
- title: the line a viewer reads first, so it decides whether they keep watching. Under 60 \
characters. It must contain, in this order: the subject people actually search for (the case or \
person's name), a concrete hook drawn from THIS part, and "Part N". Example shape: \
"The car with no passenger seat | Ted Bundy Part 3". Use a specific detail from the part -- a \
missing seat, a misspelled name, a date -- never a vague tease like "You won't believe this". \
It must be accurate: no claim that isn't in the script.
- caption: a 1-3 sentence TikTok post caption. Open with a question or an unresolved detail from \
this part, and for parts after the first say which series and part it continues \
(e.g. "Part 4 of the Ted Bundy case..."). End with a reason to watch the next part.
- hashtags: 6-10 relevant hashtags, each a single lowercase word/phrase with no spaces and no \
leading '#' (it will be added later). Include case-specific ones a viewer would actually search \
(the person's name, the city, the era) plus broad true-crime ones. You will be given a pool of \
approved tags -- prefer tags from it where they fit, and add case-specific tags of your own. \
Never use a tag from the "banned" list. Only tags that genuinely fit -- don't pad.

Only use facts consistent with the given script. Do not invent claims not present in it.

Output ONLY the JSON object matching the given schema.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "part_number": {"type": "integer"},
                    "title": {"type": "string"},
                    "caption": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["part_number", "title", "caption", "hashtags"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["case_id", "parts"],
    "additionalProperties": False,
}


def _case_dir(case_id: str):
    d = CASES_DIR / case_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_pool() -> dict:
    if not HASHTAG_POOL_PATH.exists():
        return {"evergreen": [], "series": [], "trending": [], "banned": [], "trending_updated": None}
    return json.loads(HASHTAG_POOL_PATH.read_text(encoding="utf-8"))


def _trending_age_days(pool: dict):
    stamp = pool.get("trending_updated")
    if not stamp:
        return None
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).days


def _compose_hashtags(model_tags: list, pool: dict, use_trending: bool) -> list:
    """Final tag list for one part: what the model chose for this case first
    (those are the ones a viewer would actually search), then trending tags
    if they're still fresh, then evergreen and series tags to fill up.
    Banned tags are dropped at every step."""
    banned = {t.lower() for t in pool.get("banned", [])}
    ordered = []

    def add(tags):
        for tag in tags:
            clean = tag.strip().lstrip("#").lower().replace(" ", "")
            if clean and clean not in banned and clean not in ordered:
                ordered.append(clean)

    add(model_tags)
    if use_trending:
        add(pool.get("trending", []))
    add(pool.get("evergreen", []))
    add(pool.get("series", []))
    return ordered[:MAX_HASHTAGS]


def _load_script(case_id: str) -> dict:
    path = _case_dir(case_id) / "script.json"
    if not path.exists():
        raise RuntimeError(f"script.json not found for case {case_id} -- run the script stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def run(case_id: str, db) -> None:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")

    script = _load_script(case_id)
    pool = _load_pool()
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_prompt = (
        "Write TikTok metadata for every part of this script.\n\n"
        "Approved hashtag pool (prefer these where they fit, and add case-specific tags "
        f"of your own):\n{json.dumps({k: pool.get(k, []) for k in ('evergreen', 'trending', 'banned')}, ensure_ascii=False)}\n\n"
        f"{json.dumps(script, ensure_ascii=False, indent=2)}"
    )

    response = client.messages.create(
        model=METADATA_MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )

    db.log_usage(case_id, "metadata", METADATA_MODEL, response.usage.input_tokens, response.usage.output_tokens)

    if response.stop_reason == "refusal":
        raise RuntimeError("Model refused the metadata request")

    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        raise RuntimeError("Model returned no text content")

    metadata = json.loads(text_blocks[0])

    age = _trending_age_days(pool)
    use_trending = bool(pool.get("trending")) and age is not None and age <= TRENDING_STALE_DAYS
    for part in metadata["parts"]:
        part["hashtags"] = _compose_hashtags(part.get("hashtags", []), pool, use_trending)

    out_path = _case_dir(case_id) / "metadata.json"
    out_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    db.update_case_status(case_id, "metadata_done")

    parts = metadata["parts"]
    print(f"  metadata written: {out_path}")
    if not pool.get("trending"):
        print("  no trending hashtags configured -- run tools/update_trending_hashtags.py "
              "with tags from TikTok Creative Center to add some")
    elif not use_trending:
        print(f"  trending hashtags are {age} days old (stale past {TRENDING_STALE_DAYS}) -- "
              "skipped; refresh with tools/update_trending_hashtags.py")
    else:
        print(f"  trending hashtags included ({age} day(s) old): {', '.join(pool['trending'][:5])}")
    for p in parts:
        print(f"    part {p['part_number']}: {p['title']!r}")
        print(f"      #{' #'.join(p['hashtags'])}")
