import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Windows terminals often default to a legacy codepage (e.g. cp1252) that
# can't encode paths under this project's Cyrillic directory name. Force
# UTF-8 stdout/stderr so console output never crashes on that.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CASES_DIR = DATA_DIR / "cases"
DB_PATH = DATA_DIR / "pipeline.db"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

USE_CHEAP_MODEL = os.environ.get("USE_CHEAP_MODEL", "false").lower() == "true"
_CHEAP_MODEL = "claude-haiku-4-5"
_FULL_MODEL = "claude-opus-4-8"

RESEARCH_MODEL = _CHEAP_MODEL if USE_CHEAP_MODEL else _FULL_MODEL
SCRIPT_MODEL = _CHEAP_MODEL if USE_CHEAP_MODEL else _FULL_MODEL
METADATA_MODEL = _CHEAP_MODEL if USE_CHEAP_MODEL else _FULL_MODEL
# Vision check for archive photo candidates -- catches keyword-coincidence
# false positives (e.g. "crowbar" matching an electronics circuit diagram)
# that pure text matching can't. Cheap model is fine here (yes/no + reason).
IMAGE_VERIFY_MODEL = _CHEAP_MODEL

# USD per 1M tokens: (input, output). Used only for local cost estimates
# printed to the console -- not a substitute for the spend limit in
# console.anthropic.com -> Settings -> Limits.
MODEL_PRICING = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

VOICES_DIR = DATA_DIR / "voices"
VOICE_MODEL_PATH = VOICES_DIR / "en_US-hfc_male-medium.onnx"
VOICE_CONFIG_PATH = VOICES_DIR / "en_US-hfc_male-medium.onnx.json"

# Publisher safety switch. Real posting requires TikTok Content Posting API
# credentials that this project does not yet have configured -- keep this
# true until that's set up and you've manually reviewed at least a few videos.
PUBLISH_DRY_RUN = os.environ.get("PUBLISH_DRY_RUN", "true").lower() == "true"

# "inbox" sends the video to the account's TikTok drafts for the owner to
# review and post from the app (video.upload scope, light app review).
# "direct" posts straight to the feed (video.publish scope, requires the full
# Direct Post consent UI before TikTok will approve the app).
TIKTOK_POST_MODE = os.environ.get("TIKTOK_POST_MODE", "inbox").lower()
