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

# Narration voice. Piper is instant but reads everything at one pitch and
# pace, which is the wrong register for this genre. Qwen3-TTS takes a
# plain-language instruction and delivers the same script with emphasis and
# timing, at roughly 3x realtime on this card -- worth the wait here.
VOICE_ENGINE = os.environ.get("VOICE_ENGINE", "qwen").lower()  # "qwen" | "piper"
QWEN_TTS_MODEL = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
QWEN_TTS_SPEAKER = os.environ.get("QWEN_TTS_SPEAKER", "Ryan")
QWEN_TTS_INSTRUCT = os.environ.get(
    "QWEN_TTS_INSTRUCT",
    "Speak briskly and expressively, stressing the key words in each sentence, "
    "with short sharp pauses before the most disturbing facts.",
)
# Asking the model to speak faster makes it more expressive, not quicker --
# the readings came back longer. Tempo is set afterwards instead, which is
# exact and leaves pitch untouched.
VOICE_TEMPO = float(os.environ.get("VOICE_TEMPO", "1.0"))
# Each scene is a separate generation, and at the model's default sampling
# temperature the delivery drifted noticeably between them -- one scene
# breathy, the next clipped. A low temperature and a fixed seed keep one
# reading across a whole episode.
QWEN_TTS_TEMPERATURE = float(os.environ.get("QWEN_TTS_TEMPERATURE", "0.5"))
QWEN_TTS_SEED = int(os.environ.get("QWEN_TTS_SEED", "20260811"))

# Fixing the seed was not enough: it pins the random stream, but the text
# differs from scene to scene, so the model kept re-interpreting the style
# brief and the delivery drifted across an episode. Conditioning every scene
# on one reference recording anchors the timbre AND the manner, which is
# what "the same narrator" actually means.
QWEN_TTS_MODE = os.environ.get("QWEN_TTS_MODE", "clone").lower()  # "clone" | "custom"
QWEN_TTS_CLONE_MODEL = os.environ.get("QWEN_TTS_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
QWEN_TTS_REF_AUDIO = DATA_DIR / "voices" / "reference_narrator.wav"
QWEN_TTS_REF_TEXT = (
    "On August 16, 1975, a highway patrol officer stopped a tan Volkswagen Beetle "
    "in Granger, Utah. When he looked inside, he saw something odd. The front "
    "passenger seat was missing. In the back he found a ski mask, handcuffs, "
    "and a crowbar."
)

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
