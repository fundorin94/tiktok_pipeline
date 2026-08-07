import base64
import io
import json
import mimetypes
import time
from pathlib import Path

import anthropic

from config import ANTHROPIC_API_KEY, IMAGE_VERIFY_MODEL

# Hard wall-clock cap on a single verify call so a slow/hung request can never
# stall the whole archive stage (a large upload once hung it for ~an hour).
VERIFY_TIMEOUT_SECONDS = 45
# Downscale before upload -- full-res archive scans can be many MB, which is
# slow to base64/upload and adds nothing to a yes/no relevance judgement.
VERIFY_MAX_DIM = 768
# The safety gate only answers "is there a person / modern object / big text",
# which needs far less detail than judging whether a photo matches a subject.
# Image tokens scale with area, so halving the edge quarters the cost of the
# single most-called check in the pipeline.
SAFETY_MAX_DIM = 384


def _encode_downscaled(path: Path, max_dim: int = VERIFY_MAX_DIM):
    """Return (base64_str, media_type) for a size-capped JPEG of the image.
    Falls back to the raw bytes if PIL can't open it."""
    try:
        from PIL import Image
        img = Image.open(path)
        img.thumbnail((max_dim, max_dim))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.standard_b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"
    except Exception:
        media_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        if media_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
            media_type = "image/jpeg"
        return base64.standard_b64encode(path.read_bytes()).decode("ascii"), media_type

SYSTEM_PROMPT = """You check whether a candidate archive photo actually depicts what it's \
supposed to, before it's used in a true-crime documentary video. Text-based search frequently \
returns wrong matches from keyword coincidences -- e.g. a query for "crowbar" (a tool) matching \
an electronics "crowbar circuit" diagram, or "Beetle" (a car) matching an actual beetle insect \
on a postage stamp, or a query for a specific location matching a same-named but unrelated \
place from a different country/era. Look at the actual image content and judge whether it \
genuinely shows the described subject -- not just whether the words could overlap.

Also reject it (matches=false) if it is not a real photograph -- an old painting, watercolor, \
drawing, sketch, engraving, or diagram does not belong in a photographic documentary even if its \
subject matches the words.

Answer with a JSON object only."""

# For a named person, identity was already confirmed by matching their name
# in the source file's title/caption -- asking the model to also confirm
# *who* is in the photo from pixels alone is an impossible, unreliable task
# it will (correctly) refuse, rejecting perfectly good photos. The vision
# check should only catch gross category mismatches for these: a diagram,
# document, unrelated object, or an obviously different kind of scene
# standing in for a person photo.
PERSON_SYSTEM_PROMPT = """You do a quick sanity check on a candidate archive photo before it's \
used in a true-crime documentary video, for a scene about a specific named person. The person's \
identity was already confirmed by matching their name in the source file's own title/caption -- \
you are NOT being asked to verify who is in the photo (that's not reliably possible from pixels \
alone, and isn't the point of this check).

Text-based search sometimes returns a wrong *kind* of result entirely due to keyword \
coincidences -- e.g. a person's name query matching an electronics diagram, a document scan, an \
unrelated object, or a completely different scene that happens to share a word. Just check: is \
this actually a photograph of a person (a portrait, mugshot, court/booking photo, yearbook \
photo, etc.), as opposed to something else entirely? Do not reject it for looking like the \
"wrong" person, wrong angle, low quality, or otherwise being an unremarkable photo of *a* \
person -- only reject if it clearly isn't a photo of a person at all.

Answer with a JSON object only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["matches", "reason"],
    "additionalProperties": False,
}

# Final safety gate for AI-GENERATED frames. Local detectors (YuNet faces,
# NudeNet body parts) miss grainy b&w edge cases -- e.g. an unclothed body
# lying on a floor shot from above passed both. AI frames must contain no
# human at all, so this is a strict binary judgement by a vision model.
AI_SAFETY_PROMPT = """You are a strict safety gate for AI-generated b-roll frames in a \
true-crime documentary. The frame is supposed to show ONLY places, objects, documents, \
vehicles or landscapes -- absolutely no people.

Answer safe=false if the image contains ANY of:
- a human being or any part of one (face, body, torso, limbs, hands, silhouette, reflection), \
clothed or not, alive or dead, sharp or blurry, at any size;
- a corpse or anything reading as a dead/unconscious body, body bag, or human-shaped covered form;
- nudity or partial nudity of any kind;
- blood, wounds, gore, or graphic violence aftermath;
- a mannequin, doll, statue or anatomical model of a person.

Also answer safe=false if:
- LETTERING DOMINATES the frame: a sign, headline, poster or document page that the eye reads as \
a main subject, or any large block of text. Image generation cannot spell, so text at that size \
reads as obvious gibberish. Small incidental markings are FINE and must NOT be rejected -- a \
logo on a hubcap, a distant shopfront, a spine on a shelf, faint background lettering.
- an object that plainly could not exist in the stated period is clearly visible: flat-screen / \
plasma / LCD television, computer monitor, laptop, mobile phone, or a car whose body shape is \
obviously from a later decade. Judge only unmistakable anachronisms -- do NOT reject a plain \
room, wall, floor or furniture merely for looking clean, undated or ambiguous in style.
- LIGHTING IS NOT AN ANACHRONISM. Never reject a frame over its light fixtures. Flat ceiling \
panels, strip lights and tall street lamps all existed as fluorescent and sodium fittings in \
the 1960s-70s, and in grainy black and white they are indistinguishable from modern LED \
equivalents. Ignore lighting entirely when judging the period.

Otherwise answer safe=true. When unsure, answer safe=false.
Answer with a JSON object only."""

AI_SAFETY_SCHEMA = {
    "type": "object",
    "properties": {
        "safe": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["safe", "reason"],
    "additionalProperties": False,
}


class SafetyCheckUnavailable(RuntimeError):
    """The safety gate could not run (no API credit, auth failure, outage).
    Raised instead of returning "unsafe" so the stage stops with a clear
    cause: a dead API once silently discarded 117 perfectly good frames,
    which looked like a quality problem rather than a billing one."""


_FATAL_API_MARKERS = (
    "credit balance is too low", "authentication_error", "permission_error",
    "invalid x-api-key", "billing",
)

_SAFETY_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "safety_cache.json"
_SAFETY_CACHE = None


def _cache_key(image_path: str) -> str:
    import hashlib
    return hashlib.sha1(Path(image_path).read_bytes()).hexdigest()


def _cache_get(key: str):
    global _SAFETY_CACHE
    if _SAFETY_CACHE is None:
        try:
            _SAFETY_CACHE = json.loads(_SAFETY_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _SAFETY_CACHE = {}
    return _SAFETY_CACHE.get(key)


def _cache_put(key: str, safe: bool, reason: str) -> None:
    _cache_get(key)  # ensure loaded
    _SAFETY_CACHE[key] = {"safe": safe, "reason": reason}
    try:
        _SAFETY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SAFETY_CACHE_PATH.write_text(json.dumps(_SAFETY_CACHE), encoding="utf-8")
    except OSError:
        pass  # cache is an optimization, never fail a run over it


def ai_frame_is_safe(image_path: str, era: str = ""):
    """Returns (safe: bool, reason: str, usage). `era` is the period the
    frame must look like ("1974"), so anachronisms are caught. Fails CLOSED:
    any API or parsing error counts as unsafe so an unchecked frame can
    never ship."""
    cache_key = _cache_key(image_path)
    hit = _cache_get(cache_key)
    if hit is not None:
        return hit["safe"], hit["reason"], None

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    data, media_type = _encode_downscaled(Path(image_path), SAFETY_MAX_DIM)
    question = "Is this frame safe under the rules above?"
    if era:
        question = f"The scene must look like {era}. " + question

    last_error = ""
    for attempt in range(3):
        safe, reason, usage = _ask_safety(client, data, media_type, question)
        if usage is not None:  # a real verdict
            _cache_put(cache_key, safe, reason)
            return safe, reason, usage
        last_error = reason
        if any(marker in reason.lower() for marker in _FATAL_API_MARKERS):
            break  # more attempts cannot help
        time.sleep(2 * (attempt + 1))
    raise SafetyCheckUnavailable(last_error)


def _ask_safety(client, data: str, media_type: str, question: str):
    try:
        response = client.messages.create(
            model=IMAGE_VERIFY_MODEL,
            max_tokens=300,
            system=AI_SAFETY_PROMPT,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": question},
            ]}],
            output_config={"format": {"type": "json_schema", "schema": AI_SAFETY_SCHEMA}},
            timeout=VERIFY_TIMEOUT_SECONDS,
        )
        verdict = json.loads(response.content[0].text)
        return bool(verdict.get("safe")), verdict.get("reason", ""), response.usage
    except Exception as exc:
        return False, f"safety check failed: {exc}", None


def verify_image(image_path: str, description: str, is_person: bool = False):
    """Returns (matches: bool, reason: str, usage) -- usage is None on failure.
    Fails open (treats the image as a match) if the API call itself errors,
    so a transient API issue doesn't block the whole archive stage; it only
    catches semantic mismatches when the check actually runs."""
    if not ANTHROPIC_API_KEY:
        return True, "verification skipped -- no API key", None

    path = Path(image_path)

    system_prompt = PERSON_SYSTEM_PROMPT if is_person else SYSTEM_PROMPT
    question = (
        f"Is this a photograph of a person (identity already confirmed by filename: {description})?"
        if is_person else f"Does this image actually show: {description}"
    )

    try:
        b64, media_type = _encode_downscaled(path)

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=VERIFY_TIMEOUT_SECONDS, max_retries=1)
        response = client.messages.create(
            model=IMAGE_VERIFY_MODEL,
            max_tokens=300,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": question},
                ],
            }],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        if not text_blocks:
            return True, "verification returned no content", response.usage
        result = json.loads(text_blocks[0])
        return bool(result.get("matches", True)), result.get("reason", ""), response.usage
    except Exception as exc:
        return True, f"verification failed ({exc}) -- accepted without check", None
