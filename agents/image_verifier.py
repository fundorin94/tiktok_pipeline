import base64
import io
import json
import mimetypes
import re
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

TWO EXCEPTIONS, because common names collide. A search for "Matt Turner", a 1991 murder \
victim, returned the United States national team goalkeeper in a 2022 match kit; "Raymond \
Smith" returned a US Navy officer's official portrait. Both are photographs of a person, both \
passed this check, and both would have appeared in a documentary captioned as someone they are \
not. So also answer matches=false when:
- THE PHOTOGRAPH IS PLAINLY LATER THAN THE CASE. You are told the era. Only LATER disqualifies: \
modern sports kit, a digital-era snapshot, current uniforms or insignia. EARLIER IS EXPECTED and \
must never be rejected -- these people had childhoods, school photographs and army portraits \
decades before the case, and a yearbook picture from twenty years earlier is exactly the kind of \
photograph that survives. Judge only clear cases; an undated studio portrait is fine.
- THE SUBJECT IS PLAINLY A DIFFERENT PUBLIC FIGURE. Someone photographed in the uniform of a \
role the description does not mention -- an athlete in team kit, an officer in dress uniform \
with service ribbons, a politician at a podium -- when the query names a private individual, \
is a name collision, not the person.

Answer with a JSON object only."""

DOCUMENT_SYSTEM_PROMPT = """You check whether a scanned document, letter, cipher, poster or newspaper page actually shows what it is supposed to, before it is used in a true-crime documentary video.

Do NOT require a photograph here. The subject IS a document: a page of handwriting, a grid of symbols, a printed poster, a police composite drawing, a newspaper column. Being a scan, a drawing or a diagram is exactly right and must never be a reason to reject it -- that rule exists to keep paintings out of photographic scenes, and does not apply to paperwork.

Reject it (matches=false) only when the document is the wrong one: a different case entirely, a modern reproduction or fan artwork rather than the historical item, an unrelated form, or a page whose subject has nothing to do with the description.

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
_PEOPLE_RULE_EMPTY = """The frame is supposed to show ONLY places, objects, documents, vehicles or landscapes -- absolutely no people."""

_PEOPLE_RULE_FIGURES = """The frame is supposed to show a scene with ANONYMOUS FIGURES in it: people seen from behind, in silhouette, at a distance, or cropped so that no face is visible. Those figures are wanted -- a back, a shoulder, a coat, a crowd seen from behind all leave safe=true. What must never appear is a FACE: no recognisable facial features at any size, front or profile, and none in a mirror, window or reflection. A frame that reads as a portrait of an identifiable individual fails, however small the face is."""

AI_SAFETY_PROMPT = """You are the gate for AI-generated b-roll frames in a true-crime documentary. {people_rule}

You give THREE SEPARATE verdicts, and they are not equally serious. `safe` is a hard rule: a frame failing it can never be used. `period_ok` and `text_ok` are quality preferences, used to choose between frames that are already safe -- judge them honestly, but do not let them colour the `safe` verdict.

Answer safe=false if the image contains ANY of:
{human_clause}
- a corpse or anything reading as a dead/unconscious body, body bag, or human-shaped covered form;
- nudity or partial nudity of any kind;
- blood, wounds, gore, or graphic violence aftermath;
- a mannequin, doll, statue or anatomical model of a person.

{implication_clause}

Otherwise answer safe=true.

Answer text_ok=false only if LETTERING DOMINATES the frame: a sign, headline, poster or document page that the eye reads as a main subject, or any large block of text. Image generation cannot spell, so text at that size reads as obvious gibberish. Small incidental markings are FINE and must leave text_ok=true -- a logo on a hubcap, a distant shopfront, a spine on a shelf, faint background lettering.

Answer period_ok=false only if an object that plainly could not exist in the stated period is clearly visible: flat-screen / plasma / LCD television, computer monitor, laptop, mobile phone, or a car whose body shape is obviously from a later decade. Judge only unmistakable anachronisms -- a plain room, wall, floor or furniture that merely looks clean, undated or ambiguous in style is period_ok=true. LIGHTING IS NEVER AN ANACHRONISM: flat ceiling panels, strip lights and tall street lamps all existed as fluorescent and sodium fittings in the 1960s-70s, and in grainy black and white they are indistinguishable from modern LED equivalents. Ignore lighting entirely when judging the period.

Put the reason for whichever verdict is false in `reason`, most serious first.
Answer with a JSON object only."""

AI_SAFETY_DOCUMENT_SYSTEM_PROMPT = """You check whether a scanned document, letter, cipher, poster or newspaper page actually shows what it is supposed to, before it is used in a true-crime documentary video.

Do NOT require a photograph here. The subject IS a document: a page of handwriting, a grid of symbols, a printed poster, a police composite drawing, a newspaper column. Being a scan, a drawing or a diagram is exactly right and must never be a reason to reject it -- that rule exists to keep paintings out of photographic scenes, and does not apply to paperwork.

Reject it (matches=false) only when the document is the wrong one: a different case entirely, a modern reproduction or fan artwork rather than the historical item, an unrelated form, or a page whose subject has nothing to do with the description.

Answer with a JSON object only."""

_HUMAN_CLAUSE_EMPTY = """- a human being or any part of one (face, body, torso, limbs, hands, silhouette, reflection), clothed or not, alive or dead, sharp or blurry, at any size;"""

_HUMAN_CLAUSE_FIGURES = """- a visible human FACE or recognisable facial features, at any size, front or profile, including one in a mirror, window or other reflection;"""

_IMPLICATION_EMPTY = """`safe` is about a person being VISIBLE, never about a person being implied. Every scene here is a place people use, so furniture, vehicles, tools, clothing on a rack, a made bed, a ceiling fan, a lit lamp or an open door are all expected and leave safe=true -- "suggests occupancy" is not a reason to fail a frame, and an empty room is the entire point of the shot. Fail it only when you can actually point at a person or a body."""

_IMPLICATION_FIGURES = """Judge only what is visible. A figure with its back turned, a silhouette against a window, a shape too distant to read, a hand at the edge of frame -- none of these is a face, and none of them fails. Fail the frame when you can actually see facial features."""


def _safety_prompt(people_allowed: bool) -> str:
    return AI_SAFETY_PROMPT.format(
        people_rule=_PEOPLE_RULE_FIGURES if people_allowed else _PEOPLE_RULE_EMPTY,
        human_clause=_HUMAN_CLAUSE_FIGURES if people_allowed else _HUMAN_CLAUSE_EMPTY,
        implication_clause=_IMPLICATION_FIGURES if people_allowed else _IMPLICATION_EMPTY,
    )


AI_SAFETY_SCHEMA = {
    "type": "object",
    "properties": {
        "safe": {"type": "boolean"},
        "period_ok": {"type": "boolean"},
        "text_ok": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["safe", "period_ok", "text_ok", "reason"],
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
# Bump this whenever AI_SAFETY_PROMPT or the verdict shape changes: the key is
# the image bytes alone, so a cached verdict outlives the rules that produced
# it. v1 entries carry only safe/reason, and a stored safe=false could have
# meant "a person" or "a modern lamp" -- guessing which would silently
# re-reject the frames the split exists to keep, so they are left unread.
_CACHE_VERSION = "v4"


def _cache_key(image_path: str, people_allowed: bool) -> str:
    """The mode is part of the key: the same bytes get a different verdict
    depending on whether figures were allowed, and reusing one for the other
    would either smuggle a face through or reject a legitimate back view."""
    import hashlib
    mode = "figures" if people_allowed else "empty"
    return f"{_CACHE_VERSION}:{mode}:" + hashlib.sha1(Path(image_path).read_bytes()).hexdigest()


def _cache_get(key: str):
    global _SAFETY_CACHE
    if _SAFETY_CACHE is None:
        try:
            _SAFETY_CACHE = json.loads(_SAFETY_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _SAFETY_CACHE = {}
    return _SAFETY_CACHE.get(key)


def _cache_put(key: str, verdict: dict) -> None:
    _cache_get(key)  # ensure loaded
    _SAFETY_CACHE[key] = verdict
    try:
        _SAFETY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SAFETY_CACHE_PATH.write_text(json.dumps(_SAFETY_CACHE), encoding="utf-8")
    except OSError:
        pass  # cache is an optimization, never fail a run over it


def ai_frame_verdict(image_path: str, era: str = "", people_allowed: bool = False):
    """Returns (verdict: dict, usage). The verdict carries three separate
    judgements:

      safe      -- no people/nudity/gore. A hard rule; a false here means the
                   frame can never be used, whatever else it has going for it.
      period_ok -- no unmistakable anachronism for `era` ("1974").
      text_ok   -- no lettering large enough to read as the subject.

    The last two are preferences, not vetoes: they decide which of several
    safe frames is best. Keeping them apart from `safe` is the whole point --
    when one verdict covered all three, four rejections in five were about a
    modern-looking lamp or a garbled sign, and each cost a full re-generation.

    Fails CLOSED on safe: any API or parsing error counts as unsafe, so an
    unchecked frame can never ship."""
    cache_key = _cache_key(image_path, people_allowed)
    hit = _cache_get(cache_key)
    if hit is not None:
        return hit, None

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    data, media_type = _encode_downscaled(Path(image_path), SAFETY_MAX_DIM)
    question = "Judge this frame under the rules above."
    if era:
        question = f"The scene must look like {era}. " + question

    last_error = ""
    for attempt in range(3):
        verdict, usage = _ask_safety(client, data, media_type, question, people_allowed)
        if usage is not None:  # a real verdict
            _cache_put(cache_key, verdict)
            return verdict, usage
        last_error = verdict["reason"]
        if any(marker in last_error.lower() for marker in _FATAL_API_MARKERS):
            break  # more attempts cannot help
        time.sleep(2 * (attempt + 1))
    raise SafetyCheckUnavailable(last_error)


def _ask_safety(client, data: str, media_type: str, question: str,
                people_allowed: bool = False):
    try:
        response = client.messages.create(
            model=IMAGE_VERIFY_MODEL,
            max_tokens=300,
            system=_safety_prompt(people_allowed),
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": question},
            ]}],
            output_config={"format": {"type": "json_schema", "schema": AI_SAFETY_SCHEMA}},
            timeout=VERIFY_TIMEOUT_SECONDS,
        )
        raw = json.loads(response.content[0].text)
        return {
            "safe": bool(raw.get("safe")),
            # Default the soft verdicts to OK: a malformed response should not
            # invent a quality complaint, and `safe` above already fails closed.
            "period_ok": bool(raw.get("period_ok", True)),
            "text_ok": bool(raw.get("text_ok", True)),
            "reason": raw.get("reason", ""),
        }, response.usage
    except Exception as exc:
        return {"safe": False, "period_ok": True, "text_ok": True,
                "reason": f"safety check failed: {exc}"}, None


_RELEVANCE_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "relevance_cache.json"
_RELEVANCE_CACHE = None


def _relevance_key(image_path: str, description: str, mode: str) -> str:
    """Same bytes, same question, same answer -- so key on all three.

    The planning pass and the archive stage search the same queries and
    download the same candidates, and the archive stage wipes accepted/ on
    every run, so without this the relevance check is paid for twice on a
    case that was planned first, and again on every rerun. Downloads were
    already cached (data/media_cache); this is the half that costs money."""
    import hashlib
    try:
        digest = hashlib.sha1(Path(image_path).read_bytes()).hexdigest()
    except OSError:
        return ""
    question = re.sub(r"\s+", " ", description.strip().lower())
    return f"{mode}:{hashlib.sha1(question.encode('utf-8')).hexdigest()[:16]}:{digest}"


def _relevance_get(key: str):
    global _RELEVANCE_CACHE
    if _RELEVANCE_CACHE is None:
        try:
            _RELEVANCE_CACHE = json.loads(_RELEVANCE_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _RELEVANCE_CACHE = {}
    return _RELEVANCE_CACHE.get(key)


def _relevance_put(key: str, matches: bool, reason: str) -> None:
    _relevance_get(key)  # ensure loaded
    _RELEVANCE_CACHE[key] = {"matches": matches, "reason": reason}
    try:
        _RELEVANCE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RELEVANCE_CACHE_PATH.write_text(json.dumps(_RELEVANCE_CACHE), encoding="utf-8")
    except OSError:
        pass  # a cache is an optimisation, never fail a run over it


def verify_image(image_path: str, description: str, is_person: bool = False,
                 is_document: bool = False, era: str = ""):
    """Returns (matches: bool, reason: str, usage) -- usage is None on failure.
    Fails open (treats the image as a match) if the API call itself errors,
    so a transient API issue doesn't block the whole archive stage; it only
    catches semantic mismatches when the check actually runs."""
    if not ANTHROPIC_API_KEY:
        return True, "verification skipped -- no API key", None

    mode = "document" if is_document else ("person" if is_person else "object")
    if era and is_person:
        mode += ":" + era   # era changes the verdict, so it belongs in the cache key
    cache_key = _relevance_key(image_path, description, mode)
    if cache_key:
        hit = _relevance_get(cache_key)
        if hit is not None:
            return hit["matches"], hit["reason"], None

    path = Path(image_path)

    if is_document:
        system_prompt = DOCUMENT_SYSTEM_PROMPT
        question = f"Is this the document described: {description}"
    elif is_person:
        system_prompt = PERSON_SYSTEM_PROMPT
        question = (f"Is this a photograph of a person (identity already confirmed "
                    f"by filename: {description})?")
        if era:
            question = (f"The case is from {era}. A photograph from BEFORE then is "
                        f"expected and fine; only one clearly taken AFTER it is wrong. ") + question
    else:
        system_prompt = SYSTEM_PROMPT
        question = f"Does this image actually show: {description}"

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
        if "matches" not in result:
            # Defaulting a missing verdict to True is how this check spent a
            # week accepting everything: a duplicate SCHEMA assignment shadowed
            # the relevance schema with the safety one, so every answer came
            # back as safe/period_ok/text_ok, "matches" was never present, and
            # the default said yes -- while the model's own reason field read
            # "it does not depict a giraffe in any way".
            return True, f"verifier returned no verdict ({sorted(result)}) -- accepted", response.usage
        matches, reason = bool(result["matches"]), result.get("reason", "")
        # Only a real verdict is cached. The fail-open paths below and above
        # return "accepted without check", and storing that would make one
        # transient outage permanently bless whatever it touched.
        if cache_key:
            _relevance_put(cache_key, matches, reason)
        return matches, reason, response.usage
    except Exception as exc:
        return True, f"verification failed ({exc}) -- accepted without check", None
