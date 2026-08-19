import json
import re
import shutil
import time
from pathlib import Path
from urllib.parse import quote, unquote

import requests

from config import CASES_DIR, IMAGE_VERIFY_MODEL

USER_AGENT = "TrueCrimeContentPipeline/0.1 (educational/personal project; local dev; contact: set-an-email-here)"
HEADERS = {"User-Agent": USER_AGENT}
REQUEST_DELAY_SECONDS = 1.0

MUGSHOT_KEYWORDS = [
    "mugshot", "mug shot", "mug-shot", "booking photo", "arrest photo", "police photo",
]

PD_LICENSE_MARKERS = [
    "public domain", "pd-old", "pd-us", "pd-usgov", "cc0", "no known copyright", "cc-pd",
]

# Restrict to real photo formats. Wikimedia's imageinfo "mime" for scanned
# books/documents (djvu, pdf) and vector art (svg) also starts with "image/",
# so a naive prefix check lets huge multi-page scans through as if they were
# ordinary photos -- one such file hung ffmpeg for 45+ minutes.
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024  # 15 MB -- larger is almost never a simple photo

INSTITUTION_KEYWORDS = [
    "university", "college", "school", "court", "county", "prison", "station",
    "house", "chair", "jury", "department", "bus", "building", "hospital",
    "indictment", "district", "circuit", "farrowing", "sorority", "tipline",
]

_NAME_PARTICLES = {"jr.", "sr.", "of", "the", "de", "van", "der"}
_ROLE_PREFIXES = {
    "officer", "detective", "deputy", "sheriff", "captain", "capt.", "sergeant",
    "sgt.", "lieutenant", "lt.", "judge", "attorney", "dr.", "dr", "professor",
    "witness", "victim", "suspect",
}

# Wikimedia/Internet Archive results frequently surface files from these
# unrelated collections when a query word coincidentally matches (e.g. a
# victim's surname matching a Civil War soldier's name in a muster roll).
# None of our queries are ever actually about these topics, so any candidate
# whose title falls in one of these categories is rejected outright.
IRRELEVANT_CATEGORY_KEYWORDS = [
    "census", "genealogy", "ancestry", "family tree", "pedigree",
    "muster roll", "military record", "regiment", "cavalry", "infantry",
    "civil war", "war record", "enlistment", "draft card", "gravestone",
    "cemetery record", "parish record", "birth certificate", "marriage record",
]

# Words too generic to prove a candidate actually matches a query on their
# own -- e.g. "photo" alone matched an unrelated defense.gov news photo for
# the query "handcuffs evidence photo".
GENERIC_QUERY_WORDS = {"photo", "photograph", "image", "picture", "evidence", "scene"}

# Case briefs tend to give a person's full formal name ("Theodore Robert
# Bundy"), but real archive photo titles almost always use the common name
# ("Ted Bundy") -- without this, a strict text match rejects every real
# photo of a well-documented public figure just because "Theodore"/"Robert"
# never literally appear in any file title.
_NICKNAME_ALIASES = {
    "theodore": {"ted", "theo"}, "robert": {"rob", "bob", "bobby", "robbie"},
    "william": {"bill", "billy", "will"}, "richard": {"rick", "dick", "ricky"},
    "elizabeth": {"liz", "beth", "betty", "eliza", "lisa"}, "james": {"jim", "jimmy", "jamie"},
    "michael": {"mike", "mikey"}, "charles": {"charlie", "chuck"},
    "margaret": {"maggie", "meg", "peggy", "margie"}, "katherine": {"kate", "katie", "kathy", "catherine"},
    "kimberly": {"kim"}, "patricia": {"pat", "patty", "trish"},
    "deborah": {"deb", "debbie"}, "susan": {"sue", "susie"},
    "kenneth": {"ken", "kenny"}, "donald": {"don", "donnie"},
    "edward": {"ed", "eddie", "ted"}, "anthony": {"tony"},
    "gerald": {"gerry", "jerry"}, "daniel": {"dan", "danny"},
    "thomas": {"tom", "tommy"}, "christopher": {"chris"},
    "nancy": {"nan"}, "cynthia": {"cindy"}, "carolyn": {"carol"},
    "caroline": {"carol"}, "jennifer": {"jen", "jenny"},
}


def _alias_group(token: str) -> set:
    return {token} | _NICKNAME_ALIASES.get(token, set())


def _looks_like_person_name(query: str, known_people: set) -> bool:
    """Heuristic: is this visual_query naming a real, identifiable person (as
    opposed to an object, evidence item, or institution)? Used to keep AI
    image generation away from fabricating a specific person's likeness --
    those scenes are left for manual sourcing instead."""
    q = query.strip()
    if q in known_people:
        return True
    ql = q.lower()
    # "<Known Name> mugshot"/"<Known Name> photo" still names a specific real
    # person -- must not fall through to the generic-object branch below,
    # which would make it eligible for AI face generation if no real photo
    # is found. Check this before the "mugshot"/"photo" bailout.
    if any(ql.startswith(name.lower()) for name in known_people):
        return True
    if "mugshot" in ql or "evidence" in ql or "photo" in ql:
        return False
    if any(kw in ql for kw in INSTITUTION_KEYWORDS):
        return False
    if not q[:1].isupper():
        return False
    if any(ch.isdigit() for ch in q):
        return False
    words = q.split()
    if not (1 <= len(words) <= 4):
        return False
    capitalized = sum(
        1 for w in words if w[:1].isupper() or w.lower().strip(".") in _NAME_PARTICLES
    )
    return capitalized == len(words)


def _case_dir(case_id: str) -> Path:
    d = CASES_DIR / case_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _media_dir(case_id: str, subfolder: str) -> Path:
    d = _case_dir(case_id) / "media" / subfolder
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_script(case_id: str) -> dict:
    path = _case_dir(case_id) / "script.json"
    if not path.exists():
        raise RuntimeError(f"script.json not found for case {case_id} -- run the script stage first")
    return json.loads(path.read_text(encoding="utf-8"))


SUBCLIP_SECONDS = 3.0  # keep in sync with agents/video_assembly.py
# Piper's actual speech rate, measured against real audio_manifest.json
# durations (was hardcoded to an assumed 2.3 words/sec, which consistently
# underestimated real duration and caused far too few frames to be
# generated -- see the words/2.79 recalibration below).
WORDS_PER_SECOND = 2.79
# Generous headroom above the exact estimate: better to generate a couple of
# unused frames (cheap, local, ~0.2s each) than to run out and cycle/repeat
# through too few images across a scene's actual 3-second sub-clips.
MAX_AI_FRAMES_PER_SCENE = 24


def _estimate_seconds(text: str) -> float:
    return len((text or "").split()) / WORDS_PER_SECOND


def _frames_needed(duration: float) -> int:
    # +1 sub-clip of headroom: the estimate is still approximate, and it's
    # cheap to generate one extra frame versus falling short and repeating.
    return max(1, min(MAX_AI_FRAMES_PER_SCENE, round(duration / SUBCLIP_SECONDS) + 1))


# With anchors placed every ~12-15 words, a query covers ~1-2 cuts; the cap
# only catches outliers, and staying low keeps variety coming from new
# subjects rather than repeated takes of one.
MAX_FRAMES_PER_QUERY = 3

_ERA_IN_TEXT = re.compile(r"\b(1[89]\d0s|1[89]\d\d|20\d0s|20\d\d)\b")


def _case_era(brief: dict) -> str:
    """The decade the case belongs to, e.g. "1970s", taken from the years in
    its timeline. Used to keep generated scenes in period."""
    years = []
    for entry in brief.get("timeline", []) or []:
        years += [int(y) for y in re.findall(r"\b(1[89]\d\d|20\d\d)\b", str(entry.get("date", "")))]
    if not years:
        return ""
    years.sort()
    return f"{(years[len(years) // 2] // 10) * 10}s"


_PEOPLE_WORDS = re.compile(
    r"\b(people|person|men|man|women|woman|boys?|girls?|child(ren)?|kids?|"
    r"crowd|crowded|commuters?|passengers?|pedestrians?|shoppers?|queue|line of|"
    r"teacher|student|pupils?|classroom|workers?|miners?|soldiers?|officers?|"
    r"police|detectives?|investigators?|witness(es)?|suspects?|prisoners?|"
    r"couple|wedding|bride|groom|guests?|parents?|mothers?|fathers?|family|"
    r"funeral|mourners?|onlookers?|bystanders?|figures?|silhouettes?|"
    r"custody|arrested|waiting|walking|standing)\b", re.I)


def _wants_people(query: str) -> bool:
    """Does this query describe a scene with people in it?

    Such a query used to be unfulfillable: the archive rarely has the photo,
    and AI generation refused every frame with a person, so the query burned
    five generations and produced nothing -- 69% of everything abandoned in
    the Chikatilo run. These are now rendered as anonymous figures seen from
    behind, which is what a documentary does with them anyway. Named real
    people are handled elsewhere and never reach this."""
    return bool(_PEOPLE_WORDS.search(query))


def _with_era(prompt: str, era: str) -> str:
    """Add the period to a prompt that doesn't state one. Without it the
    model renders the present day -- a flat-screen television appeared over
    a 1974 fireplace because the query was just "Elizabeth Kloepfer
    fireplace"."""
    if not era or _ERA_IN_TEXT.search(prompt):
        return prompt
    return f"{prompt}, {era}"


def _frames_per_query(scene_text: str, anchors: list, n_queries: int,
                      duration: float, target_frames: int) -> list:
    """Frames to source for each query, sized to how long the narration
    actually dwells on it. Uses the video assembly's own anchor windows so
    generation and playback agree; falls back to an even split when anchors
    are missing or unusable."""
    if n_queries <= 0:
        return []
    even = max(1, round(target_frames / n_queries))
    try:
        from agents.video_assembly import _query_time_windows
        windows = _query_time_windows(scene_text, anchors, n_queries, duration)
    except Exception:
        return [even] * n_queries
    return [max(1, min(MAX_FRAMES_PER_QUERY, round((end - start) / SUBCLIP_SECONDS)))
            for start, end in windows]


def _is_mugshot_query(text: str) -> bool:
    lowered = (text or "").lower()
    return any(kw in lowered for kw in MUGSHOT_KEYWORDS)


def _is_pd_license(license_text: str) -> bool:
    lowered = (license_text or "").lower()
    return any(marker in lowered for marker in PD_LICENSE_MARKERS)


def _get_json(url: str, params: dict, retries: int = 3) -> dict | None:
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        except requests.RequestException:
            return None
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 3 * (attempt + 1)))
            time.sleep(wait)
            continue
        try:
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            return None
    return None


def _search_wikimedia_commons(query: str, limit: int = 5) -> list[dict]:
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,  # File namespace
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime",
        "format": "json",
    }
    data = _get_json(url, params)
    if not data:
        return []

    pages = data.get("query", {}).get("pages", {}) or {}
    candidates = []
    for page in pages.values():
        imageinfo = page.get("imageinfo")
        if not imageinfo:
            continue
        info = imageinfo[0]
        mime = info.get("mime", "")
        if mime not in ALLOWED_IMAGE_MIME_TYPES:
            continue
        extmeta = info.get("extmetadata", {})
        license_short = extmeta.get("LicenseShortName", {}).get("value", "")
        usage_terms = extmeta.get("UsageTerms", {}).get("value", "")
        license_text = f"{license_short} {usage_terms}".strip()
        if not _is_pd_license(license_text):
            continue
        title = page.get("title", "").replace("File:", "")
        candidates.append({
            "source": "wikimedia_commons",
            "title": title,
            "url": info.get("url"),
            "page_url": info.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{quote(page.get('title', ''))}",
            "license": license_text or "public domain (unspecified)",
        })
    return candidates


def _search_internet_archive(query: str, limit: int = 5) -> list[dict]:
    url = "https://archive.org/advancedsearch.php"
    params = {
        "q": f"{query} AND mediatype:(image)",
        "fl[]": ["identifier", "title", "licenseurl"],
        "rows": limit,
        "output": "json",
    }
    data = _get_json(url, params)
    if not data:
        return []

    docs = data.get("response", {}).get("docs", []) or []
    candidates = []
    for doc in docs:
        license_url = doc.get("licenseurl", "") or ""
        if not _is_pd_license(license_url) and "publicdomain" not in license_url.lower():
            continue
        identifier = doc.get("identifier")
        if not identifier:
            continue
        candidates.append({
            "source": "internet_archive",
            "title": doc.get("title", identifier),
            "url": f"https://archive.org/download/{identifier}/{identifier}.jpg",
            "page_url": f"https://archive.org/details/{identifier}",
            "license": license_url or "public domain",
        })
    return candidates


def _search_library_of_congress(query: str, limit: int = 5) -> list[dict]:
    """Library of Congress photo archive (loc.gov JSON API, no key). Strong
    on 1900s-1980s press photography -- courthouses, police, street scenes."""
    data = _get_json("https://www.loc.gov/photos/", {
        "q": query, "fo": "json", "c": limit,
    })
    if not data:
        return []
    candidates = []
    for item in (data.get("results") or [])[:limit]:
        # LoC "photos" results are overwhelmingly PD (federal / pre-1929 /
        # no known restrictions); skip anything explicitly restricted.
        rights = " ".join(str(r) for r in item.get("rights", []) or []).lower()
        if "restrict" in rights:
            continue
        urls = item.get("image_url") or []
        if not urls:
            continue
        candidates.append({
            "source": "library_of_congress",
            "title": item.get("title", ""),
            "url": urls[-1],  # last entry is the largest rendition
            "page_url": item.get("id", ""),
            "license": "public domain (LoC)",
        })
    return candidates


def _wikipedia_find_articles(query: str, limit: int = 2) -> list[str]:
    """Resolve a search phrase to real Wikipedia article titles. The brief's
    own source list can't be relied on to cite Wikipedia (it often doesn't),
    so the image pool is built by searching for the case/subject directly."""
    data = _get_json("https://en.wikipedia.org/w/api.php", {
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": limit, "format": "json",
    })
    if not data:
        return []
    return [hit["title"] for hit in (data.get("query", {}).get("search", []) or [])]


def _search_wikipedia_article_images(article_title: str, limit: int = 30) -> list[dict]:
    """Images already curated into the case's Wikipedia article -- the
    highest-relevance real photos that exist (mugshots, locations, trial).
    Licenses are verified per-file via the Commons imageinfo API."""
    data = _get_json("https://en.wikipedia.org/w/api.php", {
        "action": "query", "titles": article_title, "generator": "images",
        "gimlimit": limit, "prop": "imageinfo",
        "iiprop": "url|extmetadata", "format": "json",
    })
    if not data:
        return []
    candidates = []
    for page in (data.get("query", {}).get("pages", {}) or {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("url", "")
        if not url or not url.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        meta = info.get("extmetadata", {}) or {}
        license_short = (meta.get("LicenseShortName", {}) or {}).get("value", "")
        if not _is_pd_license(license_short):
            continue
        candidates.append({
            "source": "wikipedia_article",
            # Strip the "File:" namespace prefix -- left on, it fuses with the
            # first word of the filename ("File:Lynda" -> "filelynda") and no
            # name match can ever succeed.
            "title": page.get("title", "").replace("File:", ""),
            "url": url,
            "page_url": info.get("descriptionurl", url),
            "license": license_short,
        })
    return candidates


def _cache_path_for(url: str) -> Path:
    """Cache slot for a source URL. The pipeline is re-run many times per
    case and each run wipes media/ and re-fetches the same archive photos --
    enough traffic that Wikimedia starts answering 429 and real victim
    photos silently vanish from the video. Cached bytes survive reruns."""
    import hashlib
    cache_dir = CASES_DIR.parent / "media_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}{Path(url.split('?')[0]).suffix[:6]}"


def _mirror_urls(url: str) -> list[str]:
    """Alternate URLs for the same file. upload.wikimedia.org throttles
    repeat traffic with 429 for long stretches, while the same file served
    through commons' Special:FilePath keeps working -- without this fallback
    a throttled window silently strips every real photo from the video."""
    urls = [url]
    if "upload.wikimedia.org" in url:
        filename = url.rstrip("/").split("/")[-1]
        urls.append(f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}")
    return urls


def _download(url: str, dest: Path, retries: int = 3) -> bool:
    cached = _cache_path_for(url)
    if cached.exists() and cached.stat().st_size > 0:
        shutil.copy2(cached, dest)
        return True
    for candidate_url in _mirror_urls(url):
        if _download_once(candidate_url, dest, retries):
            try:
                shutil.copy2(dest, _cache_path_for(url))
            except OSError:
                pass  # cache is an optimization, never fail the download for it
            return True
    return False


def _download_once(url: str, dest: Path, retries: int) -> bool:
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30, stream=True)
            # Wikimedia rate-limits bursts of image downloads with 429. Treated
            # as a plain error this silently dropped real victim photos and
            # pushed those people into the manual-sourcing queue -- back off
            # and retry instead.
            if resp.status_code == 429:
                resp.close()
                if attempt < retries:
                    time.sleep(float(resp.headers.get("Retry-After", 2 * (attempt + 1))))
                    continue
                return False
            resp.raise_for_status()
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
                return False
            written = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    written += len(chunk)
                    if written > MAX_DOWNLOAD_BYTES:
                        f.close()
                        dest.unlink(missing_ok=True)
                        return False
                    f.write(chunk)
            time.sleep(REQUEST_DELAY_SECONDS)  # stay under Wikimedia's rate limit
            return True
        except requests.RequestException:
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return False
    return False


def _extension_for(url: str) -> str:
    match = re.search(r"\.(jpg|jpeg|png|gif|webp|tiff)(\?.*)?$", url, re.IGNORECASE)
    return match.group(1).lower() if match else "jpg"


def _verify(path: Path, query: str, is_person: bool = False):
    """Vision check that a downloaded candidate actually depicts the query
    subject -- catches keyword-coincidence false positives (a "crowbar"
    electronics diagram, a "Beetle" insect stamp) that text matching alone
    can't. Returns (matches, reason, usage)."""
    from agents.image_verifier import verify_image
    return verify_image(str(path), query, is_person=is_person)


def _name_tokens(query: str) -> list[str]:
    """Significant name words to require a match on -- strips role prefixes
    ("Officer", "Detective", ...) and particles ("of", "van", ...) so a
    two-word name like "Officer David Lee" requires both "david" and "lee"
    in the candidate title, not just the common surname "lee" alone."""
    words = [w.strip(".") for w in query.split()]
    return [w.lower() for w in words if w.lower() not in _NAME_PARTICLES and w.lower() not in _ROLE_PREFIXES]


def _tokens_close(a: str, b: str) -> bool:
    """True if two name words match exactly, via a nickname alias, or differ
    by a single edit. Case briefs and archive captions routinely disagree by
    one letter on uncommon given names ("Georgeann" vs "Georgann"), which
    otherwise blocks a genuine photo of that person."""
    if a == b or b in _alias_group(a) or a in _alias_group(b):
        return True
    if min(len(a), len(b)) < 5 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(long)):  # one deletion from the longer word
        if long[:i] + long[i + 1:] == short:
            return True
    return False


def _name_appears_in_title(core_tokens: list, title_lower: str) -> bool:
    """The name's first and last word must appear in the title in order and
    close together (at most one word between them, leaving room for a
    dropped middle name) -- loose 'each word somewhere in the title'
    matching once let one title naming two different people pass."""
    # Split on any non-word run, not just spaces: real archive titles glue
    # words with punctuation ("bundy_1975", "healy-photo.jpg").
    title_words = [w for w in re.split(r"[^\w']+", title_lower) if w]
    if len(core_tokens) == 1:
        return any(_tokens_close(core_tokens[0], w) for w in title_words)
    first, last = core_tokens[0], core_tokens[-1]
    for i, w in enumerate(title_words):
        if not _tokens_close(first, w):
            continue
        for j in range(i + 1, min(i + 3, len(title_words))):
            if _tokens_close(last, title_words[j]):
                return True
    return False


def _rank_candidates(query: str, candidates: list[dict], known_people: set | None = None) -> list[dict]:
    """Filter out candidates that don't actually match the query, then sort
    the survivors so the closest title match comes first. The search APIs
    rank by their own relevance heuristics, which frequently put an unrelated
    file ahead of (or instead of rejecting) the one that actually matches --
    e.g. a genealogy census record surfacing for a victim's surname."""
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 2]
    is_person = bool(known_people) and _looks_like_person_name(query, known_people)
    name_tokens = _name_tokens(query) if is_person else None

    survivors = []
    for c in candidates:
        title_lower = c["title"].lower()

        if any(kw in title_lower for kw in IRRELEVANT_CATEGORY_KEYWORDS):
            continue

        if is_person:
            # Match on the first and last name words only (with common
            # nickname aliases) -- middle names/formal given names in a case
            # brief ("Theodore Robert Bundy") are almost never how real
            # archive photos are titled ("Ted Bundy mug shot.jpg"), so
            # requiring every token would reject genuine matches. The words
            # must still appear together with at most one word between them
            # (room for a dropped middle name/initial) -- requiring each
            # word to merely appear *somewhere* in the title is what let
            # "Officer David Lee" match a title naming two unrelated people,
            # "David Fluker" and "Bradley Lee", each occurring separately.
            # Strip trailing descriptor words ("mugshot", "photo", ...) before
            # picking the first/last name token -- otherwise a query like
            # "Theodore Robert Bundy mugshot" would treat "mugshot" as the
            # surname instead of "Bundy".
            pure_name_tokens = [t for t in name_tokens if t not in GENERIC_QUERY_WORDS and "mugshot" not in t and "mug" != t and "shot" != t]
            pure_name_tokens = pure_name_tokens or name_tokens
            core_tokens = [pure_name_tokens[0], pure_name_tokens[-1]] if len(pure_name_tokens) > 1 else pure_name_tokens
            if not _name_appears_in_title(core_tokens, title_lower):
                continue
        else:
            # Generic words like "photo" or "evidence" are too common to
            # count as a real match on their own (that's how "handcuffs
            # evidence photo" matched an unrelated defense.gov news photo
            # via the word "photo" alone) -- require most of the
            # *meaningful* words to actually appear in the title.
            meaningful_words = [w for w in query_words if w not in GENERIC_QUERY_WORDS]
            words_to_check = meaningful_words or query_words
            missing_words = sum(1 for w in words_to_check if w not in title_lower)
            if words_to_check and missing_words > len(words_to_check) // 2:
                continue

        survivors.append(c)

    # Surnames of other people in the case: a title naming someone else
    # ("Caryn campbell ted bundy.jpg") is a photo OF that other person, so it
    # must rank below one where the queried person is the actual subject.
    other_surnames = set()
    if known_people:
        queried = set(name_tokens or [])
        for person in known_people:
            surname = person.split()[-1].lower()
            if len(surname) > 3 and not any(_tokens_close(surname, t) for t in queried):
                other_surnames.add(surname)

    def score(c: dict) -> tuple:
        title_lower = c["title"].lower()
        title_words = [w for w in re.split(r"[^\w']+", title_lower) if w]
        foreign_subject = 1 if any(
            any(_tokens_close(s, w) for w in title_words) for s in other_surnames
        ) else 0
        # Prefer titles that lead with the queried subject over ones where it
        # trails after someone/something else.
        leads = 0 if (name_tokens and title_words
                      and _tokens_close(name_tokens[0], title_words[0])) else 1
        exact = 0 if query_lower in title_lower else 1
        missing_words = sum(1 for w in query_words if w not in title_lower)
        return (foreign_subject, leads, exact, missing_words)

    return sorted(survivors, key=score)


# Filled once per run() from the case's Wikipedia article(s): every image the
# article curates, license-checked. These are the most case-relevant real
# photos available, so they compete in ranking for every query.
_WIKI_IMAGE_POOL: list[dict] = []


def _has_proper_noun(query: str) -> bool:
    """True if the query names something specific (a person, place or
    institution with capitalized words), as opposed to a generic object
    ("handcuffs evidence photo")."""
    return any(w[:1].isupper() and len(w) > 2 and not w.isupper() for w in query.split())


def _search_all(query: str, known_people: set | None = None) -> list[dict]:
    # Generic object/scene queries must NOT hit the public search APIs:
    # keyword coincidence returns real photos from unrelated events (an
    # Abu Ghraib photo for "handcuffs", an 1898 Taiwanese document for
    # "execution warrant") -- factually foreign material that is worse than
    # an AI fallback. Generic queries may only match the case's own curated
    # Wikipedia-article pool; everything else goes to AI generation.
    is_person = bool(known_people) and _looks_like_person_name(query, known_people)
    if not is_person and not _has_proper_noun(query):
        return _rank_candidates(query, list(_WIKI_IMAGE_POOL), known_people)

    candidates = _search_wikimedia_commons(query)
    time.sleep(REQUEST_DELAY_SECONDS)
    if not candidates:
        candidates = _search_library_of_congress(query)
        time.sleep(REQUEST_DELAY_SECONDS)
    if not candidates:
        candidates = _search_internet_archive(query)
        time.sleep(REQUEST_DELAY_SECONDS)

    # visual_query is sometimes a comma-separated list of ideas rather than one
    # searchable phrase -- the full string then matches nothing. Retry with just
    # the first clause, which is usually the most specific one.
    if not candidates and "," in query:
        first_clause = query.split(",")[0].strip()
        candidates = _search_wikimedia_commons(first_clause)
        time.sleep(REQUEST_DELAY_SECONDS)
        if not candidates:
            candidates = _search_internet_archive(first_clause)
            time.sleep(REQUEST_DELAY_SECONDS)

    # A case brief's full formal name ("Theodore Robert Bundy mugshot") often
    # returns nothing -- Wikimedia's own search seems to lose recall past a
    # handful of terms, and real archive titles use short/common names
    # anyway. Retry with just the last two words (surname + any trailing
    # descriptor like "mugshot"), which is much closer to how files are
    # actually titled.
    if not candidates and len(query.split()) > 2:
        short_query = " ".join(query.split()[-2:])
        candidates = _search_wikimedia_commons(short_query)
        time.sleep(REQUEST_DELAY_SECONDS)
        if not candidates:
            candidates = _search_internet_archive(short_query)
            time.sleep(REQUEST_DELAY_SECONDS)

    # The article pool always competes: its images are pre-curated for this
    # exact case, so ranking decides per query whether one of them fits
    # better than generic search hits.
    seen = {c["url"] for c in candidates}
    candidates += [c for c in _WIKI_IMAGE_POOL if c["url"] not in seen]

    return _rank_candidates(query, candidates, known_people)


def _resolve_query(case_id, part_number, scene_index, q_index, query, known_people,
                   frames_wanted, next_index_for_query, db, fallback_query: str = "",
                   era: str = ""):
    """Resolve a single visual_query within a scene. Returns a dict:
      {query, kind, status, frames, review_frames, note}
    - kind: "person" | "object_or_location"
    - status: "found" (>=1 real photo), "ai_generated" (only AI frames),
              "needs_review" (a mugshot pending human approval),
              "manual_person" (named person, no photo, no AI faces allowed),
              "unresolved" (nothing usable)
    - frames: approved, auto-playable frame paths (real verified + AI)
    - review_frames: mugshot frames staged for manual review (NOT auto-played)
    """
    tag = f"part{part_number}_scene{scene_index}_q{q_index}"
    is_person = _looks_like_person_name(query, known_people)
    candidates = _search_all(query, known_people)

    result = {"query": query, "kind": "person" if is_person else "object_or_location",
              "status": "unresolved", "frames": [], "review_frames": [], "note": ""}

    # A person query should show the person NOW, not sit in a review queue:
    # when the query itself isn't asking for a mugshot, prefer non-mugshot
    # candidates (yearbook photo, press photo, FBI poster) so the face
    # auto-plays; only fall into the review path when every candidate is
    # mugshot-titled. (Previously a mugshot merely ranking first sent the
    # whole query to review and the video shipped with no photo at all.)
    if candidates and not _is_mugshot_query(query):
        non_mug = [c for c in candidates if not _is_mugshot_query(c["title"])]
        if non_mug:
            candidates = non_mug + [c for c in candidates if _is_mugshot_query(c["title"])]

    needs_review = _is_mugshot_query(query) or (candidates and _is_mugshot_query(candidates[0]["title"]))

    if needs_review:
        # Mugshots never auto-play -- stage the first candidate that passes the
        # vision check into the review folder for manual approval.
        for cand in candidates:
            ext = _extension_for(cand["url"])
            dest_path = _media_dir(case_id, "review") / f"{tag}.{ext}"
            if not _download(cand["url"], dest_path):
                continue
            matches, _reason, usage = _verify(dest_path, query, is_person=True)
            if usage:
                db.log_usage(case_id, "archive_verify", IMAGE_VERIFY_MODEL, usage.input_tokens, usage.output_tokens)
            if matches:
                result["status"] = "needs_review"
                result["review_frames"] = [str(dest_path)]
                result["source"] = cand["source"]
                result["title"] = cand["title"]
                result["license"] = cand["license"]
                result["page_url"] = cand["page_url"]
                result["note"] = "mugshot of a real person -- pending manual review"
                return result
            dest_path.unlink(missing_ok=True)
        # No mugshot candidate passed; fall through to treat as a person needing manual sourcing.
        result["status"] = "manual_person"
        result["note"] = "no public-domain mugshot found -- manual sourcing (no AI faces)"
        return result

    # Download distinct, relevance-verified real photos, rotating the start
    # point so a repeated query yields different photos across scenes.
    frames = []
    if candidates:
        idx = next_index_for_query.get(query, 0) % len(candidates)
        next_index_for_query[query] = idx + 1
        ordered = candidates[idx:] + candidates[:idx]
        # Cap how many candidates we download+verify per query. Object/location
        # queries fall back to (free, local) AI generation anyway, so there's
        # no point spending many network+LLM round-trips chasing a few extra
        # real photos -- this keeps the stage from ballooning now that each
        # scene has several queries.
        attempts_left = frames_wanted + 2
        for cand in ordered:
            if len(frames) >= frames_wanted or attempts_left <= 0:
                break
            attempts_left -= 1
            ext = _extension_for(cand["url"])
            dest_path = _media_dir(case_id, "accepted") / f"{tag}_{len(frames)}.{ext}"
            if not _download(cand["url"], dest_path):
                continue
            matches, _reason, usage = _verify(dest_path, query, is_person=is_person)
            if usage:
                db.log_usage(case_id, "archive_verify", IMAGE_VERIFY_MODEL, usage.input_tokens, usage.output_tokens)
            if not matches:
                dest_path.unlink(missing_ok=True)
                continue
            frames.append(str(dest_path))

    from agents.image_generator import generate_image
    from agents.image_verifier import SafetyCheckUnavailable, ai_frame_verdict

    def vision_gate(path: str, people: bool = False) -> dict:
        try:
            verdict, usage = ai_frame_verdict(path, era=era, people_allowed=people)
        except SafetyCheckUnavailable as exc:
            raise RuntimeError(
                "AI frame safety check is unavailable, so generated frames cannot be "
                f"cleared for use -- stopping instead of shipping unchecked images.\n  cause: {exc}"
            ) from exc
        if usage:
            db.log_usage(case_id, "ai_frame_safety", IMAGE_VERIFY_MODEL,
                         usage.input_tokens, usage.output_tokens)
        # Only the hard verdict is announced as a rejection. A soft flag is
        # not a rejection -- generate_image decides whether a cleaner re-roll
        # turns up, and says so itself if it settles for the flagged frame.
        if not verdict["safe"]:
            print(f"    vision gate rejected a frame: {verdict['reason'][:80]}", flush=True)
        return verdict

    def generate(prompt: str, wanted: int, start_index: int = 0):
        # Decided from the query text before the era is appended -- the era is
        # a bare year and carries no hint of who is in the shot.
        people = _wants_people(prompt)
        prompt = _with_era(prompt, era)
        for ai_idx in range(start_index, wanted):
            dest_path = _media_dir(case_id, "ai_generated") / f"{tag}_ai{ai_idx}.png"
            if generate_image(prompt, dest_path, variant=ai_idx,
                              vision_gate=vision_gate, people=people):
                frames.append(str(dest_path))

    if is_person:
        # Named real person: never AI-generate a face. Use their real photos
        # if any were found; otherwise render the beat's own setting from the
        # scene's fallback. Previously this returned nothing and the assembly
        # filled the gap with an image already used elsewhere -- which is why
        # a list of five victims looked like the same two pictures repeating.
        if frames:
            result["status"] = "found"
            result["frames"] = frames
            return result
        if fallback_query:
            generate(fallback_query, frames_wanted)
        if frames:
            result["status"] = "ai_fallback"
            result["frames"] = frames
            result["note"] = f"no public-domain photo -- rendered the setting ({fallback_query[:60]})"
        else:
            result["status"] = "manual_person"
            result["note"] = "no public-domain photo found -- manual sourcing (no AI faces)"
        return result

    # Object/location: fill the query's share of the scene's ~3s slots with
    # AI frames, so a long window gets a fresh image at every cut instead of
    # one photo held under a pan. Each frame is generated with a different
    # shot direction (see SHOT_MODIFIERS) so they read as separate shots of
    # the beat rather than repeated takes of the same composition.
    if len(frames) < frames_wanted:
        generate(query, frames_wanted, start_index=len(frames))
    # Still short (generation kept failing the safety gate)? Fill the rest
    # from the scene's fallback rather than leaving the window to repeat.
    if len(frames) < frames_wanted and fallback_query:
        generate(fallback_query, frames_wanted, start_index=len(frames))

    if frames:
        result["frames"] = frames
        result["status"] = "found" if any("accepted" in f for f in frames) else "ai_generated"
    else:
        result["status"] = "unresolved"
        result["note"] = "no public-domain candidate and AI generation failed"
    return result


def _parts_limit(parts):
    """Optional dev-iteration cap: PIPELINE_MAX_PARTS=N processes only the
    first N parts through the heavy stages (script still writes all parts)."""
    import os
    try:
        n = int(os.environ.get("PIPELINE_MAX_PARTS", "0"))
    except ValueError:
        n = 0
    if n > 0:
        kept = [p for p in parts if p.get("part_number", 0) <= n]
        if kept and len(kept) < len(parts):
            print(f"  PIPELINE_MAX_PARTS={n}: processing only part(s) " +
                  ", ".join(str(p["part_number"]) for p in kept), flush=True)
            return kept
    return parts


def run(case_id: str, db) -> None:
    script = _load_script(case_id)

    # Every rerun writes a brand-new media_manifest.json from scratch, but
    # without this, files from earlier runs (different scene counts/queries
    # after script edits) stay on disk under the same media/ subfolders --
    # orphaned images no longer referenced by any manifest, but still
    # visible and confusing when browsing the folder directly.
    for subfolder in ("accepted", "review", "ai_generated"):
        shutil.rmtree(_media_dir(case_id, subfolder), ignore_errors=True)

    brief_path = _case_dir(case_id) / "brief.json"
    known_people = set()
    _WIKI_IMAGE_POOL.clear()
    if brief_path.exists():
        brief = json.loads(brief_path.read_text(encoding="utf-8"))
        known_people = {p["name"] for p in brief.get("key_people", [])}
        case_era = _case_era(brief)
        if case_era:
            print(f"  case era: {case_era} (added to generated scenes that don't state one)", flush=True)

        # Build the curated image pool from the case's Wikipedia article(s).
        # Sources cited in the brief are used when present, but they often
        # don't include Wikipedia at all -- so the case title and the main
        # subject's name are searched directly as well. These articles carry
        # the case's real photos (victims, suspect, evidence, trial), which
        # is the difference between a victim being shown and going to the
        # manual-sourcing queue.
        articles = []
        for src in brief.get("sources", []) or []:
            url = src.get("url", "")
            if "wikipedia.org/wiki/" in url:
                articles.append(unquote(url.rsplit("/wiki/", 1)[-1]).replace("_", " "))

        search_terms = [brief.get("title", "")]
        search_terms += [p["name"] for p in brief.get("key_people", [])
                         if p.get("role") == "suspect"]
        for term in search_terms:
            if not term:
                continue
            articles += _wikipedia_find_articles(term)
            time.sleep(REQUEST_DELAY_SECONDS)

        seen_articles = set()
        for article in articles:
            if article in seen_articles:
                continue
            seen_articles.add(article)
            pool = _search_wikipedia_article_images(article)
            print(f"  wikipedia article pool: {article!r} -> {len(pool)} usable image(s)", flush=True)
            _WIKI_IMAGE_POOL.extend(pool)
            time.sleep(REQUEST_DELAY_SECONDS)

    items = []
    resolved_count = 0
    review_scene_count = 0
    manual_scene_count = 0
    unresolved_count = 0
    total_frames = 0
    next_index_for_query: dict[str, int] = {}

    for part in _parts_limit(script["parts"]):
        part_number = part["part_number"]
        for scene_index, scene in enumerate(part["scenes"]):
            queries = scene.get("visual_queries") or [scene.get("visual_query", "")]
            queries = [q for q in queries if q]

            duration = _estimate_seconds(scene.get("text", ""))
            target_frames = _frames_needed(duration)
            # How many ~3s slots each query actually occupies, using the same
            # anchor windows the video assembly cuts on -- a query the
            # narration dwells on gets several distinct frames instead of one
            # image held under a slow pan, while a passing mention gets one.
            anchors = scene.get("visual_anchors") or []
            frames_per_query = _frames_per_query(
                scene.get("text", ""), anchors, len(queries), duration, target_frames)

            item = {
                "part_number": part_number,
                "scene_index": scene_index,
                "visual_queries": queries,
                "visual_anchors": scene.get("visual_anchors") or [],
                "visual_fallbacks": scene.get("visual_fallbacks") or [],
                "scene_text": scene["text"],
                "queries": [],
                "local_paths": [],   # ordered, auto-playable frames across all queries
                "review_frames": [],
                "manual_people": [],
            }

            fallbacks = scene.get("visual_fallbacks") or []
            for q_index, query in enumerate(queries):
                print(f"  [p{part_number}s{scene_index} q{q_index}/{len(queries)}] {query[:50]}", flush=True)
                r = _resolve_query(case_id, part_number, scene_index, q_index, query,
                                   known_people, frames_per_query[q_index],
                                   next_index_for_query, db,
                                   fallback_query=fallbacks[q_index] if q_index < len(fallbacks) else "",
                                   era=case_era)
                item["queries"].append({k: r[k] for k in ("query", "kind", "status", "note")})
                # Only record frames that are actually on disk. A rejected
                # re-roll unlinks its file, and when that file's path was
                # already collected the manifest ends up pointing at nothing --
                # which the video stage then hits hours later, at the end of
                # the run, with no way to recover the frame.
                for frame in r["frames"]:
                    if Path(frame).is_file():
                        item["local_paths"].append(frame)
                    else:
                        print(f"    dropped a frame that is no longer on disk: {Path(frame).name}", flush=True)
                item["review_frames"].extend(r["review_frames"])
                if r["status"] == "manual_person":
                    item["manual_people"].append(query)

            total_frames += len(item["local_paths"])
            if item["local_paths"]:
                item["status"] = "resolved"
                resolved_count += 1
            else:
                item["status"] = "unresolved"
                unresolved_count += 1
            if item["review_frames"]:
                review_scene_count += 1
            if item["manual_people"]:
                manual_scene_count += 1

            items.append(item)

    manifest = {"case_id": script.get("case_id", case_id), "items": items}
    manifest_path = _case_dir(case_id) / "media_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    review_items = [i for i in items if i["review_frames"]]
    review_path = _case_dir(case_id) / "review_queue.json"
    review_path.write_text(json.dumps(review_items, ensure_ascii=False, indent=2), encoding="utf-8")

    manual_items = [i for i in items if i["manual_people"]]
    manual_path = _case_dir(case_id) / "manual_sourcing_queue.json"
    manual_path.write_text(json.dumps(manual_items, ensure_ascii=False, indent=2), encoding="utf-8")

    db.update_case_status(case_id, "archive_done")

    total = len(items)
    print(f"  media manifest written: {manifest_path}")
    print(f"  scenes: {total} | resolved: {resolved_count} | unresolved: {unresolved_count} "
          f"| total playable frames: {total_frames}")
    if review_scene_count:
        print(f"  review queue written: {review_path} -- {review_scene_count} scene(s) have a mugshot pending manual review")
    if manual_scene_count:
        print(f"  manual sourcing queue written: {manual_path} -- {manual_scene_count} scene(s) reference a named person with no public-domain photo")
