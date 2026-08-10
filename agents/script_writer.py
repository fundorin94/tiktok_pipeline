import json
import re

import anthropic

from config import ANTHROPIC_API_KEY, CASES_DIR, SCRIPT_MODEL

SYSTEM_PROMPT = """You are a scriptwriter for a true-crime short-form video series distributed \
as TikTok "Part 1, Part 2, ..." episodes. You receive a factual case brief (JSON: summary, \
timeline, key_details, key_people, sources) and turn it into a narration script split into \
parts.

Rules:
- Use ONLY facts present in the brief. Do not invent dialogue, quotes, dates, or details not \
given.
- Write in a direct, spoken narration style meant to be read aloud out loud by a narrator -- \
short sentences, consistent tense, no headers or bullet points inside the narration text.
- GO DEEP, don't skim. True-crime viewers stay for the unsettling specifics, not a Wikipedia \
summary. When the brief gives a psychologically revealing detail (an obsession with violent \
pornography, a rehearsed fake injury, charm used as a weapon, a double life), spend 2-4 \
sentences unpacking it: what exactly he did, how it escalated, what it looked like from the \
outside, what those around him missed. Prefer one vividly developed thread over three facts \
mentioned in passing -- but never invent details not supported by the brief.
- HARD CONSTRAINT: each part must be 280-420 words of narration (about 2-3 minutes spoken \
aloud).
- HARD CAP: the whole script must have AT MOST 6 parts. Never exceed 6, no matter how much \
material the brief contains. Aim for 5-6 parts for a well-documented case; use fewer only if \
the case genuinely has little material. This is a hard ceiling, not a soft target.
- Because of the 6-part cap, you will usually have more material than fits. Prioritize: keep \
the events and details that drive the narrative forward and create the strongest cliffhangers; \
cut or compress secondary details, minor timeline entries, and less pivotal people rather than \
trying to include everything. Do not pad a thin case with filler to reach 6 parts, and do not \
rush a rich case by cramming -- compress by choosing what matters, not by writing faster or \
skipping the word-count target.
- Every part except possibly the last must end on a genuine cliffhanger or unresolved \
question that makes the viewer want the next part -- an actual narrative beat (a discovery \
about to be made, a suspect about to be confronted), not a cheap "but that's not all" filler \
line.
- The 280-420 words/part and AT MOST 6 parts caps above are absolute and do not bend for \
  anything below -- not for splitting scenes more finely, not for including more violent detail, \
  not for covering more victims individually, not for anything else. Splitting into more/shorter \
  scenes must happen INSIDE the same word budget, not by writing more words. If following the \
  scene-granularity or content guidance below would push a part past 420 words or the script past \
  6 parts, cut material (fewer timeline entries, less granular coverage of minor figures) rather \
  than exceed either cap. A script that blows through these caps is a failed script regardless of \
  how well it follows every other instruction.
- Split the narration within each part into "scenes" of 1-3 sentences each.
- Each scene gets a "visual_queries" LIST -- an ordered list of the concrete things the \
  narration shows AS IT PROGRESSES through that scene. This is the most important rule: the \
  video shows a new image roughly every 3 seconds, cycling through this list in order, so the \
  list must have enough distinct entries to cover the scene without repeating one subject for \
  many seconds. HARD DENSITY RULE: one query per roughly 12-15 words of narration -- so a \
  60-word scene needs 4-5 queries, a 120-word scene needs 8-10. Never let one subject cover \
  more than ~6 seconds (~17 words): if a stretch of narration has no new concrete subject, find \
  another real thing in it to show (the place, the weather, a document, an object being used, \
  the view from where it happened). Two adjacent queries must never be the same subject from a \
  different angle -- "tan Volkswagen at night" then "Volkswagen from the front" is wrong; the \
  second should move on to what the narration moved on to.
  - If a sentence lists several items ("a search of the car revealed a ski mask, handcuffs, a \
    crowbar, rope, and an ice pick"), put EACH as its own query: ["ski mask", "handcuffs \
    evidence photo", "crowbar", "rope", "ice pick", "car interior with passenger seat removed"]. \
    Do not collapse a five-item list into one query -- the viewer should see each item named.
  - If a scene describes an action toward a victim, include BOTH the person and the setting/means \
    as separate queries in order, e.g. ["Lynda Ann Healy", "1970s student bedroom", "bloodstained \
    bedsheets crime scene"] -- so her photo actually appears AND the scene isn't just one static \
    portrait for 15 seconds.
- Each scene ALSO gets a "visual_fallbacks" LIST, exactly the same length as visual_queries. \
  Entry i is an AI-renderable scene used INSTEAD of query i when no real photo can be found for \
  it. It must be a NEW image specific to that beat -- never a person, never a repeat of another \
  query in the scene. For a person query it's where or how that moment happened: "Georgann \
  Hawkins" -> "narrow alley behind a 1974 university street at night, single porch light"; \
  "Carol DaRonch" -> "empty 1974 shopping mall corridor, closed storefronts". For an object \
  query it's a different real element of the same beat, not the same object again: "handcuffs \
  on a car seat" -> "car interior at night lit by a torch beam". These matter: without a \
  fallback the video repeats an image already used elsewhere, which is exactly what makes a \
  list of victims look like the same two pictures over and over.
- Each scene ALSO gets a "visual_anchors" LIST, exactly the same length as visual_queries: for \
  each query, a VERBATIM snippet (3-8 consecutive words, copied character-for-character from that \
  scene's "text") marking the moment the narration reaches that query's subject. The image \
  switches on screen exactly when the narrator says the anchor words, so each anchor must sit at \
  or just before the words that name the query's subject. The first anchor must be the scene \
  text's opening words. Anchors must appear in the same order as their queries and must not \
  repeat. Example -- text: "Police stopped the tan Volkswagen. Inside they found handcuffs and a \
  crowbar." queries: ["tan Volkswagen Beetle at night", "handcuffs on a car seat"] -> anchors: \
  ["Police stopped the tan", "Inside they found handcuffs"].
  - Order the queries to match the sentence order, so the image on screen tracks what's being \
    narrated at that moment.
- Each query in the list must be ONE of these three kinds -- nothing else:
  (a) A named person from key_people: just their full name, e.g. "Ted Bundy". For an arrest/booking \
  scene append "mugshot", e.g. "Ted Bundy mugshot". A real archive photo is searched for these; \
  if none is found the query is queued for manual photo sourcing rather than AI-generated -- never \
  fabricate a specific person's likeness. Because a person query may yield no photo, do NOT make a \
  scene's ONLY query a person who has no photo -- always pair a named victim with at least one \
  setting/object query in the same list, so the scene still has something to show.
  (b) A specific physical object or piece of evidence: a weapon, tool, vehicle, or item, as \
  concrete as the brief allows, e.g. "Volkswagen Beetle 1968", "handcuffs evidence photo", "ice \
  pick", "crowbar". Prefer to introduce a NEW object rather than repeat one already used in an \
  earlier scene -- a repeated object query with no new photo behind it is just the same picture \
  again.
  (c) A generic location or scene type, concrete and period-appropriate, e.g. "1970s courtroom \
  interior", "suburban house exterior 1974", "stone prison building exterior". These are \
  AI-generated. Do not fill a scene with only bare building exteriors -- a video that's mostly \
  generic buildings feels empty; use locations to support a person/object query, not replace it. \
  When the narration describes violence, an assault, or a crime scene, say so in the query \
  ("ransacked bedroom crime scene", "sorority room in disarray after attack", "bloodstained \
  bedsheets crime scene") -- a bare "sorority house interior" renders as a tidy ordinary room, \
  which doesn't match narration describing carnage. Whenever the narration is graphic, the \
  queries should reflect that intensity, not default to a tame establishing shot.
- Choose queries that actually match what the narration says at that point -- if a sentence is \
  about the perpetrator's own actions (his approach, arrest, trial, execution), include his name; \
  don't replace him with a nearby prop or building just to avoid naming him. Repeating the \
  central figure's name across scenes is fine (real photos rotate downstream); still name the \
  specific victim/investigator/witness when a beat is actually about them.
- Do not show a detail before the narration reaches it. A query must depict something the CURRENT \
  scene's sentences actually describe -- e.g. the removed passenger seat / the tools found in the \
  car belong ONLY to the arrest-and-search scene, not to an earlier scene about his method or his \
  car in general. Don't preview later evidence in an earlier scene just because it's related.
- HARD RULE: AI-generated queries (kinds (b) and (c)) must contain NO human beings and NO human \
  anatomy, in any form -- no person, figure, silhouette, crowd, hands, body, face, skull, teeth, \
  jaw, bite mark, mannequin, and no collage/grid of photos or yearbook page (those render as \
  walls of fake faces). The AI renders ONLY: locations/buildings, interiors, vehicles, evidence \
  objects, documents/newspapers, and landscapes. A human being appears in the video ONLY through \
  a real archive photo (key_people name query, kind (a)); if no real photo exists, the beat is \
  carried by a place or object. For any human action, query the place or the thing, never the \
  actor: "he approached her" -> "shopping mall parking lot 1974"; "the officer arrested him" -> \
  "1970s police cruiser with lights on at night"; "the jury deliberated" -> "empty 1970s jury \
  room, twelve chairs around a wooden table"; "posed as an officer" -> "a police badge on a car \
  seat"; forensic/medical beats -> "forensic laboratory bench 1970s", "autopsy report document", \
  never anatomy.
  - A person query is ONLY their plain name ("Georgann Hawkins") -- never their name plus a \
    descriptor about their body or death ("... skeletal remains", "... autopsy", "... body"). \
    Such queries can only ever return something unusable or grotesque; if the beat is about a \
    discovery, use the place instead ("wooded hillside search area, overcast").
  - CRIME SCENES are always AFTERMATH WITHOUT A VICTIM: no body, no blood, no gore, nothing \
    body-shaped (no covered forms, no chalk outlines). Convey it through the disturbed place \
    itself: "1970s bedroom with overturned nightstand and police evidence markers", "police tape \
    across a dormitory doorway", "flashlight beams in a dark forest clearing at night". The bare \
    words "crime scene" in a query make the model draw a victim -- always spell out the \
    empty-room composition instead.

- QUERY CRAFT for AI-generated queries -- generic queries produce ten identical dull buildings. \
  Every AI query must be a specific, art-directed shot:
  - CONCRETE CASE DETAIL, not a category. Pull the actual object/place from the narration: not \
    "1970s car" but "tan Volkswagen Beetle parked on a dark suburban street"; not "evidence" but \
    "handcuffs and a crowbar laid out on an evidence table with numbered tags"; not "lake" but \
    "Lake Sammamish shoreline with 1970s picnic tables, summer haze".
  - VARY THE SHOT within a scene: its AI queries must differ in framing -- one wide establishing \
    (whole location), one medium (an object in its setting), one close detail (texture, a single \
    prop filling the frame). Never two variations of the same wide building shot in one scene.
  - LIGHT AND ATMOSPHERE from the narration's moment: night scene -> "night, lit by a single \
    streetlamp, fog"; courtroom verdict -> "morning light through tall windows"; discovery in \
    the woods -> "overcast, bare trees, long shadows". Every query names a time of day or \
    weather/light condition.
  - NO REPEATS across the whole part: the same location type (courtroom, street, forest, police \
    station...) may recur only with a different angle, distance, or time of day -- an empty \
    courtroom wide shot may appear once; later courtroom beats use the gavel, the bench detail, \
    the corridor outside.
  - LIVED-IN, NOT STERILE. An empty scene must still look like life just stepped out of frame, \
    never like a studio product shot. Bad: a single pole on an empty road; a dress hanging on a \
    display rack; neat stacks of banknotes. Good: "1970s diner table with a half-finished coffee \
    and a folded newspaper", "car door left open at night, engine still running, headlights on", \
    "cluttered detective's desk with case files, full ashtray and a rotary phone off the hook". \
    Add traces of human presence -- objects mid-use, doors ajar, tire tracks, a burning cigarette \
    -- just never the human.
  - NEVER show clothing as a display: no garments on hangers, racks, mannequins or laid flat like \
    a catalog. If clothing matters to the beat, put it in situ: "a white shirt sleeve caught on a \
    car door", "sneakers left at the water's edge on an empty beach".
  - SHOW THE DISTINCTIVE CLUE, not the category. When the narration states a specific odd detail, \
    the query must depict exactly that detail: "the passenger seat was missing" -> "1970s \
    Volkswagen Beetle interior with the front passenger seat removed, bare metal floor"; "he found \
    handcuffs in the car" -> "handcuffs lying on the back seat of a 1970s car, flashlight beam". \
    Named evidence items each get their own close-up query.
  - DO NOT LITERALLY ILLUSTRATE what the narrator just said with a static object -- complement it. \
    If the narration already described the tennis outfit, don't render the outfit again; render \
    where it was worn ("empty tennis court at dusk, gate swinging open"). The visual should add \
    information or mood, not echo the sentence.
  - EVERY query and fallback must state the period ("1974", "1970s"). Image generation renders \
    the present day by default -- "Elizabeth Kloepfer fireplace" came back as a modern living \
    room with a flat-screen television over the hearth. Write "a 1974 living room, brick \
    fireplace, worn armchair" instead. Even a bare object needs it: "a rotary telephone on a \
    1970s kitchen counter", not "a telephone".
  - NEVER query a calendar, clock face, licence plate, price tag or anything whose point is \
    numbers -- generation garbles digits exactly like letters. For "weeks passed" use a seasonal \
    or light change ("bare trees along the same street, low winter sun").
  - MATCH THE SUBJECT TO WHAT IS BEING SAID. A stretch of narration about the murders or about \
    the man himself must NOT be illustrated with courtroom furniture just because a trial is \
    mentioned nearby. Court imagery belongs only to sentences about the trial. Late in a part, \
    when the narration turns back to the crimes, the victims or the manhunt, the visuals must \
    turn with it -- the ending of a part must not become a run of interchangeable courtroom \
    shots.
  - NO READABLE TEXT ANYWHERE. Image generation cannot spell -- newspapers, documents, signs and \
    posters come out covered in broken pseudo-letters. Never query a headline, document, report, \
    warrant, letter, poster, licence plate or shop sign. Where the story needs one, show the \
    object without its text being the subject: not "1975 newspaper headline about the arrest" \
    but "folded newspaper on a diner counter beside a coffee cup, shallow depth of field"; not \
    "arrest warrant document" but "manila case folders stacked on a desk under a lamp".
  - EVIDENCE ITEMS need a consistent, literal framing or they come out as vague clutter. Phrase \
    each as a single named object photographed as evidence: "a ski mask on a plain table, \
    police evidence photograph, harsh flash, numbered tag", "a crowbar on a plain table, police \
    evidence photograph, numbered tag". One object per query -- never "ski mask, rope and \
    handcuffs together".
  - MONEY: never render piles or stacks of cash. Small sums -> show the transaction trace: "a \
    handwritten receipt beside a wallet on a counter", "an open cash register drawer in a 1970s \
    store". One money-related image per part at most.
- Give each part a one-line "hook" -- a short teaser phrase summarizing that part's \
cliffhanger, usable later as part of a video title/caption.

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
                    "hook": {"type": "string"},
                    "scenes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "visual_queries": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                },
                                "visual_anchors": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                },
                                "visual_fallbacks": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                },
                            },
                            "required": ["text", "visual_queries", "visual_anchors", "visual_fallbacks"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["part_number", "hook", "scenes"],
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


def _load_brief(case_id: str) -> dict:
    path = _case_dir(case_id) / "brief.json"
    if not path.exists():
        raise RuntimeError(f"brief.json not found for case {case_id} -- run the story stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def _annotate_lengths(script: dict) -> dict:
    for part in script["parts"]:
        text = " ".join(scene["text"] for scene in part["scenes"])
        words = len(text.split())
        part["word_count"] = words
        part["est_seconds"] = round(words / 2.3)  # ~140 words/minute spoken narration
    return script


_REPAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "visual_queries": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "visual_anchors": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "visual_fallbacks": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "required": ["visual_queries", "visual_anchors", "visual_fallbacks"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["scenes"],
    "additionalProperties": False,
}


def _scene_sentence_count(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s])


# Subjects the prompt forbids but the model still asks for now and then --
# a query for "blood on white pillowcases" got through with the rule sitting
# right there in the instructions. The image gate caught the frame, but a
# query that can only produce something unusable wastes a generation slot
# and would have shipped had the gate blinked.
_FORBIDDEN_SUBJECTS = re.compile(
    r"\b(blood|bloody|bloodstain\w*|corpse|dead body|body bag|remains|skeletal|"
    r"autopsy|wound\w*|injur\w*|gore|strangl\w*|stab\w*|nude|naked)\b", re.I)


def _forbidden_query(scene: dict):
    """Queries asking for something the content rules exclude outright."""
    return [q for q in (scene.get("visual_queries") or []) if _FORBIDDEN_SUBJECTS.search(q)]


def _missing_person_query(scene: dict, known_names: list) -> bool:
    """True when the narration names a known person (victim, perpetrator,
    witness) but no visual query asks for that person -- every named person
    mentioned in a scene must get their own name query so their real photo
    can be found and shown at that moment."""
    text = (scene.get("text") or "").lower()
    queries_l = " | ".join(q.lower() for q in (scene.get("visual_queries") or []))
    for name in known_names:
        surname = name.split()[-1].lower()
        if len(surname) > 3 and surname in text and surname not in queries_l:
            return True
    return False


WORDS_PER_QUERY = 15  # one visual beat per ~5-6 seconds of narration
# Scenes per repair request. Each repaired scene costs a few hundred output
# tokens, so a whole case at once overruns the reply limit and returns
# truncated JSON.
REPAIR_BATCH_SIZE = 10


def _min_queries_for(text: str) -> int:
    words = len((text or "").split())
    n_sent = _scene_sentence_count(text)
    return max(1, round(words / WORDS_PER_QUERY), round(0.7 * n_sent))


def _scene_is_undercovered(scene: dict) -> bool:
    """A scene fails validation when its visual plan can't follow the
    narration: anchors don't pair 1:1 with queries, or the queries are too
    sparse to keep up with the words. Sparse anchors are what forced one
    subject to hold the screen for 15 seconds (and several near-identical
    generated frames of it) instead of the picture moving with the story."""
    queries = scene.get("visual_queries") or []
    anchors = scene.get("visual_anchors") or []
    fallbacks = scene.get("visual_fallbacks") or []
    if not (len(anchors) == len(fallbacks) == len(queries)):
        return True
    return len(queries) < _min_queries_for(scene.get("text", ""))


MAX_SCENE_WORDS = 60  # ~20s of narration; beyond this, anchor timing drifts


def _split_long_scenes(script: dict) -> int:
    """Split over-long scenes at sentence boundaries, carrying each query to
    the chunk its anchor falls in. Scene text is one audio file and one
    timing frame, so a 230-word scene makes every image position a guess
    accumulated over a minute of speech; ~60-word scenes keep each cut close
    to the words that justify it. Narration wording is never changed."""
    split_count = 0
    for part in script["parts"]:
        new_scenes = []
        for scene in part["scenes"]:
            text = scene.get("text", "")
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
            if len(text.split()) <= MAX_SCENE_WORDS or len(sentences) < 2:
                new_scenes.append(scene)
                continue

            queries = scene.get("visual_queries") or []
            anchors = scene.get("visual_anchors") or []
            fallbacks = (scene.get("visual_fallbacks") or [])[:]
            fallbacks += [""] * (len(queries) - len(fallbacks))
            # Word offset in the scene where each anchor's text begins.
            lowered = text.lower()
            offsets = []
            for anchor in anchors:
                pos = lowered.find(anchor.lower().strip())
                offsets.append(len(text[:pos].split()) if pos >= 0 else None)

            chunks, current, current_words = [], [], 0
            for sentence in sentences:
                current.append(sentence)
                current_words += len(sentence.split())
                if current_words >= MAX_SCENE_WORDS:
                    chunks.append(current)
                    current, current_words = [], 0
            if current:
                if chunks and current_words < 15:  # avoid a stub tail scene
                    chunks[-1] += current
                else:
                    chunks.append(current)

            word_cursor = 0
            for chunk in chunks:
                chunk_text = " ".join(chunk)
                chunk_len = len(chunk_text.split())
                lo, hi = word_cursor, word_cursor + chunk_len
                picked = [(q, a, f) for q, a, f, off in zip(queries, anchors, fallbacks, offsets)
                          if off is not None and lo <= off < hi]
                if not picked:  # no anchor landed here -- open on the chunk's own words
                    picked = [(queries[0] if queries else chunk[0][:40],
                               " ".join(chunk[0].split()[:4]),
                               fallbacks[0] if fallbacks else "")]
                new_scenes.append({
                    "text": chunk_text,
                    "visual_queries": [q for q, _, _ in picked],
                    "visual_anchors": [a for _, a, _ in picked],
                    "visual_fallbacks": [f for _, _, f in picked],
                })
                word_cursor = hi
            split_count += 1
        part["scenes"] = new_scenes
    return split_count


# Words that say nothing about the subject of a shot, so repeating them
# across a part is not repetition of imagery.
_SUBJECT_STOPWORDS = {
    "1970s", "1974", "1975", "1976", "1977", "1978", "1979", "night", "dusk",
    "dawn", "daytime", "evening", "morning", "interior", "exterior", "empty",
    "shot", "photograph", "photo", "close", "wide", "view", "with", "from",
    "and", "the", "under", "over", "beside", "across", "light", "lit", "dark",
}


def _overused_subjects(part: dict) -> set:
    """Subject words that dominate a part's visuals. The end of part 1 became
    a run of interchangeable courtroom shots while the narration had moved on
    to the murders -- counting subject words catches that mechanically
    instead of trusting the model to notice."""
    tally = {}
    total = 0
    for scene in part.get("scenes", []):
        for query in scene.get("visual_queries") or []:
            total += 1
            for word in {w.strip(",.").lower() for w in query.split() if len(w) > 4}:
                if word not in _SUBJECT_STOPWORDS:
                    tally[word] = tally.get(word, 0) + 1
    if total < 6:
        return set()
    return {word for word, n in tally.items() if n >= max(4, round(0.15 * total))}


def _repair_scenes(client, db, case_id: str, script: dict, known_names: list,
                   rounds: int = 2) -> int:
    """LLM pass(es) that rewrite ONLY the visual plan (queries + anchors) of
    scenes failing validation. Narration text is untouched. Runs up to
    `rounds` times because a single pass reliably came back a little short
    of the required density."""
    total = 0
    for _ in range(rounds):
        repaired = _repair_scenes_once(client, db, case_id, script, known_names)
        total += repaired
        if repaired == 0:
            break
    return total


def _repair_scenes_once(client, db, case_id: str, script: dict, known_names: list) -> int:
    bad = []
    for part in script["parts"]:
        overused = _overused_subjects(part)
        for scene in part["scenes"]:
            repeats = overused and any(
                any(w.strip(",.").lower() in overused for w in q.split())
                for q in (scene.get("visual_queries") or []))
            banned = _forbidden_query(scene)
            if banned:
                print(f"  rejected forbidden visual(s): {'; '.join(q[:44] for q in banned)}")
            if (_scene_is_undercovered(scene) or _missing_person_query(scene, known_names)
                    or repeats or banned):
                avoid = sorted(overused) if repeats else []
                bad.append((part, scene, avoid))
    if not bad:
        return 0

    # State the required count per scene explicitly. A rate rule ("one per
    # 12-15 words") alone came back consistently short (17.4 words/visual),
    # leaving single subjects on screen far too long.
    payload = []
    for i, (_p, s, overused) in enumerate(bad):
        entry = {
            "scene_number": i,
            "text": s["text"],
            "min_queries": _min_queries_for(s.get("text", "")),
        }
        if overused:
            entry["avoid_subjects"] = overused
        payload.append(entry)

    # Repair in batches. A whole 6-part case needs far more scenes rewritten
    # than fit in one reply -- the response was truncated mid-JSON and the
    # stage died. Batching keeps each reply comfortably inside the cap
    # regardless of how long the case is.
    repaired = 0
    for start in range(0, len(payload), REPAIR_BATCH_SIZE):
        batch = payload[start:start + REPAIR_BATCH_SIZE]
        fixes = _request_repairs(client, db, case_id, batch)
        for (_part, scene, _overused), fix in zip(bad[start:start + REPAIR_BATCH_SIZE], fixes):
            if _apply_repair(scene, fix):
                repaired += 1
    return repaired


def _apply_repair(scene: dict, fix: dict) -> bool:
    queries, anchors = fix.get("visual_queries") or [], fix.get("visual_anchors") or []
    newfalls = fix.get("visual_fallbacks") or []
    # Only take the rewrite if it's actually an improvement -- a "repair"
    # returning fewer visuals than the scene already had would make the
    # pacing worse, and applying it would also mask the failure.
    if not (len(queries) == len(anchors) == len(newfalls) and queries):
        return False
    if len(queries) < len(scene.get("visual_queries") or []):
        return False
    scene["visual_queries"] = queries
    scene["visual_anchors"] = anchors
    scene["visual_fallbacks"] = newfalls
    return True


def _request_repairs(client, db, case_id: str, payload: list) -> list:
    response = client.messages.create(
        model=SCRIPT_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT + "\n\nYou are now REPAIRING the visual plan of existing scenes. "
        "For each scene given, produce visual_queries and visual_anchors that follow every rule "
        "above: one query per roughly 12-15 words of narration (more when a sentence lists "
        "several concrete items), no subject holding the screen longer than ~6 seconds, no two "
        "adjacent queries showing the same subject, anchors verbatim from the text, same length "
        "as queries, in narration order. "
        "Each scene carries a \"min_queries\" number: return AT LEAST that many visual_queries "
        "(and the same number of anchors and fallbacks) for it -- fewer is a failed repair. "
        "visual_fallbacks[i] is the AI-renderable, person-free alternative for query i, specific "
        "to that beat and never a repeat of another query in the scene. "
        "EVERY person named in a scene's text (victim, perpetrator, witness) must get their own "
        "query that is exactly their full name, anchored where the narration first names them -- "
        "real archive photos of faces are the strongest images available, so never skip a name. "
        "If a scene carries \"avoid_subjects\", that imagery is already over-used elsewhere in "
        "the part: choose different subjects that match what THIS scene's narration is about. "
        "Return scenes in the same order they were given.",
        messages=[{"role": "user", "content": json.dumps({"scenes": payload}, ensure_ascii=False)}],
        output_config={"format": {"type": "json_schema", "schema": _REPAIR_SCHEMA}},
    )
    db.log_usage(case_id, "script_repair", SCRIPT_MODEL, response.usage.input_tokens, response.usage.output_tokens)
    blocks = [b.text for b in response.content if b.type == "text"]
    if not blocks:
        return []
    try:
        return json.loads(blocks[0])["scenes"]
    except (json.JSONDecodeError, KeyError):
        # A truncated or malformed reply must not take down the stage: the
        # script is already usable, the repair is an improvement pass.
        print(f"  warning: a repair batch came back unusable "
              f"(stop_reason={response.stop_reason}); those scenes keep their original visuals")
        return []


def run(case_id: str, db) -> None:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")

    brief = _load_brief(case_id)
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_prompt = (
        "Write the multi-part narration script for this case brief.\n\n"
        f"{json.dumps(brief, ensure_ascii=False, indent=2)}"
    )

    response = client.messages.create(
        model=SCRIPT_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )

    db.log_usage(case_id, "script", SCRIPT_MODEL, response.usage.input_tokens, response.usage.output_tokens)

    if response.stop_reason == "refusal":
        raise RuntimeError("Model refused the scriptwriting request")

    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        raise RuntimeError("Model returned no text content")

    script = json.loads(text_blocks[0])
    known_names = [p["name"] for p in brief.get("key_people", [])]
    repaired = _repair_scenes(client, db, case_id, script, known_names)
    if repaired:
        print(f"  visual plan repaired for {repaired} scene(s) (query coverage / anchor mismatch)")
    # Split after repair: the repair pass sees whole scenes, and splitting
    # first would hide long-range anchor ordering from it.
    split = _split_long_scenes(script)
    if split:
        print(f"  {split} over-long scene(s) split into ~{MAX_SCENE_WORDS}-word scenes for tighter image timing")
    script = _annotate_lengths(script)

    out_path = _case_dir(case_id) / "script.json"
    out_path.write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")

    db.update_case_status(case_id, "script_done")

    parts = script["parts"]
    print(f"  script written: {out_path}")
    cap_note = "" if len(parts) <= 6 else "  <- EXCEEDS 6-part cap, review prompt/output"
    print(f"  parts: {len(parts)}{cap_note}")
    for p in parts:
        note = "" if 260 <= p["word_count"] <= 440 else "  <- outside 2-3min target, review"
        print(f"    part {p['part_number']}: {p['word_count']} words (~{p['est_seconds']}s){note}")
