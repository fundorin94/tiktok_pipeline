import json
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from config import CASES_DIR

WIDTH, HEIGHT, FPS = 1080, 1920, 30
AUDIO_RATE = 22050
FONT = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
TITLE_OVERLAY_SECONDS = 3.0
LEAD_IN_SECONDS = 0.5  # silence before each part's first word so it isn't clipped
APPROVED_STATUSES = {"resolved", "found", "ai_generated", "ai_fallback"}
SUBCLIP_SECONDS = 3.0  # keep in sync with agents/archive_finder.py

# Cycled per sub-clip so even a single reused photo doesn't look static --
# different starting corner / zoom direction each time. max_zoom/mode are
# used to build a per-frame increment scaled to the clip's actual duration
# (see _build_visual_subclip) so the pan/zoom animates across the WHOLE
# clip -- a fixed absolute increment reaches its cap early on a long clip
# (e.g. a single-photo scene held for 20+s) and then sits frozen for the
# rest, which is exactly what it looked like to a viewer.
KEN_BURNS_PRESETS = [
    {"mode": "zoom_in", "max_zoom": 1.25, "x": "iw/2-(iw/zoom/2)", "y": "ih/2-(ih/zoom/2)"},
    {"mode": "zoom_out", "max_zoom": 1.25, "x": "iw/2-(iw/zoom/2)", "y": "ih/2-(ih/zoom/2)"},
    {"mode": "zoom_in", "max_zoom": 1.2, "x": "0", "y": "ih/2-(ih/zoom/2)"},
    {"mode": "zoom_in", "max_zoom": 1.2, "x": "iw-(iw/zoom)", "y": "ih/2-(ih/zoom/2)"},
]


def _case_dir(case_id: str) -> Path:
    d = CASES_DIR / case_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _video_dir(case_id: str) -> Path:
    d = _case_dir(case_id) / "video"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ascii_work_dir(case_id: str) -> Path:
    """ffmpeg is a C binary that mishandles non-ASCII argv on this machine
    (the project lives under a Cyrillic path) -- do all ffmpeg I/O in a
    plain ASCII temp directory and copy the final results back afterward."""
    d = Path(tempfile.gettempdir()) / "tiktok_pipeline" / case_id / "video"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise RuntimeError(f"{label} not found at {path} -- run the earlier stages first")
    return json.loads(path.read_text(encoding="utf-8"))


def _slash(p) -> str:
    return str(p).replace("\\", "/")


def _filter_path(p) -> str:
    """Escape a path for embedding as a drawtext option value inside an
    ffmpeg -vf filtergraph string (colons are filter-syntax separators)."""
    return _slash(p).replace(":", "\\:")


def _write_caption_file(text: str, dest: Path, width: int = 32) -> None:
    wrapped = "\n".join(textwrap.wrap(text, width=width))
    dest.write_text(wrapped, encoding="utf-8")


def _split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _sentence_time_windows(sentences: list, duration: float) -> list:
    """Split a scene's duration across its sentences proportional to word
    count. Not real forced alignment (we don't have word-level timestamps
    from Piper), but a much closer approximation than showing the whole
    scene's text as one static block for its full duration."""
    word_counts = [max(len(s.split()), 1) for s in sentences]
    total_words = sum(word_counts)
    windows = []
    t = 0.0
    for i, wc in enumerate(word_counts):
        if i == len(sentences) - 1:
            end = duration
        else:
            end = t + duration * (wc / total_words)
        windows.append((t, end))
        t = end
    return windows


def _run_ffmpeg(args: list, timeout: int = 300) -> None:
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error"] + args,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timed out after {timeout}s (likely a bad/oversized input file)")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-2000:]}")


PERPETRATOR_ROLES = {"suspect", "perpetrator", "killer", "offender", "murderer"}
OPENING_OVERRIDE = "opening"


def _perpetrator_frame(case_id: str, manifest: dict) -> str | None:
    """The photograph every part opens on.

    A viewer scrolling past decides in the first second whether this is a
    case they know, and the face is what tells them. Left to itself the
    opening frame is whatever the script's first visual beat happened to be
    -- a village, a road -- which identifies nothing.

    media/manual/opening.* wins if present, so the choice can be made by
    hand. Otherwise the brief names the perpetrator and the manifest is
    searched for a frame that was sourced for them."""
    manual = _case_dir(case_id) / "media" / "manual"
    if manual.is_dir():
        for path in sorted(manual.iterdir()):
            if path.stem.lower() == OPENING_OVERRIDE:
                return str(path)

    brief_path = _case_dir(case_id) / "brief.json"
    if not brief_path.exists():
        return None
    brief = json.loads(brief_path.read_text(encoding="utf-8"))

    # An unsolved case has no perpetrator to open on -- it has SUSPECTS, and
    # putting one of their faces at the head of six videos states as fact the
    # thing the case never established. The Zodiac brief lists two men, both
    # named, neither charged, both dead and unable to answer it. So when the
    # story stage raises its sensitivity flag, no face is chosen automatically.
    # media/manual/opening.* still works: a deliberate human choice is a
    # different act from a filename match, and it is checked before this.
    sensitivity = brief.get("sensitivity")
    if isinstance(sensitivity, dict) and sensitivity.get("flag"):
        print("  case is flagged sensitive (unsolved / no conviction) -- not opening "
              "on a suspect's photograph. Put a chosen image in media/manual/opening.jpg")
        return None

    names = [p.get("name", "") for p in (brief.get("key_people") or [])
             if (p.get("role") or "").lower() in PERPETRATOR_ROLES]
    if not names:
        return None

    # Match on surname: briefs give the full formal name while a visual query
    # often carries a shorter form of it.
    keys = {n.split()[-1].lower() for n in names if n.split()}
    for item in manifest["items"]:
        for q_index, meta in enumerate(item.get("queries") or []):
            if not any(k in (meta.get("query") or "").lower() for k in keys):
                continue
            marker = f"_q{q_index}"
            for path in item.get("local_paths") or []:
                # Real photographs only. A person query that found nothing
                # falls back to an AI rendering of the setting -- a street, a
                # doorway -- so an ai_generated frame filed under the
                # perpetrator's query is not a picture of him, and opening on
                # it would promise a face and show a road.
                if "ai_generated" in Path(path).parts or "ai_generated" in str(path):
                    continue
                if marker in Path(path).name and Path(path).is_file():
                    return path
    return None


def _index_media(manifest: dict) -> dict:
    return {(i["part_number"], i["scene_index"]): i for i in manifest["items"]}


_EXT_FOR_FORMAT = {"JPEG": ".jpg", "PNG": ".png", "GIF": ".gif", "WEBP": ".webp",
                   "BMP": ".bmp", "TIFF": ".tif"}


def _stage_input(src_path: str, work_dir: Path, tag: str) -> str:
    """Copy a source file (which may live under a non-ASCII path) into the
    ASCII work dir so ffmpeg can read it, and return the ASCII path.

    The extension comes from what the file actually IS, not what it is
    called. ffmpeg picks its demuxer by extension, and a hand-supplied
    "Edward Smith.jpg" that was really a GIF made it reach for the image2
    demuxer, reject -loop, and take a whole seven-frame scene down to a text
    placeholder. Downloaded pictures get renamed all the time, so the name
    is not evidence."""
    src = Path(src_path)
    suffix = src.suffix
    try:
        from PIL import Image
        with Image.open(src) as probe:
            suffix = _EXT_FOR_FORMAT.get(probe.format, suffix)
    except Exception:
        pass  # unreadable by PIL: keep the name's suffix and let ffmpeg try
    dest = work_dir / f"{tag}{suffix}"
    shutil.copy2(src, dest)
    return str(dest)


def _load_clips(case_id: str) -> dict:
    """Frames that have an animated version, written by tools/animate_frames.
    Absent for most cases -- generating motion costs about six minutes of
    compute per second of footage, so only a few shots per part get it."""
    path = _case_dir(case_id) / "clips_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_clip_subclip(local_clip: str, duration: float, work_dir: Path, tag: str) -> Path:
    """A shot cut from an animated clip. The clip is short, so it loops to
    fill its slot; no Ken Burns move is applied because the footage already
    moves, and stacking a pan on top reads as seasickness."""
    out_path = work_dir / f"{tag}.mp4"
    vf = (f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
          f"crop={WIDTH}:{HEIGHT},fps={FPS}")
    args = [
        "-stream_loop", "-1", "-i", local_clip,
        "-vf", vf, "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-t", str(duration),
        str(out_path),
    ]
    _run_ffmpeg(args)
    return out_path


def _build_visual_subclip(local_image: str, duration: float, work_dir: Path, tag: str, preset_index: int) -> Path:
    """One silent Ken Burns clip from a single image, for a ~3s sub-window
    of a scene. preset_index picks the pan/zoom direction so consecutive
    sub-clips (even of the same source photo) don't look identical."""
    out_path = work_dir / f"{tag}.mp4"
    frames = max(int(round(duration * FPS)), 1)
    preset = KEN_BURNS_PRESETS[preset_index % len(KEN_BURNS_PRESETS)]
    # Scale the per-frame increment to this clip's actual frame count so the
    # zoom reaches max_zoom right at the last frame, whether this is a ~3s
    # sub-clip or a much longer single-photo continuous clip.
    increment = (preset["max_zoom"] - 1.0) / frames
    if preset["mode"] == "zoom_in":
        z_expr = f"min(zoom+{increment:.6f},{preset['max_zoom']})"
    else:  # zoom_out -- start at max_zoom, ease back down to 1.0
        z_expr = f"if(eq(on,0),{preset['max_zoom']},max(zoom-{increment:.6f},1.0))"
    vf = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},"
        f"zoompan=z='{z_expr}':x='{preset['x']}':y='{preset['y']}':"
        f"d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS}"
    )
    args = [
        "-loop", "1", "-i", local_image,
        "-vf", vf, "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-t", str(duration),
        str(out_path),
    ]
    _run_ffmpeg(args)
    return out_path


def _normalize_words(text: str) -> list:
    return [re.sub(r"[^\w']+", "", w).lower() for w in text.split()]


def _anchor_word_offset(scene_words: list, anchor: str, search_from: int) -> int | None:
    """Index of the word where `anchor` starts in the scene text, scanning
    from `search_from` so identical anchors/phrases resolve in order.
    Matches on normalized words; tolerates one mismatched word inside the
    anchor (the LLM occasionally paraphrases a single token)."""
    a_words = _normalize_words(anchor)
    if not a_words:
        return None
    n, m = len(scene_words), len(a_words)
    for i in range(search_from, n - m + 1):
        mismatches = sum(1 for j in range(m) if scene_words[i + j] != a_words[j])
        if mismatches <= (1 if m >= 4 else 0):
            return i
    return None


def _speech_weights(raw_words: list) -> list:
    """Relative speaking time per word. Counting words as equal made images
    land early: "On August 16, 1975," is four words but is spoken as roughly
    a dozen ("nineteen seventy-five"), so everything anchored after a date
    appeared well before the narrator got there. Length, digit expansion and
    punctuation pauses together track real narration far more closely."""
    weights = []
    for word in raw_words:
        letters = sum(1 for ch in word if ch.isalpha())
        digits = sum(1 for ch in word if ch.isdigit())
        weight = letters + digits * 4 + 1  # digits are read out as words
        if word.endswith((".", "!", "?")):
            weight += 6  # full stop -- a real pause
        elif word.endswith((",", ";", ":")):
            weight += 3
        weights.append(float(weight))
    return weights


# Images that appear before the narrator says the words spoil the beat, so
# each cut is nudged slightly late rather than early.
ANCHOR_LAG_SECONDS = 0.4


def _query_time_windows(scene_text: str, anchors: list, n_queries: int, duration: float) -> list:
    """Per-query display windows for a scene: each visual query's images
    appear while the narration covers that query's anchor snippet. Position
    within the text maps to time by estimated speaking duration (see
    _speech_weights). Falls back to an even split when anchors are
    missing/broken."""
    even = [(duration * i / n_queries, duration * (i + 1) / n_queries) for i in range(n_queries)]
    if not anchors or len(anchors) != n_queries:
        return even

    raw_words = scene_text.split()
    scene_words = _normalize_words(scene_text)
    offsets = []
    search_from = 0
    for anchor in anchors:
        off = _anchor_word_offset(scene_words, anchor, search_from)
        if off is None:
            return even  # any unresolvable anchor -> don't guess, keep even
        offsets.append(off)
        search_from = off + 1
    if offsets != sorted(offsets):
        return even

    weights = _speech_weights(raw_words)
    cumulative, running = [0.0], 0.0
    for w in weights:
        running += w
        cumulative.append(running)
    total = running or 1.0

    def time_at(word_index: int) -> float:
        idx = min(word_index, len(cumulative) - 1)
        return duration * (cumulative[idx] / total)

    windows = []
    for i, off in enumerate(offsets):
        start = 0.0 if i == 0 else min(time_at(off) + ANCHOR_LAG_SECONDS, duration)
        end = duration if i == n_queries - 1 else min(
            time_at(offsets[i + 1]) + ANCHOR_LAG_SECONDS, duration)
        if end <= start:  # zero-length window (adjacent anchors) -> even split
            return even
        windows.append((start, end))
    return windows


_QUERY_INDEX_RE = re.compile(r"_q(\d+)_")


def _group_paths_by_query(image_paths: list, n_queries: int, kinds: list | None = None) -> list | None:
    """Split a scene's ordered frame list into per-query groups using the
    _q<idx>_ tag in the filenames. Returns None if any filename lacks the
    tag (legacy manifests) so the caller can fall back to the even walk.

    A query with no frames of its own borrows from a neighbour, but NEVER
    from another person's query: filling "Denise Naslund" with the photo
    sourced for "Janice Ott" puts the wrong woman on screen under a name
    the narrator is speaking. Those windows take a place/object frame
    instead, which reads as 'no photo of her' rather than a false ID."""
    groups = [[] for _ in range(n_queries)]
    for p in image_paths:
        match = _QUERY_INDEX_RE.search(Path(p).name)
        if not match:
            return None
        q = int(match.group(1))
        if q >= n_queries:
            return None
        groups[q].append(p)

    kinds = kinds or [""] * n_queries
    for i in range(n_queries):
        if groups[i]:
            continue
        order = [j for step in range(1, n_queries) for j in (i + step, i - step)]
        donor = next((groups[j] for j in order
                      if 0 <= j < n_queries and groups[j] and kinds[j] != "person"), None)
        if donor is None:  # nothing impersonal to fall back on
            return None
        groups[i] = donor
    return groups


def _borrow_frame(groups: list, g_idx: int, exclude: str, nth: int, kinds: list | None = None):
    """A frame from the nearest other query, used to break up a run of the
    same image. Searches outward (next query first -- a glimpse of what the
    narration is about to reach reads better than going back). Person photos
    are never borrowed into another query's window: a real face belongs only
    to the moment the narration names that person."""
    kinds = kinds or [""] * len(groups)
    order = []
    for step in range(1, len(groups)):
        order += [g_idx + step, g_idx - step]
    # Pool every other query's frames, nearest first, then index by nth: a
    # long window borrowing repeatedly cycles through different subjects
    # instead of flip-flopping with the same neighbor.
    pool = []
    for j in order:
        if 0 <= j < len(groups) and kinds[j] != "person":
            pool += [p for p in groups[j] if p != exclude and p not in pool]
    return pool[nth % len(pool)] if pool else None


def _build_scene_visual_track(image_paths: list, duration: float, work_dir: Path, tag: str,
                              sync: tuple | None = None, clips: dict | None = None) -> Path:
    """Cut the scene's duration into ~3s sub-clips, cycling through the
    available image(s) (and Ken Burns presets) so the visual changes every
    few seconds instead of holding one static frame for the whole scene.

    Even with only one real photo for the scene (common for a named
    person/object -- only one candidate is ever kept per query), still cut
    into ~3s sub-clips: a single continuous pan/zoom held for the scene's
    whole duration (previously used here) reads as a frozen/static image on
    anything longer than a few seconds. Each sub-clip cycles to a different
    Ken Burns preset (different starting crop/zoom direction), and now that
    the zoom rate is scaled to each sub-clip's own duration (see
    _build_visual_subclip), consecutive sub-clips of the same photo look
    like distinct shots rather than an obvious repeat."""
    first_preset_index = hash(tag) % len(KEN_BURNS_PRESETS)
    clips = clips or {}

    # NARRATION SYNC: when the scene has resolvable visual_anchors, each
    # query's images are shown during that query's own narration window
    # instead of walking all frames evenly across the scene.
    plan = []  # list of (source_path, subclip_duration)
    if sync:
        scene_text, anchors, n_queries, lead_in, kinds = sync
        groups = _group_paths_by_query(image_paths, n_queries, kinds)
        if groups:
            windows = _query_time_windows(scene_text, anchors, n_queries, duration - lead_in)
            # Fill ~3s slots across each query's window. A slot never repeats
            # the previous slot's file while any other frame in the scene is
            # available: holding one photo through a long window and relying
            # on the Ken Burns pan to carry it reads as the video freezing.
            previous = None
            for g_idx, ((w_start, w_end), group) in enumerate(zip(windows, groups)):
                w_dur = w_end - w_start + (lead_in if w_start == 0.0 else 0.0)
                k = max(1, round(w_dur / SUBCLIP_SECONDS))
                for j in range(k):
                    pick = group[j % len(group)]
                    # An animated shot may hold consecutive slots: it is moving
                    # footage, so repeating it reads as one continuous take
                    # rather than a frozen image, and a three-second clip in a
                    # two-minute part is otherwise too brief to notice.
                    if pick == previous and not clips.get(pick):
                        alt = _borrow_frame(groups, g_idx, exclude=previous, nth=j, kinds=kinds)
                        if alt:
                            pick = alt
                    plan.append((pick, w_dur / k))
                    previous = pick

    if not plan:
        # even walk fallback (legacy manifests / unresolvable anchors)
        n = max(1, round(duration / SUBCLIP_SECONDS))
        m = len(image_paths)
        plan = [(image_paths[min(i * m // n, m - 1)], duration / n) for i in range(n)]

    subclips = []
    for i, (src, sub_duration) in enumerate(plan):
        animated = clips.get(src)
        if animated and Path(animated).is_file():
            local = _stage_input(animated, work_dir, f"{tag}_sub{i}_src")
            subclips.append(_build_clip_subclip(local, sub_duration, work_dir, f"{tag}_sub{i}"))
            continue
        local_image = _stage_input(src, work_dir, f"{tag}_sub{i}_src")
        subclips.append(_build_visual_subclip(local_image, sub_duration, work_dir, f"{tag}_sub{i}", first_preset_index + i))

    if len(subclips) == 1:
        return subclips[0]

    track_path = work_dir / f"{tag}_track.mp4"
    _concat_segments(subclips, track_path, work_dir, f"{tag}_track")
    return track_path


_NAME_PLATE_NOISE = re.compile(
    r"\b(mugshot|mug shot|photo(graph)?|portrait|in court|booking|arrest|image)\b", re.I)


def _person_display_name(query: str) -> str:
    """The person's name as it should appear on the name plate -- the query
    minus search-helper words ("Ted Bundy mugshot" -> "Ted Bundy")."""
    cleaned = _NAME_PLATE_NOISE.sub("", query)
    return " ".join(cleaned.split()).strip(" ,-") or query


def _build_scene_segment(
    scene_text: str,
    duration: float,
    audio_path: str,
    image_paths: list | None,
    work_dir: Path,
    tag: str,
    title_overlay: tuple | None = None,
    lead_in: float = 0.0,
    anchors: list | None = None,
    n_queries: int = 0,
    query_meta: list | None = None,
    clips: dict | None = None,
) -> Path:
    # lead_in prepends a short silence so the narration's first word isn't
    # clipped by AAC encoder priming at the very start of each part.
    sentences = _split_sentences(scene_text) or [scene_text]
    windows = _sentence_time_windows(sentences, duration)
    if lead_in:
        windows = [(s + lead_in, e + lead_in) for (s, e) in windows]
    total_duration = duration + lead_in

    local_audio = _stage_input(audio_path, work_dir, f"{tag}_audio")
    kinds = [m.get("kind", "") for m in (query_meta or [])] or None
    sync = ((scene_text, anchors, n_queries, lead_in, kinds or [""] * n_queries)
            if (anchors and n_queries) else None)
    visual_track = _build_scene_visual_track(image_paths, total_duration, work_dir, f"{tag}_visual", sync, clips) if image_paths else None

    out_path = work_dir / f"{tag}.mp4"

    caption_filters = []
    for i, (sentence, (start, end)) in enumerate(zip(sentences, windows)):
        cap_path = work_dir / f"{tag}_caption{i}.txt"
        _write_caption_file(sentence, cap_path)
        caption_filters.append(
            f"drawtext=fontfile='{_filter_path(FONT)}':textfile='{_filter_path(cap_path)}':"
            f"fontsize=52:fontcolor=white:borderw=3:bordercolor=black:"
            f"line_spacing=8:x=(w-text_w)/2:y=h-380:enable='between(t,{start:.2f},{end:.2f})'"
        )
    # Name plate over a real person's photo: the narration lists several
    # victims in a row, and without a label the viewer can't tell which
    # woman is on screen. Only shown for queries that actually resolved to a
    # real photo of that person, so a plate never labels a stand-in image.
    if query_meta and anchors and n_queries:
        name_windows = _query_time_windows(scene_text, anchors, n_queries, duration)
        for i, (meta, (start, end)) in enumerate(zip(query_meta, name_windows)):
            if meta.get("kind") != "person" or meta.get("status") != "found":
                continue
            plate_path = work_dir / f"{tag}_name{i}.txt"
            plate_path.write_text(_person_display_name(meta.get("query", "")), encoding="utf-8")
            caption_filters.append(
                f"drawtext=fontfile='{_filter_path(FONT_BOLD)}':textfile='{_filter_path(plate_path)}':"
                f"fontsize=46:fontcolor=white:borderw=3:bordercolor=black:"
                f"box=1:boxcolor=black@0.5:boxborderw=14:"
                f"x=(w-text_w)/2:y=240:"
                f"enable='between(t,{start + lead_in:.2f},{end + lead_in:.2f})'"
            )
    drawtext = ",".join(caption_filters)

    if title_overlay:
        part_number, _hook = title_overlay
        # The hook used to be drawn here too, four boxed lines under the part
        # badge. On the published part 2 that put three separate texts on
        # screen at once -- badge, hook, subtitle -- over the faces on the
        # composite, while TikTok printed the same hook again as the post's
        # own caption directly below the frame. The hook is still written to
        # metadata.json and still goes out with the video; it just no longer
        # covers the picture twice over.
        enable = f"lt(t,{TITLE_OVERLAY_SECONDS})"
        drawtext += (
            f",drawtext=fontfile='{_filter_path(FONT_BOLD)}':text='PART {part_number}':"
            f"fontsize=80:fontcolor=white:borderw=4:bordercolor=black:"
            f"box=1:boxcolor=black@0.45:boxborderw=16:"
            f"x=(w-text_w)/2:y=180:enable='{enable}'"
        )

    if visual_track:
        args = ["-i", str(visual_track), "-i", local_audio, "-vf", drawtext]
    else:
        vf = drawtext
        args = [
            "-f", "lavfi", "-i", f"color=c=0x1a1a1a:s={WIDTH}x{HEIGHT}:d={total_duration}",
            "-i", local_audio, "-vf", vf,
        ]

    # Delay the narration by lead_in (silence padding) so the first word
    # survives; without it the audio starts at t=0 and gets clipped.
    if lead_in:
        args += ["-af", f"adelay={int(lead_in * 1000)}:all=1"]

    args += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-ar", str(AUDIO_RATE), "-b:a", "128k",
        "-t", str(total_duration),
        str(out_path),
    ]
    _run_ffmpeg(args)
    return out_path




def _concat_segments(segments: list, out_path: Path, work_dir: Path, tag: str) -> None:
    list_path = work_dir / f"{tag}_concat.txt"
    list_path.write_text(
        "\n".join(f"file '{_slash(p)}'" for p in segments), encoding="utf-8"
    )
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(out_path)])


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
    case_dir = _case_dir(case_id)
    script = _load_json(case_dir / "script.json", "script.json")
    media = _load_json(case_dir / "media_manifest.json", "media_manifest.json")
    audio = _load_json(case_dir / "audio_manifest.json", "audio_manifest.json")

    media_by_scene = _index_media(media)
    audio_by_scene = {(a["part_number"], a["scene_index"]): a for a in audio["scenes"]}

    work_dir = _ascii_work_dir(case_id)
    video_dir = _video_dir(case_id)
    clips = _load_clips(case_id)
    opening_frame = _perpetrator_frame(case_id, media)
    if opening_frame:
        print(f"  opening every part on: {Path(opening_frame).name}")
    else:
        # Stated rather than passed over: the rule is that a part opens on the
        # perpetrator, so failing to find one is a result, not a non-event.
        print("  warning: no photograph of the perpetrator found -- parts will open "
              "on their first scripted visual instead. Put one in media/manual/opening.jpg")
    if clips:
        print(f"  {len(clips)} animated clip(s) available -- those shots will move")

    placeholder_count = 0
    total_scene_count = 0
    part_paths = []
    last_good_images = None  # carried across scenes/parts so gaps reuse the most recent real photo

    for part in _parts_limit(script["parts"]):
        part_number = part["part_number"]
        segments = []

        for scene_index, scene in enumerate(part["scenes"]):
            total_scene_count += 1
            key = (part_number, scene_index)
            media_item = media_by_scene.get(key)
            audio_item = audio_by_scene.get(key)
            if not audio_item:
                raise RuntimeError(f"missing audio for part {part_number} scene {scene_index}")

            item_paths = media_item.get("local_paths") if media_item else None
            if item_paths:
                # A manifest entry can outlive its file: the archive stage
                # records a frame that a later re-roll then deletes. Drop those
                # here rather than letting the copy into the work dir raise --
                # a scene keeps whatever frames survived, and only a scene that
                # lost all of them borrows the previous one.
                on_disk = [p for p in item_paths if Path(p).is_file()]
                if len(on_disk) != len(item_paths):
                    print(f"  warning: part {part_number} scene {scene_index}: "
                          f"{len(item_paths) - len(on_disk)} frame(s) in the manifest are missing from disk")
                item_paths = on_disk
            has_visual = media_item and media_item.get("status") in APPROVED_STATUSES and item_paths
            if has_visual:
                image_paths = item_paths
                last_good_images = item_paths
            else:
                # No approved visual for this scene -- reuse the last real
                # photo shown rather than cutting to a blank/empty frame.
                image_paths = last_good_images
                placeholder_count += 1

            # Every part opens on the perpetrator. The frame is copied under a
            # _q0 name so it joins the first query's group -- the grouping
            # reads the query index out of the filename, so the original name
            # would file it under whichever query it was sourced for.
            if scene_index == 0 and opening_frame and image_paths:
                opener = work_dir / f"part{part_number}_scene0_q0_open{Path(opening_frame).suffix}"
                if not opener.exists():
                    shutil.copy2(opening_frame, opener)
                image_paths = [str(opener)] + [p for p in image_paths if p != str(opener)]

            title_overlay = (part_number, part["hook"]) if scene_index == 0 else None
            # Pad silence before the very first scene of each part so the
            # narration's opening word isn't clipped on playback.
            lead_in = LEAD_IN_SECONDS if scene_index == 0 else 0.0

            tag = f"part{part_number}_scene{scene_index}"
            # Narration sync inputs -- only when this scene's own frames are
            # in use (borrowed placeholder frames can't map to its queries).
            anchors = (media_item.get("visual_anchors") or []) if has_visual else []
            n_queries = len(media_item.get("visual_queries") or []) if has_visual else 0
            query_meta = (media_item.get("queries") or []) if has_visual else None
            try:
                seg_path = _build_scene_segment(
                    scene["text"], audio_item["duration_seconds"], audio_item["audio_path"],
                    image_paths, work_dir, tag, title_overlay, lead_in,
                    anchors, n_queries, query_meta, clips,
                )
            except (RuntimeError, OSError) as exc:
                # A bad/oversized/unreadable source image shouldn't take down
                # the whole run -- fall back to a blank card for just this
                # scene. OSError is in here because a missing or locked file
                # raises that, not RuntimeError, and escaping this handler
                # costs the entire run at its very last stage.
                print(f"  warning: part {part_number} scene {scene_index} visual failed ({exc}); using placeholder")
                seg_path = _build_scene_segment(
                    scene["text"], audio_item["duration_seconds"], audio_item["audio_path"],
                    None, work_dir, tag, title_overlay, lead_in,
                )
            segments.append(seg_path)

        local_part_out = work_dir / f"part{part_number}.mp4"
        _concat_segments(segments, local_part_out, work_dir, f"part{part_number}")

        final_part_out = video_dir / f"part{part_number}.mp4"
        shutil.copy2(local_part_out, final_part_out)
        part_paths.append(final_part_out)

    db.update_case_status(case_id, "video_done")

    print(f"  {len(part_paths)} part video(s) written to {video_dir}")
    for p in part_paths:
        print(f"    {p.name}")
    if placeholder_count:
        print(
            f"  {placeholder_count}/{total_scene_count} scene(s) used a text placeholder "
            f"(no approved visual) -- review review_queue.json / manual_sourcing_queue.json before publishing"
        )
