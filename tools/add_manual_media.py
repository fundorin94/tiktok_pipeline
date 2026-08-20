"""Fold hand-supplied photos into a case's media manifest.

Some visuals the pipeline cannot get on its own: a named real person whose
photo isn't in any public-domain archive, and whose face AI is not allowed to
invent. The archive stage lists those in manual_sourcing_queue.json and moves
on, leaving the scene to borrow a frame from its neighbour.

Drop the photo into data/cases/<case>/media/manual/, named after the query it
answers, and this puts it where the video stage will find it:

    part2_scene2_q0.jpg      the first frame for part 2, scene 2, query 0
    part2_scene2_q0_2.jpg    a second frame for that same query

Position matters, so the file is inserted next to the other frames of its own
query rather than appended -- frames are cut to the narration through
visual_anchors, and a face landing on the wrong sentence is worse than no face.

Re-running the archive stage picks these up too (media/manual is the one
folder it never wipes); this tool is the shortcut that avoids re-running it.

Mugshots are a separate case. The archive stage downloads one when a query
asks for it, but parks it in media/review/ instead of playing it, because
putting a real person's police photograph on screen is a decision for a human.
--approve-review is that decision being taken: it moves everything staged in
review/ into the scenes that asked for it.

Usage:
  python tools/add_manual_media.py chikatilo
  python tools/add_manual_media.py chikatilo --dry-run
  python tools/add_manual_media.py chikatilo --approve-review
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
NAME = re.compile(r"^part(\d+)_scene(\d+)_q(\d+)(?:_(\d+))?$", re.I)
Q_IN_PATH = re.compile(r"_q(\d+)")
# Statuses the video stage will actually play. A scene that resolved to
# nothing sits outside them, so adding a frame there means fixing the status
# too or the frame is ignored.
APPROVED = {"resolved", "found", "ai_generated", "ai_fallback"}


def _insert_at(local_paths: list, q_index: int) -> int:
    """Where a frame for query q_index belongs, keeping the list in query
    order. Existing frames carry their query in the filename, which is what
    makes this possible without changing the manifest format."""
    for position, path in enumerate(local_paths):
        found = Q_IN_PATH.search(Path(path).stem)
        if found and int(found.group(1)) > q_index:
            return position
    return len(local_paths)


def _approve_review(manifest: dict, case_dir: Path, dry_run: bool) -> int:
    """Play the mugshots that were staged for a human to look at."""
    approved = 0
    for item in manifest["items"]:
        frames = item.get("review_frames") or []
        for frame in frames:
            if not Path(frame).is_file():
                print(f"  ? {Path(frame).name}: staged but missing from disk -- skipped")
                continue
            if frame in item["local_paths"]:
                continue
            found = Q_IN_PATH.search(Path(frame).stem)
            q_index = int(found.group(1)) if found else len(item["local_paths"])
            item["local_paths"].insert(_insert_at(item["local_paths"], q_index), frame)
            was = item.get("status")
            if was not in APPROVED:
                item["status"] = "resolved"
            print(f"  + {Path(frame).name}  approved into part{item['part_number']} "
                  f"scene{item['scene_index']}" + (f" (was '{was}')" if was not in APPROVED else ""))
            approved += 1
        if frames and not dry_run:
            # Cleared so the publisher stops counting this as outstanding --
            # the file stays on disk, it is the pending flag that is resolved.
            item["review_frames"] = []
    if approved and not dry_run:
        queue_path = case_dir / "review_queue.json"
        if queue_path.exists():
            queue_path.write_text("[]", encoding="utf-8")
    return approved


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit("usage: python tools/add_manual_media.py <case_id> [--dry-run]")
    case_id = args[0]
    dry_run = "--dry-run" in sys.argv[1:]
    approve = "--approve-review" in sys.argv[1:]

    case_dir = ROOT / "data" / "cases" / case_id
    manifest_path = case_dir / "media_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found -- run the archive stage first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_scene = {(i["part_number"], i["scene_index"]): i for i in manifest["items"]}

    added = skipped = 0
    if approve:
        added += _approve_review(manifest, case_dir, dry_run)

    folder = case_dir / "media" / "manual"
    if not folder.is_dir():
        if not approve:
            raise SystemExit(f"nothing to add: {folder} does not exist")
        folder = None
    for path in sorted(folder.iterdir()) if folder else []:
        if path.suffix.lower() not in EXTENSIONS:
            continue
        parsed = NAME.match(path.stem)
        if not parsed:
            print(f"  ? {path.name}: name must look like part2_scene2_q0.jpg -- skipped")
            continue
        part, scene, q_index = (int(parsed.group(1)), int(parsed.group(2)), int(parsed.group(3)))
        item = by_scene.get((part, scene))
        if item is None:
            print(f"  ? {path.name}: no part {part} scene {scene} in this case -- skipped")
            continue
        if str(path) in item["local_paths"]:
            skipped += 1
            continue
        item["local_paths"].insert(_insert_at(item["local_paths"], q_index), str(path))
        was = item.get("status")
        if was not in APPROVED:
            item["status"] = "resolved"
            print(f"  + {path.name}  (scene status was '{was}', now resolved)")
        else:
            print(f"  + {path.name}")
        added += 1

    if dry_run:
        print(f"\ndry run: {added} would be added, {skipped} already present")
        return
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{added} added, {skipped} already present -> {manifest_path}")
    if added:
        print(f"now rebuild the video:  venv/Scripts/python.exe run_pipeline.py "
              f"--case-id {case_id} --stage video")


if __name__ == "__main__":
    main()
