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

Usage:
  python tools/add_manual_media.py chikatilo
  python tools/add_manual_media.py chikatilo --dry-run
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


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit("usage: python tools/add_manual_media.py <case_id> [--dry-run]")
    case_id, dry_run = args[0], "--dry-run" in sys.argv[1:]

    case_dir = ROOT / "data" / "cases" / case_id
    manifest_path = case_dir / "media_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found -- run the archive stage first")
    folder = case_dir / "media" / "manual"
    if not folder.is_dir():
        raise SystemExit(f"nothing to add: {folder} does not exist")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_scene = {(i["part_number"], i["scene_index"]): i for i in manifest["items"]}

    added = skipped = 0
    for path in sorted(folder.iterdir()):
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
        if item.get("status") not in APPROVED:
            item["status"] = "resolved"
            print(f"  + {path.name}  (scene status was '{item.get('status')}', now resolved)")
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
