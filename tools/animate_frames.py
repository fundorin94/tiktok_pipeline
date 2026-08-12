"""Animate a few stills per part so the video isn't entirely pans over
photographs.

Generating the whole video is out of reach on this card -- roughly 6 minutes
of compute per second of footage -- but a handful of moving shots per part is
an overnight job, and it lands where it matters: the establishing frame of a
scene, where a little wind or drift sells the place.

Only frames the pipeline already generated and cleared are used. Animating an
approved still keeps every image check that was run on it meaningful; asking
a video model for new footage would put unvetted content on screen.

Usage:
  venv/Scripts/python.exe tools/animate_frames.py --case-id test3
  venv/Scripts/python.exe tools/animate_frames.py --case-id test3 --max-parts 2 --per-part 3
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CASES_DIR  # noqa: E402

WAN2GP = Path(r"C:\Users\Andrey\OneDrive\Документы\Wan2GP")


def pick_frames(case_id: str, max_parts: int, per_part: int, all_frames: bool = False) -> list:
    """Which stills to animate.

    By default the first AI frame of each scene: openers are where movement
    reads as establishing rather than fidgety. With all_frames, every
    generated frame in range, which turns the part into moving footage
    throughout.

    Real archive photographs are never animated in either mode -- a moving
    portrait of a victim would be a fabrication, and those frames staying
    still also marks them out as the genuine material."""
    manifest = json.loads((CASES_DIR / case_id / "media_manifest.json").read_text(encoding="utf-8"))
    clips_dir = CASES_DIR / case_id / "media" / "clips"

    chosen, per_part_count = [], {}
    for item in manifest["items"]:
        part = item["part_number"]
        if max_parts and part > max_parts:
            continue
        frames = [p for p in item["local_paths"] if "ai_generated" in p]
        if not all_frames:
            frames = frames[:1]
        for frame in frames:
            if not all_frames and per_part_count.get(part, 0) >= per_part:
                break
            per_part_count[part] = per_part_count.get(part, 0) + 1
            chosen.append({"image": frame, "out": str(clips_dir / (Path(frame).stem + ".mp4"))})
    # The same frame can serve several slots; animate it once.
    seen, unique = set(), []
    for job in chosen:
        if job["image"] not in seen:
            seen.add(job["image"])
            unique.append(job)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--max-parts", type=int, default=0, help="0 = all parts")
    # One moving shot per scene. Three per part came out to roughly nine
    # seconds of movement in a two-minute video, which nobody noticed.
    parser.add_argument("--per-part", type=int, default=8)
    parser.add_argument("--all-frames", action="store_true",
                        help="animate every generated frame, not just scene openers, "
                             "so the part plays as footage rather than stills")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--model", default="fun_inp_1.3B")
    args = parser.parse_args()

    jobs = pick_frames(args.case_id, args.max_parts, args.per_part, args.all_frames)
    if not jobs:
        print("no AI frames found -- run the archive stage first")
        return 1

    est = len(jobs) * args.seconds * 382 / 60  # measured: ~382s of compute per second
    print(f"{len(jobs)} frame(s) to animate, roughly {est / 60:.1f} h at this card's speed")
    for j in jobs:
        print(f"  {Path(j['image']).name}")

    spec = CASES_DIR / args.case_id / "animate_jobs.json"
    spec.write_text(json.dumps({"model": args.model, "seconds": args.seconds, "jobs": jobs},
                               ensure_ascii=False, indent=2), encoding="utf-8")

    # Wan2GP pins diffusers/transformers/numpy versions that would break this
    # project's own stack, so it lives in its own venv and is driven as a
    # subprocess rather than imported.
    proc = subprocess.run(
        [str(WAN2GP / "venv" / "Scripts" / "python.exe"), "batch_i2v.py", str(spec)],
        cwd=str(WAN2GP),
    )
    if proc.returncode != 0:
        print("animation failed -- see output above")
        return proc.returncode

    written = {j["image"]: j["out"] for j in jobs if Path(j["out"]).is_file()}
    out_path = CASES_DIR / args.case_id / "clips_manifest.json"
    out_path.write_text(json.dumps(written, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(written)} clip(s) recorded in {out_path.name}")
    print("Re-run the video stage to use them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
