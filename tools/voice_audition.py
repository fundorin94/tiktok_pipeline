"""Audition Qwen3-TTS voices on real narration from this project.

Piper reads the script accurately but flatly, which is the wrong register for
true crime -- the genre lives on restraint and timing, not on information
delivery. Qwen3-TTS takes a plain-language instruction alongside the text
("measured, low, unhurried"), so the same paragraph can be heard in several
readings and picked by ear rather than guessed at.

Only the English speakers are auditioned; the others are native to languages
this series doesn't use.

Run:  venv/Scripts/python.exe tools/voice_audition.py
      venv/Scripts/python.exe tools/voice_audition.py --model 0.6B
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "voice_auditions"

# Speakers whose native language is English. Qwen3-TTS also ships Chinese,
# Japanese and Korean voices, which read English with a strong accent.
SPEAKERS = ["Ryan", "Aiden"]

# Readings worth comparing for this genre. The neutral one is the control:
# if an instruction doesn't beat it, it isn't earning its place.
STYLES = {
    "neutral": "",
    "documentary": "Speak as a documentary narrator: measured, low, unhurried, "
                   "letting facts land without dramatising them.",
    "restrained": "Speak quietly and seriously, with restraint, as if describing "
                  "something you find difficult to say.",
    "grave": "Speak slowly and gravely, with long pauses between sentences.",
}

# A real paragraph from the pipeline's own output, so the comparison is made
# on the kind of sentences these voices will actually have to read.
SAMPLE = (
    "On August 16, 1975, a highway patrol officer stopped a tan Volkswagen Beetle "
    "in Granger, Utah. When he looked inside, he saw something odd. The front "
    "passenger seat was missing. In the back he found a ski mask, handcuffs, "
    "and a crowbar."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="1.7B", choices=["0.6B", "1.7B"],
                        help="1.7B sounds better; 0.6B is faster and lighter")
    parser.add_argument("--text", default=SAMPLE, help="text to read")
    args = parser.parse_args()

    try:
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        print(f"missing dependency: {exc}\n\nInstall with:\n"
              "  venv/Scripts/pip.exe install qwen-tts soundfile")
        return 1

    model_id = f"Qwen/Qwen3-TTS-12Hz-{args.model}-CustomVoice"
    print(f"loading {model_id} ...")
    t0 = time.time()
    # bfloat16 keeps 1.7B near 3.5GB, which fits the 6GB card. flash-attention
    # is left off deliberately: it needs a compile step this machine doesn't
    # have set up, and the default attention is fast enough at this size.
    model = Qwen3TTSModel.from_pretrained(model_id, device_map="cuda:0", dtype=torch.bfloat16)
    print(f"loaded in {time.time() - t0:.0f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for speaker in SPEAKERS:
        for style, instruct in STYLES.items():
            dest = OUT_DIR / f"{speaker.lower()}_{style}.wav"
            t0 = time.time()
            try:
                wavs, sr = model.generate_custom_voice(
                    text=args.text, language="English", speaker=speaker,
                    **({"instruct": instruct} if instruct else {}),
                )
            except Exception as exc:
                print(f"  {speaker}/{style}: failed -- {exc}")
                continue
            sf.write(dest, wavs[0], sr)
            print(f"  {dest.name:28} {time.time() - t0:5.1f}s")

    print(f"\nAuditions in {OUT_DIR}")
    print("Listen through, then set the winner in agents/voiceover.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
