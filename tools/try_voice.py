"""Standalone helper to preview Piper voices/intonation before picking one for
the pipeline. Not part of the agent pipeline -- run it directly.

Usage:
  venv\\Scripts\\python.exe tools\\try_voice.py en_US-ryan-high
  venv\\Scripts\\python.exe tools\\try_voice.py en_GB-alan-medium "Custom sample line." --length-scale 1.1

Full voice list: https://github.com/rhasspy/piper/blob/master/VOICES.md
(use the "key" column, e.g. "en_US-lessac-medium")

Tuning:
  --length-scale   speed: >1.0 slower/more dramatic, <1.0 faster (default 1.0)
  --noise-scale    expressiveness/variation (default 0.667)
  --noise-w-scale  phoneme width variation (default 0.8)
"""
import argparse
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import VOICES_DIR  # noqa: E402
from agents.voiceover import _ascii_safe_espeak_data_dir  # noqa: E402

DEFAULT_TEXT = (
    "In the early hours of the morning, detectives received a call that would "
    "change the course of the investigation."
)


def _download_voice(name: str) -> tuple[Path, Path]:
    import requests

    lang, rest = name.split("-", 1)
    speaker, quality = rest.rsplit("-", 1)
    lang_short = lang.split("_")[0]
    base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_short}/{lang}/{speaker}/{quality}"

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = VOICES_DIR / f"{name}.onnx"
    json_path = VOICES_DIR / f"{name}.onnx.json"

    for fname, dest in [(f"{name}.onnx", onnx_path), (f"{name}.onnx.json", json_path)]:
        if dest.exists():
            continue
        url = f"{base}/{fname}"
        print(f"downloading {fname}...")
        for attempt in range(5):
            try:
                resp = requests.get(url, timeout=60, stream=True)
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                break
            except requests.RequestException as exc:
                print(f"  retry {attempt + 1}/5: {exc}")
                time.sleep(2)
        else:
            raise RuntimeError(f"failed to download {fname} -- check the voice name against VOICES.md")

    return onnx_path, json_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("voice", help='Voice key, e.g. "en_US-ryan-high"')
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-w-scale", type=float, default=0.8)
    parser.add_argument("--out", default=None, help="Output WAV path (default: data/voices/preview_<voice>.wav)")
    args = parser.parse_args()

    onnx_path, json_path = _download_voice(args.voice)

    from piper import PiperVoice
    from piper.config import SynthesisConfig

    voice = PiperVoice.load(
        str(onnx_path), str(json_path), espeak_data_dir=_ascii_safe_espeak_data_dir()
    )
    syn_config = SynthesisConfig(
        length_scale=args.length_scale,
        noise_scale=args.noise_scale,
        noise_w_scale=args.noise_w_scale,
    )

    out_path = Path(args.out) if args.out else VOICES_DIR / f"preview_{args.voice}.wav"
    with wave.open(str(out_path), "wb") as wav_file:
        voice.synthesize_wav(args.text, wav_file, syn_config=syn_config)

    print(f"saved: {out_path}")
    print("Open it in File Explorer / any media player to listen.")


if __name__ == "__main__":
    main()
