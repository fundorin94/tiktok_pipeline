import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from config import (CASES_DIR, QWEN_TTS_CLONE_MODEL, QWEN_TTS_INSTRUCT, QWEN_TTS_MODE,
                    QWEN_TTS_MODEL, QWEN_TTS_REF_AUDIO, QWEN_TTS_REF_TEXT, QWEN_TTS_SEED,
                    QWEN_TTS_SPEAKER, QWEN_TTS_TEMPERATURE, VOICE_CONFIG_PATH,
                    VOICE_ENGINE, VOICE_MODEL_PATH, VOICE_TEMPO)

_VOICE = None
_QWEN = None


def _case_dir(case_id: str) -> Path:
    d = CASES_DIR / case_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _audio_dir(case_id: str) -> Path:
    d = _case_dir(case_id) / "audio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_script(case_id: str) -> dict:
    path = _case_dir(case_id) / "script.json"
    if not path.exists():
        raise RuntimeError(f"script.json not found for case {case_id} -- run the script stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def _ascii_safe_espeak_data_dir() -> str:
    """Piper's bundled espeak-ng C extension fails silently on non-ASCII
    paths on Windows (e.g. a Cyrillic project directory). If the package's
    default data dir isn't pure ASCII, copy it once to a safe temp path."""
    from piper.phonemize_espeak import ESPEAK_DATA_DIR

    default_dir = Path(ESPEAK_DATA_DIR)
    try:
        str(default_dir).encode("ascii")
        return str(default_dir)
    except UnicodeEncodeError:
        pass

    safe_dir = Path(tempfile.gettempdir()) / "piper_espeak_data"
    if not safe_dir.exists():
        import shutil

        shutil.copytree(default_dir, safe_dir)
    return str(safe_dir)


def _get_voice():
    global _VOICE
    if _VOICE is not None:
        return _VOICE

    if not VOICE_MODEL_PATH.exists() or not VOICE_CONFIG_PATH.exists():
        raise RuntimeError(
            f"Piper voice model not found at {VOICE_MODEL_PATH}. "
            "Download it first (see project setup)."
        )

    from piper import PiperVoice

    _VOICE = PiperVoice.load(
        str(VOICE_MODEL_PATH),
        str(VOICE_CONFIG_PATH),
        espeak_data_dir=_ascii_safe_espeak_data_dir(),
    )
    return _VOICE


def _get_qwen():
    global _QWEN
    if _QWEN is None:
        import torch
        from qwen_tts import Qwen3TTSModel

        model_id = QWEN_TTS_CLONE_MODEL if QWEN_TTS_MODE == "clone" else QWEN_TTS_MODEL
        print(f"  loading {model_id} (first run downloads ~3.5GB) ...", flush=True)
        _QWEN = Qwen3TTSModel.from_pretrained(
            model_id, device_map="cuda:0", dtype=torch.bfloat16)
    return _QWEN


def _apply_tempo(path: Path) -> None:
    """Adjust pace without shifting pitch. ffmpeg cannot open this project's
    Cyrillic path on this machine, so the filtering happens on ASCII copies
    in the temp directory."""
    if abs(VOICE_TEMPO - 1.0) < 0.01:
        return
    tmp_dir = Path(tempfile.gettempdir())
    src = tmp_dir / f"tts_src_{os.getpid()}.wav"
    out = tmp_dir / f"tts_out_{os.getpid()}.wav"
    try:
        shutil.copy2(path, src)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-filter:a", f"atempo={VOICE_TEMPO}", str(out)],
            check=True, capture_output=True,
        )
        shutil.copy2(out, path)
    finally:
        src.unlink(missing_ok=True)
        out.unlink(missing_ok=True)


def _synthesize(text: str, dest_path: Path) -> float:
    if VOICE_ENGINE == "qwen":
        import soundfile as sf

        import torch

        # Same seed and a low temperature for every scene: each one is its own
        # generation, and at default sampling the reading drifted between them.
        torch.manual_seed(QWEN_TTS_SEED)
        if QWEN_TTS_MODE == "clone":
            if not QWEN_TTS_REF_AUDIO.exists():
                raise RuntimeError(
                    f"reference narration not found at {QWEN_TTS_REF_AUDIO} -- "
                    "pick a take with tools/voice_audition.py and copy it there")
            wavs, sample_rate = _get_qwen().generate_voice_clone(
                text=text, language="English",
                ref_audio=str(QWEN_TTS_REF_AUDIO), ref_text=QWEN_TTS_REF_TEXT,
                temperature=QWEN_TTS_TEMPERATURE,
            )
        else:
            wavs, sample_rate = _get_qwen().generate_custom_voice(
                text=text, language="English",
                speaker=QWEN_TTS_SPEAKER, instruct=QWEN_TTS_INSTRUCT,
                temperature=QWEN_TTS_TEMPERATURE,
            )
        sf.write(dest_path, wavs[0], sample_rate)
        # Tempo is applied before the duration is read: the video stage times
        # every cut against these numbers, so they must describe the audio as
        # it will actually play.
        _apply_tempo(dest_path)
    else:
        voice = _get_voice()
        with wave.open(str(dest_path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

    with wave.open(str(dest_path), "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    return round(frames / float(rate), 2)


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
    audio_dir = _audio_dir(case_id)

    entries = []
    total_seconds = 0.0

    for part in _parts_limit(script["parts"]):
        part_number = part["part_number"]
        for scene_index, scene in enumerate(part["scenes"]):
            text = scene["text"]
            filename = f"part{part_number}_scene{scene_index}.wav"
            dest_path = audio_dir / filename
            duration = _synthesize(text, dest_path)
            total_seconds += duration
            entries.append({
                "part_number": part_number,
                "scene_index": scene_index,
                "text": text,
                "audio_path": str(dest_path),
                "duration_seconds": duration,
            })

    manifest = {"case_id": script.get("case_id", case_id), "scenes": entries}
    manifest_path = _case_dir(case_id) / "audio_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    db.update_case_status(case_id, "voiceover_done")

    print(f"  audio manifest written: {manifest_path}")
    print(f"  {len(entries)} scene(s) synthesized, total narration: {round(total_seconds / 60, 1)} min")
