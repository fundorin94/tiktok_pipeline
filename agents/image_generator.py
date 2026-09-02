import itertools
import secrets
from pathlib import Path

_PIPE = None
_NUDE_DETECTOR = None
_TMP_COUNTER = itertools.count()

# NudeNet labels for exposed body parts we must never publish. The SD safety
# checker only reliably catches explicit sexual content and lets bare torsos
# through, so this is the real backstop.
_NUDE_EXPOSED_LABELS = {
    "FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED", "ANUS_EXPOSED",
}
_NUDE_SCORE_THRESHOLD = 0.30  # conservative -- drop anything even borderline


def _get_nude_detector():
    global _NUDE_DETECTOR
    if _NUDE_DETECTOR is None:
        from nudenet import NudeDetector
        _NUDE_DETECTOR = NudeDetector()
    return _NUDE_DETECTOR


_FACE_LABELS = {"FACE_FEMALE", "FACE_MALE"}
_FACE_SCORE_THRESHOLD = 0.35  # NudeNet faces -- secondary signal, low bar
_PERSON_PART_THRESHOLD = 0.35  # any NudeNet body-part label at this score = a person

_YUNET_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
_YUNET_SCORE_THRESHOLD = 0.4  # aggressive -- AI frames must have no people at all
_YUNET = None


def _get_yunet():
    """YuNet face detector (OpenCV FaceDetectorYN). Catches small, distant,
    tilted and profile faces that NudeNet's detector misses -- the leak class
    behind tiny wrong-looking background people and face collages."""
    global _YUNET
    if _YUNET is None:
        import cv2
        import shutil
        import tempfile
        # OpenCV can't open files under the Cyrillic project path -- load the
        # model from an ASCII copy in the temp dir.
        ascii_model = Path(tempfile.gettempdir()) / _YUNET_MODEL_PATH.name
        if not ascii_model.exists():
            shutil.copy2(_YUNET_MODEL_PATH, ascii_model)
        _YUNET = cv2.FaceDetectorYN_create(
            str(ascii_model), "", (0, 0),
            score_threshold=_YUNET_SCORE_THRESHOLD,
        )
    return _YUNET


def _face_count(image_path: str) -> int:
    """Count faces in the image with YuNet. Returns a large sentinel on
    read/detector failure so the caller fails safe."""
    import cv2
    import numpy as np
    try:
        data = np.fromfile(image_path, dtype=np.uint8)  # handles Cyrillic paths
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return 99
        detector = _get_yunet()
        total = 0
        # Two passes: native size, and 2x upscale so small/distant/tilted
        # faces (the part-4 leak class) clear YuNet's minimum face size.
        for scale in (1.0, 2.0):
            scaled = img if scale == 1.0 else cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            detector.setInputSize((scaled.shape[1], scaled.shape[0]))
            _, faces = detector.detect(scaled)
            total += 0 if faces is None else len(faces)
        return total
    except Exception:
        return 99


def _detect(image_path: str):
    """Run NudeNet on an ASCII-path copy (OpenCV can't read the Cyrillic
    project path) and return the raw detection list. Fails safe: returns a
    sentinel that _is_unsafe treats as unsafe if detection can't run."""
    import shutil
    import tempfile
    src = Path(image_path)
    ascii_tmp = Path(tempfile.gettempdir()) / f"nudecheck_{next(_TMP_COUNTER)}{src.suffix}"
    try:
        shutil.copy2(src, ascii_tmp)
        return _get_nude_detector().detect(str(ascii_tmp))
    except Exception:
        return None
    finally:
        ascii_tmp.unlink(missing_ok=True)


_YOLO_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"
# 0.60, measured: across 70 empty-mode frames from the Zodiac run, 69 score
# exactly 0.00 and the only detection at all is a stack of paper at 0.54.
# The threshold sits above that. This backstop caught nothing real on that
# sample -- RealVisXL genuinely does not put people into object scenes the
# way SD 1.5 did -- so it is insurance against a failure that has not
# happened here, at about 0.2s a frame.
_YOLO_PERSON_CONF = 0.60
_YOLO = None


def _get_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO(str(_YOLO_MODEL_PATH))
    return _YOLO


def _person_count(image_path: str) -> int:
    """People in the frame, faces or not.

    This is the check YuNet and NudeNet cannot make. A figure walking away
    from camera has no face to find and, fully clothed, no body part NudeNet
    scores -- so a frame of two men on a dune, one in a coat with his back
    turned, passed both detectors as empty and shipped. YOLO finds them at
    0.9 confidence.

    Returns a large sentinel on failure so the caller fails safe, matching
    _face_count."""
    try:
        result = _get_yolo()(image_path, verbose=False, classes=[0])[0]
        return sum(1 for c in result.boxes.conf if float(c) >= _YOLO_PERSON_CONF)
    except Exception:
        return 99


def _unsafe_reason(image_path: str, people_allowed: bool = False) -> str:
    """Why this frame is rejected, or "" if it is fine.

    Returns the reason rather than a bare bool because these two detectors
    were the pipeline's only silent rejecter. The vision gate announces every
    verdict it makes; these dropped frames without a word, so a run where 112
    of 198 queries produced nothing looked like the model simply failing,
    with no way to tell an actual person from a false positive on a hand.
    """
    faces = _face_count(image_path)
    if faces >= 99:
        return "face detector could not read the file"
    if faces > 0:
        return f"{faces} face(s) detected"
    detections = _detect(image_path)
    if detections is None:
        return "body-part detector failed to run"
    if people_allowed:
        nude = [d for d in detections
                if d.get("class") in _NUDE_EXPOSED_LABELS
                and d.get("score", 0) >= _NUDE_SCORE_THRESHOLD]
        if nude:
            top = max(nude, key=lambda d: d.get("score", 0))
            return f"nudity: {top.get('class')} {top.get('score', 0):.2f}"
        return ""
    parts = [d for d in detections if d.get("score", 0) >= _PERSON_PART_THRESHOLD]
    if parts:
        top = max(parts, key=lambda d: d.get("score", 0))
        return f"body part: {top.get('class')} {top.get('score', 0):.2f}"
    # Last and broadest: a whole person, with or without a face. Only in
    # empty mode -- in figures mode people are the point of the shot, and the
    # face veto above is what keeps them anonymous.
    people = _person_count(image_path)
    if people >= 99:
        return "person detector could not read the file"
    if people > 0:
        return f"{people} person(s) detected (no face visible)"
    return ""


def _is_unsafe(image_path: str, people_allowed: bool = False) -> bool:
    return bool(_unsafe_reason(image_path, people_allowed))


def _is_nude(image_path: str) -> bool:
    # NudeNet uses OpenCV's imread under the hood, which -- like ffmpeg and
    # espeak -- cannot open non-ASCII paths on Windows (this project lives
    # under a Cyrillic directory). Run the detector on an ASCII-path temp
    # copy so it can actually read the file; otherwise imread silently
    # returns nothing, every frame reads as "unsafe", and each one gets
    # re-rolled to no purpose (the whole stage crawled for hours on this).
    import shutil
    import tempfile
    src = Path(image_path)
    ascii_tmp = Path(tempfile.gettempdir()) / f"nudecheck_{next(_TMP_COUNTER)}{src.suffix}"
    try:
        shutil.copy2(src, ascii_tmp)
        detections = _get_nude_detector().detect(str(ascii_tmp))
    except Exception:
        # If the detector genuinely fails to run, fail SAFE: treat as nude so
        # the frame is dropped rather than risk publishing something unchecked.
        return True
    finally:
        ascii_tmp.unlink(missing_ok=True)
    return any(
        d.get("class") in _NUDE_EXPOSED_LABELS and d.get("score", 0) >= _NUDE_SCORE_THRESHOLD
        for d in detections
    )

import os

# PIPELINE_FAST_IMAGES=1 switches to the old SD 1.5 checkpoint (~10s/frame
# vs SDXL's ~3.5min/frame with offload) for fast iteration on script/sync
# logic, where frame quality doesn't matter. All safety gates (detectors +
# vision model) apply in both modes.
FAST_MODE = os.environ.get("PIPELINE_FAST_IMAGES", "") == "1"
FAST_MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"

# SDXL portrait resolution -- native near-9:16, far less upscale blur in the
# final 1080x1920 video than SD 1.5's 512x768.
GEN_WIDTH, GEN_HEIGHT = (512, 768) if FAST_MODE else (832, 1216)

# RealVisXL V4.0 (SDXL photoreal, same author as Realistic Vision). Switched
# 2026-07-25 after benchmarking on this 6GB GPU: ~2-4 min/frame with CPU
# offload (peak ~5.6GB VRAM), but dramatically better prompt adherence and
# realism -- object/place shots look like actual archive photographs, and it
# almost never spontaneously draws people into object scenes like SD 1.5 did.
MODEL_ID = "SG161222/RealVisXL_V4.0"
VAE_FIX_ID = "madebyollin/sdxl-vae-fp16-fix"  # stock SDXL VAE breaks in fp16
INFERENCE_STEPS = 25 if FAST_MODE else 28
GUIDANCE_SCALE = 7.0 if FAST_MODE else 5.5  # SD 1.5 needs stronger CFG
MAX_NSFW_RETRIES = 4  # re-roll the seed if a frame is flagged nude/unsafe

# CLIP cuts the prompt at 77 tokens and silently drops the tail, so this
# suffix lives on a budget: near 37 tokens, leaving room for the query and the
# shot modifier. Anything appended here is paid for by something at the end
# falling off unnoticed -- a longer version of this string was losing its own
# last clause on every frame it generated.
# The decade is deliberately NOT named: _with_era appends the case's own
# period, and hardcoding "1970s" here styled a 1936 Ukrainian village as a
# 1970s press photo for the whole of the Chikatilo run.
STYLE_SUFFIX = (
    ", empty scene, no people, documentary photograph, black and white, "
    "grainy 35mm film, period-correct fixtures, unmarked surfaces, no signage, "
    "candid archival press photo"
)
# POLICY (2026-07-24): AI generation renders NO people, period. SD 1.5 kept
# leaking wrong faces, nudity, face collages and grotesque anatomy through
# every prompt/threshold combination we tried. AI now draws only places,
# objects, evidence props, documents, vehicles and landscapes; humans appear
# in the video exclusively via real archive photos. A face detector below
# hard-rejects any frame where a person slipped in anyway.
# CLIP truncates the negative prompt at 77 tokens too, and silently. The old
# version ran to 264 -- so 72% of it, every anti-text and anti-anachronism
# term in it, never reached the model at all. That is why frames kept being
# rejected for signage and modern fixtures the negative prompt already named.
# Both variants below are kept under 75 tokens and ordered by what the gate
# actually rejects: lettering first, period second, safety terms last (the
# detectors and the vision gate are the real backstop there, this is steering).
_NEG_SHARED = (
    "corpse, blood, gore, nude, "
    "text, letters, signage, sign, poster, watermark, logo, "
    "flat screen tv, computer monitor, laptop, smartphone, modern car, "
    "modern furniture, cartoon, painting, 3d render, lowres, color"
)
# Empty-place frames: push every trace of a person out.
NEGATIVE_PROMPT = (
    "person, people, face, figure, body, silhouette, crowd, hands, mannequin, "
    + _NEG_SHARED
)
# Figure frames: people are wanted, faces are not. A back, a shoulder or a
# distant shape carries the scene; a face makes it a portrait of someone who
# does not exist, or worse, of someone who does.
PEOPLE_NEGATIVE_PROMPT = (
    "face, facing camera, looking at viewer, portrait, headshot, deformed, "
    "extra limbs, " + _NEG_SHARED
)

# Companion to STYLE_SUFFIX for frames that are allowed figures. Same token
# budget, and it must actively ask for the back of the shot -- left to itself
# the model turns everyone around to face the camera.
PEOPLE_STYLE_SUFFIX = (
    ", seen from behind, faces not visible, distant anonymous figures, "
    "documentary photograph, black and white, grainy 35mm film, "
    "unmarked surfaces, no signage, candid archival press photo"
)


def _get_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    import torch
    from diffusers import (
        StableDiffusionXLPipeline, DPMSolverMultistepScheduler, AutoencoderKL,
    )

    if FAST_MODE:
        from diffusers import StableDiffusionPipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        try:
            pipe = StableDiffusionPipeline.from_pretrained(FAST_MODEL_ID, torch_dtype=dtype, local_files_only=True)
        except Exception:
            pipe = StableDiffusionPipeline.from_pretrained(FAST_MODEL_ID, torch_dtype=dtype)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++")
        pipe = pipe.to(device)
        if device == "cuda":
            pipe.enable_attention_slicing()
            pipe.vae.enable_slicing()
        print("  image generator: FAST mode (SD 1.5, iteration quality)", flush=True)
        _PIPE = pipe
        return pipe

    # Weights are cached under ~/.cache/huggingface after the first download;
    # load from disk with no network once cached.
    def _load(local_only):
        vae = AutoencoderKL.from_pretrained(
            VAE_FIX_ID, torch_dtype=torch.float16, local_files_only=local_only)
        vae.config.force_upcast = False
        return StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID, vae=vae, torch_dtype=torch.float16, variant="fp16",
            local_files_only=local_only)

    try:
        pipe = _load(True)
    except Exception:
        pipe = _load(False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    # -- Workarounds for fp16 dtype bugs in this diffusers/torch combo --
    # 1) text_encoder_2's projection breaks in fp16 under offload: run both
    #    text encoders in fp32 (they're small; encoding is a one-off per frame).
    pipe.text_encoder.to(torch.float32)
    pipe.text_encoder_2.to(torch.float32)

    # 2) float32 tensors leak into the fp16 unet/vae (added_cond kwargs, DPM
    #    solver latents) -- cast everything entering them to half.
    def _cast_half(v):
        return v.half() if torch.is_tensor(v) and v.is_floating_point() else v

    orig_unet_forward = pipe.unet.forward

    def patched_forward(sample, timestep, encoder_hidden_states, *args, **kwargs):
        sample = _cast_half(sample)
        encoder_hidden_states = _cast_half(encoder_hidden_states)
        if kwargs.get("added_cond_kwargs"):
            kwargs["added_cond_kwargs"] = {
                k: _cast_half(v) for k, v in kwargs["added_cond_kwargs"].items()
            }
        return orig_unet_forward(sample, timestep, encoder_hidden_states, *args, **kwargs)

    pipe.unet.forward = patched_forward
    orig_vae_decode = pipe.vae.decode
    pipe.vae.decode = lambda z, *a, **kw: orig_vae_decode(_cast_half(z), *a, **kw)

    # SDXL doesn't fit 6GB resident: offload keeps only the active block on
    # the GPU (~5.6GB peak at 832x1216). No safety_checker in the SDXL
    # pipeline -- _is_unsafe() below is the (stricter) gate instead.
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
    _PIPE = pipe
    return pipe


def _is_black(image) -> bool:
    """The safety checker replaces flagged frames with an all-black image."""
    extrema = image.convert("L").getextrema()
    return extrema == (0, 0)


# Per-variant shot direction: without this, the N frames of one query come
# out as near-identical takes ("three parked Beetles in a row"), which reads
# as the video hanging on one image. Variant i gets a different framing.
SHOT_MODIFIERS = [
    ", wide establishing shot",
    ", close-up detail shot",
    ", medium shot from a different angle, off-center composition",
    ", low angle shot",
    ", view through a doorway or window frame",
]


def generate_image(visual_query: str, dest_path: Path, variant: int = 0,
                   vision_gate=None, people: bool = False) -> bool:
    """vision_gate: optional callable(path) -> verdict dict carrying
    safe/period_ok/text_ok/reason (see agents.image_verifier.ai_frame_verdict).

    An unsafe verdict rejects the frame and re-rolls, same as the local
    detectors. A frame that is safe but merely off-period or text-heavy is
    held aside instead: we re-roll hoping for a clean one, and ship the best
    flagged frame if none arrives. Treating those two as one verdict is what
    made the gate expensive -- four rejections in five were a modern-looking
    lamp or a garbled sign, each paid for with a full re-generation, and a
    query that never came up clean lost its slot entirely."""
    spare = dest_path.with_suffix(".spare.png")
    try:
        import torch
        pipe = _get_pipeline()
        suffix = PEOPLE_STYLE_SUFFIX if people else STYLE_SUFFIX
        negative = PEOPLE_NEGATIVE_PROMPT if people else NEGATIVE_PROMPT
        prompt = visual_query + SHOT_MODIFIERS[variant % len(SHOT_MODIFIERS)] + suffix
        spare_score, spare_reason = -1, ""

        for attempt in range(MAX_NSFW_RETRIES + 1):
            # A fresh random seed every call -- generate_image is invoked once
            # per frame, so a fixed seed would make all of a query's frames
            # (ai0, ai1, ai2 ...) come out nearly identical. Random also gives
            # a genuinely different image on an NSFW re-roll.
            # CPU generator -- with model offload the pipeline's device moves
            # around; a CPU generator is valid for latent creation regardless.
            generator = torch.Generator().manual_seed(secrets.randbelow(2**31))
            out = pipe(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                width=GEN_WIDTH,
                height=GEN_HEIGHT,
                generator=generator,
            )
            image = out.images[0]
            flagged = False
            if getattr(out, "nsfw_content_detected", None):
                flagged = bool(out.nsfw_content_detected[0])
            if flagged or _is_black(image):
                # SD safety checker tripped -- try a different seed.
                continue

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest_path)
            # Second, stricter gate: reject any frame with nudity OR a human
            # face (AI should render only objects/places -- a face means it
            # wrongly drew a person). Re-roll on a new seed; drop the frame
            # entirely if it keeps producing people.
            local_reason = _unsafe_reason(str(dest_path), people_allowed=people)
            if local_reason:
                print(f"    local detector rejected a frame: {local_reason}", flush=True)
                dest_path.unlink(missing_ok=True)
                continue
            # Third, outermost gate: a vision model looks at the frame and
            # rejects any human/corpse/nudity/gore the local detectors missed
            # (grainy b&w bodies shot from above slipped past both).
            if vision_gate is None:
                return True
            verdict = vision_gate(str(dest_path), people)
            if not verdict["safe"]:
                dest_path.unlink(missing_ok=True)
                continue
            if verdict["period_ok"] and verdict["text_ok"]:
                spare.unlink(missing_ok=True)
                return True
            # Safe, but off-period or dominated by lettering. Hold on to the
            # least-flagged one seen so far and re-roll for something clean.
            score = int(verdict["period_ok"]) + int(verdict["text_ok"])
            if score > spare_score:
                spare_score, spare_reason = score, verdict["reason"]
                dest_path.replace(spare)
            else:
                dest_path.unlink(missing_ok=True)

        if spare_score >= 0:
            spare.replace(dest_path)
            print(f"    no clean re-roll -- keeping a flagged frame: {spare_reason[:70]}")
            return True
        print(f"    AI generation skipped for {visual_query!r}: kept producing a person/unsafe content")
        return False
    except Exception as exc:
        spare.unlink(missing_ok=True)
        # A gate that cannot run is not a bad frame -- it means nothing after
        # this point is checked, so it has to reach the top and stop the
        # stage. Swallowing it here turned a broken safety check into a run
        # that took 17 hours to produce a video with 112 empty queries.
        from agents.image_verifier import SafetyCheckUnavailable
        if isinstance(exc, SafetyCheckUnavailable):
            raise
        print(f"    AI generation failed for {visual_query!r}: {exc}")
        return False
