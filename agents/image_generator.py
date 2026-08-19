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


def _is_unsafe(image_path: str) -> bool:
    """A generated frame is unsafe if it contains any nudity OR any human
    face. AI output must be people-free (places/objects/documents only), so
    any detected face means the model wrongly drew a person -- reject it.
    Two independent detectors: YuNet catches small/distant/profile faces
    and collages; NudeNet's body-part labels catch everything else human --
    since NO people are allowed in AI frames, ANY body-part detection
    (face, belly, feet, armpits, breast -- covered or exposed alike) means
    the model drew a person and the frame is rejected."""
    if _face_count(image_path) > 0:
        return True
    detections = _detect(image_path)
    if detections is None:
        return True  # detector failed -> fail safe, drop the frame
    # every NudeNet label is a human body part -> any hit means a person
    return any(x.get("score", 0) >= _PERSON_PART_THRESHOLD for x in detections)


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
NEGATIVE_PROMPT = (
    "person, people, human, human figure, man, woman, boy, girl, child, "
    "face, portrait, headshot, body, torso, silhouette, crowd, pedestrian, "
    "mannequin, statue of a person, dress form, clothes on a hanger, "
    "garment display, legs, arms, hands, "
    "skull, skeleton, teeth, jaw, anatomy model, body part, "
    "corpse, dead body, body bag, blood, bloodstain, gore, wound, "
    "nude, naked, nsfw, erotic, sexual, suggestive, bare skin, "
    "anime, manga, cartoon, illustration, painting, drawing, sketch, cgi, "
    "3d render, collage, photo grid, yearbook, "
    "deformed, disfigured, malformed, extra limbs, mutated hands, "
    # SD cannot spell: any attempt at lettering renders as broken pseudo-text,
    # so all writing is pushed out of frame rather than merely discouraged.
    "text, letters, words, writing, handwriting, typography, font, caption, "
    "subtitles, signage, sign, poster, headline, newspaper text, document text, "
    "label, watermark, signature, logo, gibberish text, garbled letters, "
    # Period drift: SD defaults to present-day interiors (a plasma TV turned
    # up over a 1974 fireplace), so modern technology is pushed out of frame.
    "flat screen tv, plasma tv, lcd, led, computer monitor, laptop, "
    "smartphone, mobile phone, modern car, modern interior, modern furniture, "
    "digital display, usb, minimalist decor, "
    "low quality, lowres, jpeg artifacts, oversaturated, color, "
    "guitar, musician, band, concert, stage"
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
                   vision_gate=None) -> bool:
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
        prompt = visual_query + SHOT_MODIFIERS[variant % len(SHOT_MODIFIERS)] + STYLE_SUFFIX
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
                negative_prompt=NEGATIVE_PROMPT,
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
            if _is_unsafe(str(dest_path)):
                dest_path.unlink(missing_ok=True)
                continue
            # Third, outermost gate: a vision model looks at the frame and
            # rejects any human/corpse/nudity/gore the local detectors missed
            # (grainy b&w bodies shot from above slipped past both).
            if vision_gate is None:
                return True
            verdict = vision_gate(str(dest_path))
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
        print(f"    AI generation failed for {visual_query!r}: {exc}")
        return False
