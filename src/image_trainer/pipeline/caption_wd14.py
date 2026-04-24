"""WD14 Danbooru-style tagger, for NSFW-aware dataset captioning.

BLIP produces English sentences but tends to be timid about nudity, anatomy,
and adult body content — the kind of details a person-LoRA training on an
adult dataset actually needs to learn. WD14 (SmilingWolf's series) outputs
Danbooru-style tag lists, trained on a booru corpus, so it says what's
actually in the picture.

The default captioner in this pipeline runs **both** BLIP and WD14 and
concatenates their outputs:

    "<trigger_word>, <blip sentence>, <wd14 tag, wd14 tag, ...>"

BLIP gives context and framing ("a woman standing in a room"), WD14 gives
specific anatomical / clothing / pose tags ("standing, 1girl, long hair,
topless, large breasts, spread legs"). The LoRA benefits from both.

Runtime:
- Model: ``SmilingWolf/wd-v1-4-moat-tagger-v2`` (ONNX). Good quality/speed
  trade-off. Override via ``Project.wd14_model_id``.
- Requires ``onnxruntime`` (or ``onnxruntime-gpu``). If missing, the
  module raises a clear error at first use; it's an optional dep declared
  in ``pyproject.toml``.
- Model weights + tag CSV are downloaded from Hugging Face on first use
  and cached under ``~/.cache/huggingface/``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

ProgressCb = Callable[[int, int], None]

#: Default tag confidence threshold. Tags below this aren't included.
DEFAULT_GENERAL_THRESHOLD = 0.35
#: Separate threshold for character tags (rarely useful for a personal LoRA,
#: so default is higher to suppress them).
DEFAULT_CHARACTER_THRESHOLD = 0.85


def _load_wd14(model_id: str):
    """Download (if needed) and return (ort session, tag list) for a WD14 model."""
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "WD14 captioning needs onnxruntime. Install with:\n"
            "  pip install onnxruntime-gpu   # or: pip install onnxruntime\n"
            "Alternatively, set Project.captioner to 'blip' to skip WD14."
        ) from e
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "WD14 captioning needs huggingface_hub. It should have come in "
            "with `transformers`; reinstall the project deps."
        ) from e

    onnx_path = hf_hub_download(repo_id=model_id, filename="model.onnx")
    csv_path = hf_hub_download(repo_id=model_id, filename="selected_tags.csv")

    # Prefer CUDA ExecutionProvider when onnxruntime-gpu is installed; fall
    # back to CPU otherwise. Both are fine for our ~20-image datasets.
    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(onnx_path, providers=providers)

    # CSV columns: tag_id, name, category, count.
    # Category: 0=general, 4=character, 9=rating.
    tags: list[tuple[str, int]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags.append((row["name"], int(row["category"])))
    return session, tags


def _preprocess(image_path: Path, target_size: int):
    """Resize + center-crop an image to the square the WD14 model expects, and
    convert to a BGR float32 NHWC tensor (WD14 uses BGR, not RGB)."""
    import numpy as np
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    side = max(w, h)
    # Pad to square on white, then resize. This matches the SmilingWolf
    # pre-processing recipe.
    new = Image.new("RGB", (side, side), (255, 255, 255))
    new.paste(img, ((side - w) // 2, (side - h) // 2))
    new = new.resize((target_size, target_size), Image.BICUBIC)

    arr = np.asarray(new).astype("float32")
    # RGB -> BGR. ascontiguousarray because the negative-stride view from
    # `[:, :, ::-1]` is rejected by some ONNX runtimes.
    arr = np.ascontiguousarray(arr[:, :, ::-1])
    return arr[None, ...]  # add batch dim


def tag_image(
    image_path: Path,
    session,
    tags: list[tuple[str, int]],
    general_threshold: float = DEFAULT_GENERAL_THRESHOLD,
    character_threshold: float = DEFAULT_CHARACTER_THRESHOLD,
) -> list[str]:
    """Run the WD14 ONNX model on one image and return a list of tag strings
    above the configured thresholds, sorted by descending confidence."""
    import numpy as np

    input_name = session.get_inputs()[0].name
    h, w = session.get_inputs()[0].shape[1:3]  # typically 448
    try:
        size = int(h)
    except Exception:
        size = 448
    x = _preprocess(image_path, size)
    probs = session.run(None, {input_name: x})[0][0]

    scored: list[tuple[str, float, int]] = []
    for (name, category), p in zip(tags, probs):
        if category == 9:  # rating tag (safe / questionable / explicit) — not useful
            continue
        threshold = general_threshold if category == 0 else character_threshold
        if p >= threshold:
            scored.append((name, float(p), category))
    scored.sort(key=lambda t: t[1], reverse=True)
    # Danbooru tags use underscores; replace with spaces for readability in
    # the training caption files.
    return [name.replace("_", " ") for name, _, _ in scored]


def caption_dataset_wd14(
    images_dir: Path,
    trigger_word: str,
    model_id: str,
    progress_cb: Optional[ProgressCb] = None,
    general_threshold: float = DEFAULT_GENERAL_THRESHOLD,
    character_threshold: float = DEFAULT_CHARACTER_THRESHOLD,
    extra_suffix: str = "",
) -> list[Path]:
    """Write ``<stem>.txt`` files with WD14 tag lists prefixed by the trigger.

    Overwrites any existing ``.txt`` in ``images_dir``. Use
    :func:`caption_dataset_both` if you want BLIP + WD14 combined.
    """
    session, tags = _load_wd14(model_id)

    images_dir = Path(images_dir)
    images = sorted(images_dir.glob("*.png"))
    total = len(images)
    written: list[Path] = []

    for i, png in enumerate(images):
        tag_strs = tag_image(
            png, session, tags,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
        )
        parts = [trigger_word]
        if tag_strs:
            parts.extend(tag_strs)
        if extra_suffix:
            parts.append(extra_suffix)
        caption = ", ".join(parts)
        out = png.with_suffix(".txt")
        out.write_text(caption, encoding="utf-8")
        written.append(out)
        print(f"WD14 {png.name}: {caption[:120]}", flush=True)
        if progress_cb is not None:
            progress_cb(i + 1, total)

    del session
    print(f"\nDone! {len(written)} WD14 captions written.", flush=True)
    return written


def caption_dataset_both(
    images_dir: Path,
    trigger_word: str,
    blip_model_id: str,
    wd14_model_id: str,
    progress_cb: Optional[ProgressCb] = None,
    general_threshold: float = DEFAULT_GENERAL_THRESHOLD,
    character_threshold: float = DEFAULT_CHARACTER_THRESHOLD,
    extra_suffix: str = "",
) -> list[Path]:
    """Run BLIP + WD14 on each image and concatenate both outputs.

    Output format::

        "<trigger>, <blip sentence>, <wd14 tag, wd14 tag, ...>"

    BLIP supplies framing/context, WD14 supplies booru-style anatomical and
    stylistic tags. Best-in-class default for NSFW person-LoRA datasets.
    """
    # Lazy-imports: heavy deps only loaded when the dual captioner actually runs.
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Dual-captioning (BLIP + WD14) requires a CUDA GPU for BLIP. "
            "Set Project.captioner = 'wd14' to skip BLIP."
        )

    processor = BlipProcessor.from_pretrained(blip_model_id)
    blip = BlipForConditionalGeneration.from_pretrained(
        blip_model_id, torch_dtype=torch.float16
    ).to("cuda")
    blip.eval()
    session, tags = _load_wd14(wd14_model_id)

    images_dir = Path(images_dir)
    images = sorted(images_dir.glob("*.png"))
    total = len(images)
    written: list[Path] = []

    try:
        with torch.no_grad():
            for i, png in enumerate(images):
                # BLIP
                image = Image.open(png).convert("RGB")
                inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
                out = blip.generate(**inputs, max_new_tokens=75)
                blip_caption = processor.decode(out[0], skip_special_tokens=True).strip()
                del inputs, out

                # WD14
                wd14_tags = tag_image(
                    png, session, tags,
                    general_threshold=general_threshold,
                    character_threshold=character_threshold,
                )

                parts = [trigger_word]
                if blip_caption:
                    parts.append(blip_caption)
                parts.extend(wd14_tags)
                if extra_suffix:
                    parts.append(extra_suffix)
                caption = ", ".join(parts)

                outp = png.with_suffix(".txt")
                outp.write_text(caption, encoding="utf-8")
                written.append(outp)
                print(f"BLIP+WD14 {png.name}: {caption[:160]}", flush=True)
                if progress_cb is not None:
                    progress_cb(i + 1, total)
    finally:
        # Release GPU memory so a subsequent train step doesn't OOM.
        del blip, processor, session
        torch.cuda.empty_cache()

    print(f"\nDone! {len(written)} dual captions written.", flush=True)
    return written
