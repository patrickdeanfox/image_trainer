"""Auto-caption processed images with BLIP-large.

Step 3 of the pipeline. Loads ``Salesforce/blip-image-captioning-large`` in
``float16`` on CUDA (no CPU fallback — by design: the CPU path is unusably
slow for dataset-scale work and hides GPU misconfiguration), then writes a
``<stem>.txt`` next to each ``<stem>.png`` containing
``"<trigger_word>, <blip_caption>"``.

Captions are deliberately rough. The expected workflow is:
1. run this step to seed every image with something reasonable;
2. open the Review tab and edit per-image captions (remove invariant
   descriptors, add chips, delete images that aren't useful).

The training loop reads ``<stem>.txt``, so any downstream edits you make in
the Review tab end up here verbatim on save.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

#: Optional GUI progress callback: ``(done, total) -> None``.
ProgressCb = Callable[[int, int], None]


def caption_dataset(
    images_dir: Path,
    trigger_word: str,
    model_id: str = "Salesforce/blip-image-captioning-large",
    progress_cb: Optional[ProgressCb] = None,
) -> list[Path]:
    """Generate a caption for every PNG in `images_dir` and write sibling ``.txt`` files.

    The model is loaded once per call (loading BLIP is the expensive part);
    inference then runs one image at a time in fp16 on CUDA.

    Args:
        images_dir: Folder of square PNGs produced by :func:`resize_dataset`.
        trigger_word: Token prepended to every caption. Keeps the subject
            identity bound to a nonce token the LoRA can learn without
            polluting real English.
        model_id: Any Hugging Face BLIP captioning model ID; defaults to the
            "large" variant which produces notably better captions than base.
        progress_cb: Optional ``(done, total)`` callback.

    Returns:
        The list of ``.txt`` paths that were written.

    Raises:
        RuntimeError: if ``torch.cuda.is_available()`` is False.
    """
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    if not torch.cuda.is_available():
        raise RuntimeError(
            "caption_dataset requires a CUDA GPU. "
            "Model is loaded in float16 on cuda; no CPU fallback."
        )

    images_dir = Path(images_dir)

    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")

    images = sorted(images_dir.glob("*.png"))
    total = len(images)
    written: list[Path] = []

    for i, img_path in enumerate(images):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
        out = model.generate(**inputs, max_new_tokens=75)
        caption = processor.decode(out[0], skip_special_tokens=True)

        full_caption = f"{trigger_word}, {caption}"
        caption_file = img_path.with_suffix(".txt")
        caption_file.write_text(full_caption)
        written.append(caption_file)

        print(f"Captioned {img_path.name}: {full_caption}", flush=True)
        if progress_cb is not None:
            progress_cb(i + 1, total)

    print(f"\nDone! {len(written)} text files created.", flush=True)
    return written
