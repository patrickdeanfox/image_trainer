from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

ProgressCb = Callable[[int, int], None]


def caption_dataset(
    images_dir: Path,
    trigger_word: str,
    model_id: str = "Salesforce/blip-image-captioning-large",
    progress_cb: Optional[ProgressCb] = None,
) -> list[Path]:
    """Run BLIP on every PNG in images_dir and write a sibling .txt file
    containing "<trigger_word>, <caption>". Requires a CUDA GPU.
    Returns list of written .txt paths."""
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
