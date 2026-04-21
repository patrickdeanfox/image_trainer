from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from PIL import Image

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

ProgressCb = Callable[[int, int], None]


def _iter_source_images(src: Path) -> Iterable[Path]:
    for p in sorted(src.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            yield p


def resize_dataset(
    src: Path,
    dst: Path,
    target_size: int = 1024,
    progress_cb: Optional[ProgressCb] = None,
) -> list[Path]:
    """Scale shortest side to target_size, center-crop to target_size x target_size,
    save as zero-padded PNGs in dst. Returns written paths in order."""
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    sources = list(_iter_source_images(src))
    total = len(sources)
    written: list[Path] = []

    for i, img_path in enumerate(sources):
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            scale = target_size / min(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            w, h = img.size
            left = (w - target_size) // 2
            top = (h - target_size) // 2
            img = img.crop((left, top, left + target_size, top + target_size))

            out = dst / f"{i:04d}.png"
            img.save(out)
            written.append(out)
            print(f"Processed: {img_path.name} -> {out.name}", flush=True)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}", flush=True)

        if progress_cb is not None:
            progress_cb(i + 1, total)

    print(f"Done! {len(written)} images ready in {dst}", flush=True)
    return written
