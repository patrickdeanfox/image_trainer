"""Resize + center-crop raw images into square training-ready PNGs.

Step 2b of the pipeline. Takes each supported image in `src`, scales the
shortest side up/down to `target_size`, center-crops the longer side away,
and writes a zero-padded-index PNG into `dst`. The output filenames are
positional (``0000.png``, ``0001.png``, …) so that the later caption step
can create sibling ``.txt`` files by shared stem.

A per-image exception is caught and logged rather than aborting the whole
run — a single corrupt source file shouldn't take down a 200-image prep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from PIL import Image

#: Image suffixes accepted by :func:`resize_dataset` and :mod:`ingest`.
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

#: Optional GUI progress callback: ``(done, total) -> None``.
ProgressCb = Callable[[int, int], None]


def _iter_source_images(src: Path) -> Iterable[Path]:
    """Yield supported image files under `src` in sorted order."""
    for p in sorted(src.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            yield p


def resize_dataset(
    src: Path,
    dst: Path,
    target_size: int = 1024,
    progress_cb: Optional[ProgressCb] = None,
) -> list[Path]:
    """Scale + center-crop every supported image in `src` into `target_size`-square PNGs in `dst`.

    The transform, in order: scale so the shortest side equals ``target_size``
    (LANCZOS), then center-crop to ``target_size × target_size``. Output files
    are renamed to zero-padded indexes starting at ``0000.png``.

    Args:
        src: Folder containing source images (e.g. the project's ``raw/``).
        dst: Output folder; created if missing.
        target_size: Edge length for the square output. SDXL base = 1024.
        progress_cb: Optional ``(done, total)`` callback for GUI progress bars.

    Returns:
        The list of destination PNG paths in write order. Files that errored
        are printed but not returned.
    """
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
