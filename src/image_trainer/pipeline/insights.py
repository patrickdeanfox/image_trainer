"""Cheap image-quality heuristics for the Review tab.

PIL-only — no OpenCV / torch dependency for these helpers so the Review tab
works even when the user hasn't installed the training stack yet.

What's here:
- `image_stats(png)` — width/height, mean brightness, a rough sharpness score
  (variance of a small Laplacian-ish kernel applied in pure Python).
- `average_hash(png)` / `hamming(a, b)` — perceptual hash for duplicate
  detection.
- `find_near_duplicates(dir, threshold=6)` — returns pairs with hamming
  distance ≤ threshold.
- `resolution_warning(w, h, target)` — human-readable flag if the source is
  too small to be center-cropped to target without upscaling.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFilter, ImageStat


def _stats_from_image(img) -> dict:
    w, h = img.size
    stat = ImageStat.Stat(img)
    brightness = sum(stat.mean) / len(stat.mean)

    gray = img.convert("L")
    edges = gray.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], 1, 0))
    edge_data = list(edges.getdata())
    sharpness = statistics.pstdev(edge_data) if edge_data else 0.0

    return {
        "width": w,
        "height": h,
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
    }


def _hash_from_image(img, hash_size: int = 8) -> int:
    resized = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(resized.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p >= avg:
            bits |= 1 << i
    return bits


def image_stats(png: Path) -> dict:
    img = Image.open(png).convert("RGB")
    return _stats_from_image(img)


def average_hash(png: Path, hash_size: int = 8) -> int:
    """64-bit average hash. Good enough for flagging near-duplicates in a small
    personal dataset; not suitable for cryptographic use."""
    img = Image.open(png)
    return _hash_from_image(img, hash_size)


def stats_and_hash(png: Path, hash_size: int = 8) -> tuple[dict, int]:
    """Compute image_stats and average_hash from a single file-open."""
    img = Image.open(png).convert("RGB")
    return _stats_from_image(img), _hash_from_image(img, hash_size)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_near_duplicates(
    pngs: Iterable[Path], threshold: int = 6
) -> list[tuple[str, str, int]]:
    """Return (stem_a, stem_b, distance) for every pair whose hash distance is
    ≤ threshold. O(n^2) — fine for hundreds of images."""
    hashes = [(p.stem, average_hash(p)) for p in pngs]
    dupes: list[tuple[str, str, int]] = []
    for i in range(len(hashes)):
        stem_i, h_i = hashes[i]
        for j in range(i + 1, len(hashes)):
            stem_j, h_j = hashes[j]
            d = hamming(h_i, h_j)
            if d <= threshold:
                dupes.append((stem_i, stem_j, d))
    return dupes


def resolution_warning(w: int, h: int, target: int = 1024) -> str:
    """Return a short human-readable warning, or "" if the image is fine for
    the target resolution."""
    short = min(w, h)
    if short < target:
        return (
            f"source short side {short}px < target {target}px — will be upscaled"
        )
    if short < target * 1.1:
        return f"source short side {short}px is tight for target {target}px"
    return ""


def find_duplicates_for_stem(
    stem: str, all_pngs: list[Path], threshold: int = 6
) -> list[tuple[str, int]]:
    """Smaller helper for the GUI: for one stem, return the other stems whose
    hash is within threshold."""
    target_path = next((p for p in all_pngs if p.stem == stem), None)
    if target_path is None:
        return []
    target_hash = average_hash(target_path)
    out: list[tuple[str, int]] = []
    for p in all_pngs:
        if p.stem == stem:
            continue
        d = hamming(target_hash, average_hash(p))
        if d <= threshold:
            out.append((p.stem, d))
    out.sort(key=lambda x: x[1])
    return out
