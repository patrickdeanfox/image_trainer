"""Resize + crop raw images into square training-ready PNGs.

Step 2b of the pipeline. Takes each supported image in ``src``, scales the
shortest side up/down to ``target_size``, crops the longer side away, and
writes a zero-padded-index PNG into ``dst``. The output filenames are
positional (``0000.png``, ``0001.png``, …) so that the later caption step
can create sibling ``.txt`` files by shared stem.

Two crop strategies, selected by the ``face_aware`` flag:

1. **Face-aware (default, when ``facenet-pytorch`` is installed).** Detect
   the largest face in the source image, then position the face on a
   rule-of-thirds intersection in the output. The intersection is chosen
   from the face's natural quadrant in the source (upper-left face →
   upper-left intersection, etc.) so variety in the dataset's composition
   is driven by the source images themselves rather than an always-centered
   crop. If no face is detected, the image is still written with a
   center-crop fallback but its stem is returned in the
   ``face_failed_stems`` list so the caller can mark it excluded.

2. **Center-crop (fallback).** Classic behaviour: scale shortest side to
   ``target_size``, center-crop the longer side.

A per-image exception is caught and logged rather than aborting the whole
run — a single corrupt source file shouldn't take down a 200-image prep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Optional, Tuple

from PIL import Image

#: Image suffixes accepted by :func:`resize_dataset` and :mod:`ingest`.
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

#: Optional GUI progress callback: ``(done, total) -> None``.
ProgressCb = Callable[[int, int], None]


class ResizeResult(NamedTuple):
    """Return value of :func:`resize_dataset`.

    ``paths`` is the list of destination PNGs in write order (unchanged from
    the old ``list[Path]`` return). ``face_failed_stems`` holds the stems of
    images that were written but where face detection failed — the caller
    (:mod:`cli`) uses this to mark those images as excluded in review.json.
    ``face_success_stems`` holds stems where a face was found and the image
    was face-crop'd; the caller mirrors this into review.json so the Review
    tab can offer a "faces / no-face" filter.
    """

    paths: list[Path]
    face_failed_stems: list[str]
    face_success_stems: list[str]


def _iter_source_images(src: Path) -> Iterable[Path]:
    """Yield supported image files under `src` in sorted order."""
    for p in sorted(src.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            yield p


def _compute_face_aware_crop(
    w: int,
    h: int,
    face: Tuple[int, int, int, int],
    target_size: int,
) -> Tuple[int, int, Tuple[int, int]]:
    """Compute the crop window for a face-aware rule-of-thirds placement.

    Given the source image dimensions and the detected face box (both in
    source pixel coords), figure out:

    - The scale factor to apply so the shortest side becomes ``target_size``.
    - The top-left corner of the crop window in *scaled* pixel coords.

    The face's natural quadrant in the source decides which rule-of-thirds
    intersection it gets placed on in the output:

    - face centroid in left third of source → place on left third of output
    - right third of source → right third of output
    - middle third → horizontal centre (honours the source's intent)

    Vertically the same logic applies, with one tweak: when the face is
    vertically centred in the source we bias to the *upper* third (portrait
    convention — faces look better with breathing room below than above).

    The resulting crop window is clamped to valid bounds, so a face near
    an edge simply ends up as close to the target intersection as the
    image allows.

    Returns ``(left, top, (scaled_w, scaled_h))`` in scaled coords.
    """
    fx, fy, fw, fh = face
    scale = target_size / min(w, h)
    sw = int(round(w * scale))
    sh = int(round(h * scale))

    # Face centroid in source and its normalized position (used to pick
    # the natural quadrant).
    face_cx_src = fx + fw / 2.0
    face_cy_src = fy + fh / 2.0
    rx = face_cx_src / w
    ry = face_cy_src / h

    # Face centroid in scaled coords.
    face_cx = face_cx_src * scale
    face_cy = face_cy_src * scale

    # Target position of the face centroid *inside* the crop window.
    if rx < 1 / 3:
        target_x = target_size / 3.0
    elif rx > 2 / 3:
        target_x = 2 * target_size / 3.0
    else:
        target_x = target_size / 2.0

    if ry < 1 / 3:
        target_y = target_size / 3.0
    elif ry > 2 / 3:
        target_y = 2 * target_size / 3.0
    else:
        # Centre-vertical face: bias to upper third (classic portrait
        # framing with headroom below).
        target_y = target_size / 3.0

    left = int(round(face_cx - target_x))
    top = int(round(face_cy - target_y))

    # Clamp so the crop fits inside the scaled image.
    left = max(0, min(left, sw - target_size))
    top = max(0, min(top, sh - target_size))
    return left, top, (sw, sh)


def _center_crop_window(w: int, h: int, target_size: int) -> Tuple[int, int, Tuple[int, int]]:
    """Classic center-crop: scale shortest side, crop longer side equally."""
    scale = target_size / min(w, h)
    sw = int(round(w * scale))
    sh = int(round(h * scale))
    left = (sw - target_size) // 2
    top = (sh - target_size) // 2
    return left, top, (sw, sh)


def resize_dataset(
    src: Path,
    dst: Path,
    target_size: int = 1024,
    face_aware: bool = True,
    progress_cb: Optional[ProgressCb] = None,
    dry_run: bool = False,
) -> ResizeResult:
    """Scale + crop every supported image in ``src`` into ``target_size``-square PNGs in ``dst``.

    Transform, in order:

    1. Detect the largest face in the source (if ``face_aware=True`` and
       ``facenet-pytorch`` is available).
    2. Compute the crop window — face-aware rule-of-thirds if a face was
       found, else a centre-crop fallback.
    3. Scale the image so the shortest side equals ``target_size`` (LANCZOS).
    4. Crop to ``target_size × target_size`` at the computed origin.
    5. Write as a zero-padded PNG (``0000.png``, ``0001.png``, …).

    Args:
        src: Folder containing source images (e.g. the project's ``raw/``).
        dst: Output folder; created if missing.
        target_size: Edge length for the square output. SDXL base = 1024.
        face_aware: If ``True``, try to detect a face in each source and
            place it on a rule-of-thirds intersection. If detection isn't
            available (missing optional dep) or no face is found for a
            particular image, that image falls back to centre-crop and is
            flagged in the returned ``face_failed_stems``.
        progress_cb: Optional ``(done, total)`` callback for GUI progress.
        dry_run: Preview mode. When ``True``, crops are still computed the
            same way (so you can audit face-aware decisions) but each output
            is downscaled to **256px** before being written to ``dst`` and
            the caller is expected to target a throwaway folder like
            ``preview/`` rather than ``processed/``. The regular run
            ("are we overwriting anything already reviewed?") is decoupled
            from the full-res write. This keeps a dry-run fast and
            non-destructive — the user can open the preview folder, decide
            they like the crops, then run without ``--dry-run`` for real.

    Returns:
        :class:`ResizeResult` with ``paths`` (destination PNGs in write order,
        errored sources are printed but not returned) and ``face_failed_stems``
        (stems whose face detection failed — the caller should mark these as
        excluded in review.json so the user can eyeball them).
    """
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Preview thumbnail edge. Kept small so a dry run over 200 images is cheap.
    preview_size = 256

    sources = list(_iter_source_images(src))
    total = len(sources)
    written: list[Path] = []
    face_failed_stems: list[str] = []
    face_success_stems: list[str] = []

    detector_available = False
    if face_aware:
        from . import face_detect

        detector_available = face_detect.available()
        if not detector_available:
            print(
                "  face-aware crop requested but facenet-pytorch not installed; "
                "falling back to centre-crop for every image. "
                "Install with: pip install facenet-pytorch --no-deps",
                flush=True,
            )

    for i, img_path in enumerate(sources):
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            face = None
            if detector_available:
                from . import face_detect

                face = face_detect.detect_largest_face(img)

            if face is not None:
                left, top, (sw, sh) = _compute_face_aware_crop(
                    w, h, face, target_size
                )
                crop_strategy = "face"
            else:
                left, top, (sw, sh) = _center_crop_window(w, h, target_size)
                crop_strategy = "centre"

            img = img.resize((sw, sh), Image.LANCZOS)
            img = img.crop((left, top, left + target_size, top + target_size))

            if dry_run:
                img = img.resize((preview_size, preview_size), Image.LANCZOS)

            out = dst / f"{i:04d}.png"
            img.save(out)
            written.append(out)

            # Only flag as "failed" if face-aware was requested *and*
            # the detector was available but didn't find anything in
            # this image. A missing detector is the user's global choice,
            # not a per-image failure.
            if face_aware and detector_available and face is None:
                face_failed_stems.append(out.stem)
            elif face_aware and detector_available and face is not None:
                face_success_stems.append(out.stem)

            print(
                f"Processed: {img_path.name} -> {out.name} [{crop_strategy}-crop]",
                flush=True,
            )
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}", flush=True)

        if progress_cb is not None:
            progress_cb(i + 1, total)

    if face_failed_stems:
        print(
            f"No face detected in {len(face_failed_stems)} image(s); "
            f"these are still written but marked excluded so you can "
            f"review them in the GUI.",
            flush=True,
        )
    print(f"Done! {len(written)} images ready in {dst}", flush=True)
    return ResizeResult(
        paths=written,
        face_failed_stems=face_failed_stems,
        face_success_stems=face_success_stems,
    )
