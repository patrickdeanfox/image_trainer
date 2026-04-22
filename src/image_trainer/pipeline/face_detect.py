"""Face detection for the face-aware crop step in prep.

Wraps MTCNN from ``facenet-pytorch`` behind a small, stable surface so the
resize module doesn't depend on MTCNN's API directly. The detector is loaded
lazily on first call — importing this module is cheap and has no side effects
even when ``facenet-pytorch`` isn't installed.

Used by :func:`pipeline.resize.resize_dataset` when ``face_aware=True``. If
``facenet-pytorch`` isn't available, :func:`detect_largest_face` returns
``None`` so callers can fall back to center-crop cleanly.

To enable, install facenet-pytorch manually with ``--no-deps`` (it pins
torch<2.3 which would downgrade the rest of the pipeline):

    .venv/bin/python -m pip install facenet-pytorch --no-deps
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

#: ``(x, y, w, h)`` in source-image pixel coordinates.
FaceBox = Tuple[int, int, int, int]

# Cached MTCNN instance. We only load it once per process; MTCNN is ~50 MB
# of weights plus some Python startup, so repeatedly instantiating it on
# every call would make prep significantly slower.
_mtcnn = None
_import_attempted = False
_import_error: Optional[str] = None


def available() -> bool:
    """Return True if the MTCNN detector is importable in this environment.

    Does not load the model — just checks that ``facenet-pytorch`` and its
    transitive deps (``torch``, ``Pillow``) are importable. Callers can use
    this to decide whether to enable face-aware crop or silently fall back.
    """
    global _import_attempted, _import_error
    if _import_attempted:
        return _import_error is None
    _import_attempted = True
    try:
        import facenet_pytorch  # noqa: F401
    except Exception as e:  # pragma: no cover - trivial branch
        _import_error = str(e)
        return False
    return True


def _get_detector():
    """Lazy singleton for the MTCNN detector. CUDA if available, else CPU."""
    global _mtcnn
    if _mtcnn is not None:
        return _mtcnn
    if not available():
        raise RuntimeError(
            f"Face detection requires facenet-pytorch. Install with:\n"
            f"  pip install -e \".[face]\"\n"
            f"(import error: {_import_error})"
        )
    import torch  # local import, already a core dep
    from facenet_pytorch import MTCNN  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # keep_all=True lets us pick the largest face when multiple are detected,
    # which is more reliable than MTCNN's built-in "best" selection for
    # personal-likeness training where the subject is usually the biggest
    # face in frame.
    _mtcnn = MTCNN(
        keep_all=True,
        post_process=False,
        device=device,
    )
    return _mtcnn


def detect_largest_face(image: Image.Image) -> Optional[FaceBox]:
    """Detect the largest face in ``image`` and return its bounding box.

    The "largest" face is the one with the biggest bounding-box area, which
    is a good proxy for "the subject" in typical portrait / selfie data. If
    no face is detected, returns ``None``.

    Args:
        image: PIL image in RGB mode.

    Returns:
        ``(x, y, w, h)`` in the source image's pixel coordinates, or ``None``
        if no face was found. Bounds are clipped to the image's extent.
    """
    if not available():
        return None
    detector = _get_detector()
    try:
        boxes, _probs = detector.detect(image)
    except Exception as e:
        print(f"  face detection failed: {e}", flush=True)
        return None
    if boxes is None or len(boxes) == 0:
        return None

    # MTCNN returns float [x1, y1, x2, y2] boxes. Convert to int (x, y, w, h),
    # then pick the largest by area.
    w_img, h_img = image.size
    candidates: list[FaceBox] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(w_img, int(round(x2)))
        y2 = min(h_img, int(round(y2)))
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w > 0 and h > 0:
            candidates.append((x1, y1, w, h))
    if not candidates:
        return None
    return max(candidates, key=lambda b: b[2] * b[3])


def detect_largest_face_from_path(path: Path) -> Optional[FaceBox]:
    """Convenience: open ``path`` and delegate to :func:`detect_largest_face`."""
    with Image.open(path) as img:
        return detect_largest_face(img.convert("RGB"))
