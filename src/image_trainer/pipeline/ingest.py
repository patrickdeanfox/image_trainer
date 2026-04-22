"""Copy user-supplied source images into a project's `raw/` folder.

This is step 2a of the pipeline. It's a plain file copy with two safeguards:
unsupported suffixes are skipped, and existing destination files are left
alone so re-running the step is idempotent. Actual resizing happens in
`resize.py` — keeping the two separate lets the Tkinter wizard call either
one independently.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .resize import SUPPORTED_SUFFIXES


def ingest_source(source: Path, raw_dir: Path) -> list[Path]:
    """Copy every image in `source` whose suffix is supported into `raw_dir`.

    Behavior:
    - Only files with a suffix in :data:`resize.SUPPORTED_SUFFIXES` are copied
      (``.jpg``, ``.jpeg``, ``.png``, ``.webp``). Everything else is silently
      skipped — this function is intentionally tolerant so the user can point
      at a mixed folder.
    - Files that already exist at the destination are left untouched (re-runs
      are safe).
    - Progress is printed line-buffered so the GUI's log pump can mirror it.

    Returns:
        The list of destination paths that were newly copied this call (not
        including files that already existed).

    Raises:
        NotADirectoryError: if `source` is not a directory.
    """
    source = Path(source)
    raw_dir = Path(raw_dir)
    if not source.is_dir():
        raise NotADirectoryError(f"source is not a directory: {source}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for p in sorted(source.iterdir()):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        dst = raw_dir / p.name
        if dst.exists():
            print(f"Skipping (exists): {p.name}", flush=True)
            continue
        shutil.copy2(p, dst)
        copied.append(dst)
        print(f"Imported: {p.name}", flush=True)

    print(f"Ingested {len(copied)} image(s) into {raw_dir}", flush=True)
    return copied
