from __future__ import annotations

import shutil
from pathlib import Path

from .resize import SUPPORTED_SUFFIXES


def ingest_source(source: Path, raw_dir: Path) -> list[Path]:
    """Copy supported image files from source into raw_dir.
    Existing files with the same name are skipped. Returns list of copied paths."""
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
