"""Pre-training image review.

One source of truth per project: `<project>/review.json` maps image stem →
{include, caption, notes}. The `.txt` caption file next to each processed
image is regenerated from this on save, so the training loop keeps reading
`<stem>.txt` exactly as before and doesn't need to know the review exists.

`load` is lenient: if an image has no review entry, it's seeded from the
existing `.txt` (or the trigger word if the .txt is missing) and defaults to
`include=True`. That means you can run prep + caption once, skim through the
Review tab, and training will skip whatever you marked off.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from ..config import Project

REVIEW_FILENAME = "review.json"


@dataclass
class ReviewEntry:
    stem: str
    include: bool = True
    caption: str = ""
    notes: str = ""
    # Tri-state face-detection bit written by ``prep``:
    #   True  → MTCNN found a face and the image was face-crop'd.
    #   False → MTCNN ran but found no face; image is centre-cropped.
    #   None  → unknown (older review.json, or prep didn't use face detection).
    # Used by the Review tab's "faces / no-face" filter.
    face_detected: Optional[bool] = None


@dataclass
class Review:
    entries: dict[str, ReviewEntry] = field(default_factory=dict)

    def stems_for_training(self) -> list[str]:
        return sorted(s for s, e in self.entries.items() if e.include)

    def included_count(self) -> int:
        return sum(1 for e in self.entries.values() if e.include)

    def excluded_count(self) -> int:
        return sum(1 for e in self.entries.values() if not e.include)

    def face_count(self) -> int:
        return sum(1 for e in self.entries.values() if e.face_detected is True)

    def non_face_count(self) -> int:
        return sum(1 for e in self.entries.values() if e.face_detected is False)


def _review_path(project: Project) -> Path:
    return project.root / REVIEW_FILENAME


def _default_caption(project: Project, stem: str, txt_path: Path) -> str:
    if txt_path.exists():
        return txt_path.read_text().strip()
    return project.trigger_word


def load(project: Project) -> Review:
    """Load review state, seeding entries for any image not yet reviewed.
    Drops entries whose PNG no longer exists so training never uses stale
    references."""
    review = Review()
    path = _review_path(project)
    if path.exists():
        try:
            blob = json.loads(path.read_text())
            for stem, raw in blob.items():
                face_raw = raw.get("face_detected", None)
                face_val: Optional[bool]
                if face_raw is None:
                    face_val = None
                else:
                    face_val = bool(face_raw)
                review.entries[stem] = ReviewEntry(
                    stem=stem,
                    include=bool(raw.get("include", True)),
                    caption=str(raw.get("caption", "")),
                    notes=str(raw.get("notes", "")),
                    face_detected=face_val,
                )
        except Exception as e:
            # Corrupt or hand-edited -> warn and rebuild from disk so the user
            # sees that their previous review state was dropped.
            print(
                f"Warning: {path} is not readable ({e}); "
                f"rebuilding review from processed/.",
                flush=True,
            )

    png_stems = {p.stem for p in project.processed_dir.glob("*.png")}

    # Drop review entries whose image was deleted.
    orphaned = [s for s in review.entries if s not in png_stems]
    for s in orphaned:
        del review.entries[s]

    # Backfill ``face_detected`` for legacy review.json written before this
    # field existed.
    #
    # Step 1: anything with a "no face detected" note is reliably negative.
    # Step 2: if the project ran in face-aware mode, every other entry was
    #   either successfully face-cropped at prep time OR the detector was
    #   unavailable. We can't distinguish those without re-running detection,
    #   but the common case by far is "user is on face-aware mode and ran
    #   prep with the dependency installed" — under that assumption every
    #   non-negative entry is face-positive.
    #
    # The negative inference is more conservative (it only fires when the
    # project explicitly opted into face-aware crop), so users on centre-crop
    # mode don't accidentally get all entries flipped to "face=True". On a
    # fresh prep this whole block is a no-op because ``_cmd_prep`` writes
    # ``face_detected`` directly.
    project_is_face_aware = bool(getattr(project, "face_aware_crop", False))
    for entry in review.entries.values():
        if entry.face_detected is not None:
            continue
        if "no face detected" in (entry.notes or ""):
            entry.face_detected = False
        elif project_is_face_aware:
            entry.face_detected = True

    # Seed any missing entries from the processed folder.
    for png in sorted(project.processed_dir.glob("*.png")):
        if png.stem in review.entries:
            # If caption is blank, pull from .txt as a fallback.
            if not review.entries[png.stem].caption:
                review.entries[png.stem].caption = _default_caption(
                    project, png.stem, png.with_suffix(".txt")
                )
            continue
        review.entries[png.stem] = ReviewEntry(
            stem=png.stem,
            include=True,
            caption=_default_caption(project, png.stem, png.with_suffix(".txt")),
            notes="",
        )

    return review


def save(project: Project, review: Review) -> Path:
    """Persist review.json and sync each entry's caption to its .txt sibling
    so the training loop's existing caption-reading code keeps working.

    Order matters: .txt files are the ground truth training reads from, so we
    update them first and only commit review.json once they're in sync. A
    crash between the two leaves the training loop consistent (it just hasn't
    seen the latest review UI state), which is the safer failure mode.
    """
    project.ensure_dirs()

    # 1. Mirror captions to .txt files (only for included images, so excluded
    # ones don't accidentally get picked up if someone bypasses the review
    # filter).
    for stem, entry in review.entries.items():
        txt_path = project.processed_dir / f"{stem}.txt"
        if entry.include and entry.caption:
            txt_path.write_text(entry.caption)
        else:
            # Remove stale .txt if the image is excluded — belt-and-suspenders.
            try:
                if txt_path.exists():
                    txt_path.unlink()
            except OSError:
                pass

    # 2. Write review.json last. At this point .txt files already match the
    # in-memory review state, so training is safe to run even if this write
    # fails.
    blob = {stem: asdict(entry) for stem, entry in review.entries.items()}
    # Trim dataclass 'stem' redundancy in serialized form.
    for v in blob.values():
        v.pop("stem", None)
    _review_path(project).write_text(json.dumps(blob, indent=2))
    return _review_path(project)


def summary(project: Project) -> dict:
    r = load(project)
    return {
        "total": len(r.entries),
        "included": r.included_count(),
        "excluded": r.excluded_count(),
    }


def append_chip(caption: str, chip: str) -> str:
    """Append a chip token to a caption, keeping the comma-separated format
    and avoiding obvious duplicates."""
    existing = [t.strip() for t in caption.split(",") if t.strip()]
    if chip in existing:
        return caption
    existing.append(chip)
    return ", ".join(existing)
