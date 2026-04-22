"""The ``trainer`` command-line entry point.

This module defines the :func:`build_parser` argparse tree and the
``_cmd_*`` handlers that back each subcommand. It intentionally owns no
model code — every subcommand imports its pipeline module lazily so that
``trainer --help`` stays cheap and failures in one step don't break the
whole CLI.

The GUI shells into these same subcommands via
:class:`image_trainer.gui.CLIRunner`, so every handler:

- prints progress line-buffered (``flush=True``) so the GUI log pump can
  tail it cleanly;
- avoids holding global process state between calls; and
- persists changes to ``config.json`` before handing off to the pipeline,
  so a crash mid-step leaves the project consistent for the next run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Project, ProjectsRoot


def _resolve_project_dir(raw: str) -> Path:
    """Turn a user-supplied project identifier into an absolute path.

    - An absolute path or a path that exists on disk is taken verbatim.
    - Anything containing a path separator is assumed to be a relative path
      and is expanded against the CWD.
    - A bare name (like ``me``) is resolved against the default
      :class:`ProjectsRoot`, i.e. ``~/Apps/image_trainer/projects/me``.

    Always returns a resolved, absolute :class:`Path`.
    """
    path = Path(raw).expanduser()
    if path.is_absolute() or path.exists() or "/" in raw or "\\" in raw:
        return path.resolve()
    return (ProjectsRoot().root / raw).resolve()


def _cmd_init(args: argparse.Namespace) -> None:
    """Handler for ``trainer init``: scaffold a new project on disk."""
    root = _resolve_project_dir(args.project_dir)
    project = Project(
        root=root,
        trigger_word=args.trigger_word,
        base_model_path=Path(args.base).expanduser().resolve() if args.base else None,
    )
    if args.rank is not None:
        project.lora_rank = args.rank
        project.lora_alpha = args.rank
    if args.resolution is not None:
        project.resolution = args.resolution
    project.ensure_dirs()
    path = project.save()
    print(f"Initialized project at {project.root}")
    print(f"Config: {path}")


def _cmd_prep(args: argparse.Namespace) -> None:
    """Handler for ``trainer prep``: optional ingest, then resize to ``processed/``.

    ``--no-face-crop`` overrides ``project.face_aware_crop`` to disable the
    face-aware rule-of-thirds crop for this run. Persists the choice to
    config.json so subsequent runs honour it until flipped again.

    ``--dry-run`` writes 256px thumbnails to ``<project>/preview/`` instead of
    full-resolution PNGs to ``processed/``. The face-aware decisions run the
    same way, so this is a cheap audit of which images need manual review
    before committing to a real prep. Review JSON is not mutated.
    """
    from .pipeline.ingest import ingest_source
    from .pipeline.resize import resize_dataset

    project = Project.load(_resolve_project_dir(args.project_dir))
    if args.no_face_crop:
        project.face_aware_crop = False
        project.save()
    if args.source:
        ingest_source(Path(args.source).expanduser().resolve(), project.raw_dir)

    dry_run = bool(getattr(args, "dry_run", False))
    if dry_run:
        dst = project.root / "preview"
        dst.mkdir(parents=True, exist_ok=True)
        print(
            f"[dry-run] writing 256px previews to {dst} — no changes to "
            f"processed/ or review.json.",
            flush=True,
        )
    else:
        dst = project.processed_dir

    result = resize_dataset(
        project.raw_dir,
        dst,
        target_size=project.target_size,
        face_aware=project.face_aware_crop,
        dry_run=dry_run,
    )

    # Any image where face detection was attempted but failed gets marked
    # include=False in review.json so the user is forced to look at it on
    # the Review tab before training. A missing detector (user hasn't
    # installed the [face] extra) is a global choice and is NOT treated as
    # a per-image failure.
    #
    # Dry-run explicitly skips this: the user hasn't committed to a crop
    # yet, so we don't want to mutate review.json.
    if not dry_run and result.face_failed_stems:
        from .pipeline import review as review_mod

        review = review_mod.load(project)
        for stem in result.face_failed_stems:
            entry = review.entries.get(stem)
            if entry is None:
                continue
            entry.include = False
            note = "prep: no face detected"
            entry.notes = (
                note if not entry.notes else f"{entry.notes}; {note}"
            )
        review_mod.save(project, review)


def _cmd_caption(args: argparse.Namespace) -> None:
    """Handler for ``trainer caption``: run the configured captioner(s).

    Respects ``Project.captioner``:

    - ``blip`` → BLIP sentence only.
    - ``wd14`` → WD14 Danbooru-style tag list only (NSFW-aware).
    - ``both`` (default) → BLIP sentence + WD14 tags concatenated.

    ``--mode`` overrides the project setting for a single run.
    """
    project = Project.load(_resolve_project_dir(args.project_dir))
    mode = (args.mode or project.captioner or "both").lower()

    if mode == "blip":
        from .pipeline.caption import caption_dataset
        caption_dataset(
            project.processed_dir,
            trigger_word=project.trigger_word,
            model_id=project.caption_model_id,
        )
    elif mode == "wd14":
        from .pipeline.caption_wd14 import caption_dataset_wd14
        caption_dataset_wd14(
            project.processed_dir,
            trigger_word=project.trigger_word,
            model_id=project.wd14_model_id,
        )
    elif mode == "both":
        from .pipeline.caption_wd14 import caption_dataset_both
        caption_dataset_both(
            project.processed_dir,
            trigger_word=project.trigger_word,
            blip_model_id=project.caption_model_id,
            wd14_model_id=project.wd14_model_id,
        )
    else:
        raise SystemExit(f"Unknown --mode {mode!r}; expected blip | wd14 | both.")


def _cmd_train(args: argparse.Namespace) -> None:
    """Handler for ``trainer train``: run the SDXL LoRA training loop.

    ``--resume`` disallows *changing* ``--rank``/``--resolution``/``--base``
    (passing the same value as what's already saved is fine). Changing any of
    those would silently invalidate the LoRA tensor shapes in the last
    checkpoint or the VAE-latent / text-embedding cache, so users get a clear
    error instead of a silent training corruption.
    """
    from .pipeline.train import train_lora

    project = Project.load(_resolve_project_dir(args.project_dir))

    # On --resume we must not silently mutate settings that would invalidate
    # the saved checkpoint's LoRA shape or the cache. Forbid the unsafe flags.
    if args.resume:
        unsafe = []
        if args.rank is not None and args.rank != project.lora_rank:
            unsafe.append("--rank")
        if args.resolution is not None and args.resolution != project.resolution:
            unsafe.append("--resolution")
        if args.base:
            new_base = Path(args.base).expanduser().resolve()
            if new_base != project.base_model_path:
                unsafe.append("--base")
        if unsafe:
            raise SystemExit(
                f"Cannot change {', '.join(unsafe)} during --resume: the checkpoint "
                f"and cache were produced with the old values. Either (a) drop "
                f"--resume and start fresh, or (b) edit config.json and delete "
                f"the cache/ and checkpoints/ folders manually."
            )

    changed = False
    if args.base:
        project.base_model_path = Path(args.base).expanduser().resolve()
        changed = True
    if args.rank is not None:
        project.lora_rank = args.rank
        project.lora_alpha = args.rank
        changed = True
    if args.resolution is not None:
        project.resolution = args.resolution
        changed = True
    if args.grad_accum is not None:
        project.gradient_accumulation_steps = args.grad_accum
        changed = True
    if changed:
        project.save()
    train_lora(
        project,
        resume=args.resume,
        max_steps_override=args.max_steps,
        note=args.note or "",
    )


def _cmd_generate(args: argparse.Namespace) -> None:
    """Handler for ``trainer generate``: load base + trained LoRA and save images."""
    from .pipeline.generate import generate

    project = Project.load(_resolve_project_dir(args.project_dir))
    generate(
        project,
        prompt=args.prompt,
        negative=args.negative or "",
        n=args.n,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )


def _cmd_gui(args: argparse.Namespace) -> None:
    """Handler for ``trainer gui``: launch the Tkinter wizard."""
    from .gui import launch

    initial_tab = getattr(args, "tab", None)
    project_dir = getattr(args, "project_dir", None)
    launch(
        initial_project_dir=Path(project_dir).expanduser().resolve() if project_dir else None,
        initial_tab=initial_tab,
    )


def _cmd_review(args: argparse.Namespace) -> None:
    """Handler for ``trainer review``: launch GUI focused on the Review tab."""
    from .gui import launch

    launch(
        initial_project_dir=_resolve_project_dir(args.project_dir),
        initial_tab="review",
    )


def _cmd_review_summary(args: argparse.Namespace) -> None:
    """Handler for ``trainer review-summary``: print include/exclude counts.

    Intended for shell scripts and CI; the GUI's Review tab already shows
    this in its header.
    """
    from .pipeline import review as review_mod

    project = Project.load(_resolve_project_dir(args.project_dir))
    s = review_mod.summary(project)
    print(f"total={s['total']} included={s['included']} excluded={s['excluded']}")


def _cmd_list(_args: argparse.Namespace) -> None:
    """Handler for ``trainer list``: print every project under the default root."""
    pr = ProjectsRoot()
    for p in pr.list_projects():
        print(p)


def _cmd_clean(args: argparse.Namespace) -> None:
    """Handler for ``trainer clean``: delete large artifacts.

    By default deletes only ``cache/`` (pre-computed VAE latents + text
    embeddings) and ``checkpoints/`` (intermediate ``step_N/`` dirs). Both
    are safe to regenerate: ``cache/`` rebuilds on next training run, and
    checkpoints only matter while you're still planning to ``--resume``.

    ``--all`` additionally deletes ``raw/`` and ``processed/``. Note: ``raw/``
    is NOT automatically regeneratable — it holds the images ``prep``
    imported. Only use ``--all`` when you've finalised the LoRA and are sure
    your source images still exist elsewhere.

    Always preserved: ``lora/``, ``outputs/``, ``logs/``, ``config.json``,
    ``review.json``.
    """
    import os
    import shutil
    import sys

    project = Project.load(_resolve_project_dir(args.project_dir))
    targets = [project.cache_dir, project.checkpoints_dir]
    if args.all:
        targets.extend([project.raw_dir, project.processed_dir])

    existing = [p for p in targets if p.exists()]
    if not existing:
        print("Nothing to clean.")
        return

    # Symlink-safe size estimation: don't follow links out of the project.
    def _du(p: Path) -> int:
        total = 0
        for dirpath, _dirnames, filenames in os.walk(p, followlinks=False):
            for name in filenames:
                try:
                    total += os.lstat(os.path.join(dirpath, name)).st_size
                except OSError:
                    pass
        return total

    for p in existing:
        size_gb = _du(p) / 1e9
        print(f"  {p}  ({size_gb:.2f} GB)")

    # Auto-require --yes when stdin isn't a tty (e.g. launched by the GUI),
    # otherwise `input()` blocks forever on a piped stdin.
    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit(
                "Refusing to delete without --yes when stdin is not a terminal."
            )
        resp = input("Delete the above? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    for p in existing:
        shutil.rmtree(p)
        print(f"Removed {p}")
    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level :mod:`argparse` tree for the ``trainer`` CLI.

    Kept as a named function (not inlined in :func:`main`) so tests and the
    README code examples can introspect the parser without running it.
    """
    p = argparse.ArgumentParser(prog="trainer", description="Local SDXL LoRA training pipeline.")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init", help="Create a project directory and config.")
    sp.add_argument("project_dir", help="Path or bare name under the projects root.")
    sp.add_argument("--trigger-word", default="ohwx person")
    sp.add_argument("--base", default=None, help="Path to base SDXL .safetensors")
    sp.add_argument("--rank", type=int, default=None)
    sp.add_argument("--resolution", type=int, default=None)
    sp.set_defaults(func=_cmd_init)

    sp = sub.add_parser("prep", help="Ingest a source folder and resize to processed/.")
    sp.add_argument("project_dir")
    sp.add_argument("--source", default=None, help="Folder of raw images to import first")
    sp.add_argument(
        "--no-face-crop",
        action="store_true",
        help=(
            "Disable the face-aware rule-of-thirds crop for this run and "
            "fall back to centre-crop. Persists to config.json."
        ),
    )
    sp.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Preview mode: write 256px thumbnails to <project>/preview/ "
            "instead of full-res PNGs to processed/. Face-aware decisions "
            "run normally so you can audit crops cheaply; review.json is "
            "not touched."
        ),
    )
    sp.set_defaults(func=_cmd_prep)

    sp = sub.add_parser(
        "caption",
        help="Caption processed/ with BLIP, WD14, or both (default: both).",
    )
    sp.add_argument("project_dir")
    sp.add_argument(
        "--mode",
        choices=["blip", "wd14", "both"],
        default=None,
        help="Override Project.captioner for this run. Default: use the project's setting.",
    )
    sp.set_defaults(func=_cmd_caption)

    sp = sub.add_parser("train", help="Train a LoRA on processed/ images.")
    sp.add_argument("project_dir")
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--max-steps", type=int, default=None)
    sp.add_argument("--base", default=None, help="Override base SDXL .safetensors path")
    sp.add_argument("--rank", type=int, default=None)
    sp.add_argument("--resolution", type=int, default=None)
    sp.add_argument("--grad-accum", type=int, default=None)
    sp.add_argument("--note", default=None, help="Short note appended to logs/journal.txt for this run.")
    sp.set_defaults(func=_cmd_train)

    sp = sub.add_parser("generate", help="Generate images with base + trained LoRA.")
    sp.add_argument("project_dir")
    sp.add_argument("--prompt", required=True)
    sp.add_argument("--negative", default="")
    sp.add_argument("--n", type=int, default=4)
    sp.add_argument("--steps", type=int, default=30)
    sp.add_argument("--guidance", type=float, default=7.0)
    sp.add_argument("--seed", type=int, default=None)
    sp.set_defaults(func=_cmd_generate)

    sp = sub.add_parser("gui", help="Launch the Tkinter wizard.")
    sp.set_defaults(func=_cmd_gui)

    sp = sub.add_parser(
        "review", help="Launch the GUI focused on the Review tab for a project."
    )
    sp.add_argument("project_dir")
    sp.set_defaults(func=_cmd_review)

    sp = sub.add_parser(
        "review-summary",
        help="Print how many processed images are marked include/exclude.",
    )
    sp.add_argument("project_dir")
    sp.set_defaults(func=_cmd_review_summary)

    sp = sub.add_parser("list", help="List projects under the default projects root.")
    sp.set_defaults(func=_cmd_list)

    sp = sub.add_parser(
        "clean",
        help="Delete regeneratable artifacts (cache/, checkpoints/) to reclaim disk.",
    )
    sp.add_argument("project_dir")
    sp.add_argument(
        "--all",
        action="store_true",
        help="Also delete raw/ and processed/. The trained LoRA and logs are always preserved.",
    )
    sp.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    sp.set_defaults(func=_cmd_clean)

    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point used by both ``python -m image_trainer.cli`` and the
    ``trainer`` console script registered in ``pyproject.toml``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
