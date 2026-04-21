"""`trainer` CLI. GUI shells into these subcommands, so keep them side-effect-
free w.r.t. global state and print progress line-buffered."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Project, ProjectsRoot


def _resolve_project_dir(raw: str) -> Path:
    """Accept either an absolute path or a bare project name (looked up under
    the default projects root)."""
    path = Path(raw).expanduser()
    if path.is_absolute() or path.exists() or "/" in raw or "\\" in raw:
        return path.resolve()
    return (ProjectsRoot().root / raw).resolve()


def _cmd_init(args: argparse.Namespace) -> None:
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
    from .pipeline.ingest import ingest_source
    from .pipeline.resize import resize_dataset

    project = Project.load(_resolve_project_dir(args.project_dir))
    if args.source:
        ingest_source(Path(args.source).expanduser().resolve(), project.raw_dir)
    resize_dataset(project.raw_dir, project.processed_dir, target_size=project.target_size)


def _cmd_caption(args: argparse.Namespace) -> None:
    from .pipeline.caption import caption_dataset

    project = Project.load(_resolve_project_dir(args.project_dir))
    caption_dataset(
        project.processed_dir,
        trigger_word=project.trigger_word,
        model_id=project.caption_model_id,
    )


def _cmd_train(args: argparse.Namespace) -> None:
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
    train_lora(project, resume=args.resume, max_steps_override=args.max_steps)


def _cmd_generate(args: argparse.Namespace) -> None:
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


def _cmd_gui(_args: argparse.Namespace) -> None:
    from .gui import launch

    launch()


def _cmd_list(_args: argparse.Namespace) -> None:
    pr = ProjectsRoot()
    for p in pr.list_projects():
        print(p)


def build_parser() -> argparse.ArgumentParser:
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
    sp.set_defaults(func=_cmd_prep)

    sp = sub.add_parser("caption", help="Run BLIP captioning over processed/.")
    sp.add_argument("project_dir")
    sp.set_defaults(func=_cmd_caption)

    sp = sub.add_parser("train", help="Train a LoRA on processed/ images.")
    sp.add_argument("project_dir")
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--max-steps", type=int, default=None)
    sp.add_argument("--base", default=None, help="Override base SDXL .safetensors path")
    sp.add_argument("--rank", type=int, default=None)
    sp.add_argument("--resolution", type=int, default=None)
    sp.add_argument("--grad-accum", type=int, default=None)
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

    sp = sub.add_parser("list", help="List projects under the default projects root.")
    sp.set_defaults(func=_cmd_list)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
