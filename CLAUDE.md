# CLAUDE.md

Instructions for Claude Code sessions working in this repo. Skim it before editing anything — the design choices here were made for a specific hardware envelope and reversing one often breaks another.

## Purpose

Local end-to-end pipeline for training a personal-likeness **SDXL LoRA** and generating images with it. Nothing is uploaded. Two faces on one core: a `trainer` CLI and a Tkinter wizard (`gui.py`) that shells into that CLI.

Primary hardware target: **10 GB VRAM / 16 GB RAM / 20 GB swap**. Overnight training is a first-class workflow. Final image quality is the top priority; every design trade-off was made in that direction first, then constrained by the 10 GB budget.

## Layout

```
src/image_trainer/
  __init__.py
  config.py            # Project dataclass (config.json per project) + ProjectsRoot helper
  cli.py               # `trainer {init,prep,caption,review,review-summary,train,generate,gui,list,clean}`
  gui.py               # Tkinter wizard: project browser + 6-step notebook
  pipeline/
    __init__.py
    ingest.py          # copy supported images into project/raw/
    resize.py          # scale shortest side → target_size, center-crop, write NNNN.png
    caption.py         # BLIP-large sentence captioner
    caption_wd14.py    # WD14 Danbooru-style tagger (NSFW-aware)
    review.py          # pre-training review model + persistence
    insights.py        # PIL-only image-quality helpers (stats, perceptual hash, near-dup)
    train.py           # SDXL LoRA training loop (UNet + optional text-encoder LoRA)
    generate.py        # load base + LoRA, run inference
scripts/
  legacy_resize_images.py   # pre-refactor standalones — reference only, do not import
  legacy_caption_images.py
launch.sh               # bootstraps venv + runs `trainer gui`
image_trainer.desktop   # machine-specific, gitignored
```

Per-project data lives **outside the repo**, one directory per LoRA run. `Project.root` is the only path the user supplies; every other path is derived. The default **projects root** is `~/Apps/image_trainer/projects/` (`DEFAULT_PROJECTS_ROOT` in `config.py`). The CLI resolves a bare name like `me` against that root; an absolute path is taken verbatim.

## CLI surface

Ten subcommands. Every pipeline action is available on both the CLI and the GUI; the GUI only shells out.

| Subcommand | Purpose |
| --- | --- |
| `trainer init <project> [--trigger-word X] [--base F] [--rank N] [--resolution N]` | Scaffold a new project dir + `config.json`. |
| `trainer prep <project> [--source DIR]` | Optional ingest from `--source` into `raw/`, then resize into `processed/`. |
| `trainer caption <project> [--mode blip\|wd14\|both]` | Generate captions. `--mode` overrides `project.captioner`. **Requires CUDA.** |
| `trainer review <project>` | Launch the GUI focused on the Review tab for `<project>` (wraps `_cmd_gui` with `initial_tab="review"`). |
| `trainer review-summary <project>` | Print `total=N included=M excluded=K` (for scripts/CI). |
| `trainer train <project> [--resume] [--max-steps N] [--base F] [--rank N] [--resolution N] [--grad-accum N] [--note "..."]` | LoRA training. `--resume` is mutually exclusive with `--rank`/`--resolution`/`--base` — changing any of those invalidates the cache/checkpoints. |
| `trainer generate <project> --prompt "..." [--negative "..."] [--n 4] [--steps 30] [--guidance 7.0] [--seed N]` | Base + LoRA inference into `outputs/<YYYYMMDD_HHMMSS>/`. |
| `trainer gui` | Launch wizard (no args). The `review` alias reuses the same handler with `initial_tab="review"`. |
| `trainer list` | Print every project under `ProjectsRoot.root`. |
| `trainer clean <project> [--all] [--yes]` | Delete regeneratable state. Default removes `cache/` + `checkpoints/`; `--all` also removes `raw/` + `processed/`. Always preserves `lora/`, `outputs/`, `logs/`, `config.json`, `review.json`. Auto-requires `--yes` if stdin is not a tty (prevents GUI hangs). |

CLI flags that overlap `Project` fields are persisted to `config.json` before the handler runs, so the next run inherits them unless overridden again.

## Pipeline

Six ordered steps, each reachable from both CLI and GUI:

1. **Init** — `trainer init` creates the folder layout and `config.json`.
2. **Prep** — `trainer prep` copies supported images (`.jpg/.jpeg/.png/.webp`) from `--source` into `raw/` (idempotent, `shutil.copy2`), then resizes to `target_size` square PNGs in `processed/` with zero-padded names (`0000.png`, `0001.png`, …). Resize algorithm: scale shortest side to `target_size` with LANCZOS, then center-crop.
3. **Caption** — `trainer caption` dispatches on `project.captioner`:
   - `blip`: BLIP-large generates a short sentence; writes `"{trigger_word}, {sentence}"` to `<stem>.txt`.
   - `wd14`: WD14 ONNX tagger returns general + character tags (rating tags always dropped) above confidence thresholds (defaults: general 0.35, character 0.85); writes `"{trigger_word}, tag1, tag2, …"`.
   - `both` (default): concatenate — `"{trigger_word}, {blip sentence}, tag1, tag2, …"`. Best for NSFW-ish datasets where BLIP alone under-describes anatomy.
4. **Review** — `trainer review` opens the GUI on the Review tab. Per-image `{include, caption, notes}` is persisted to `review.json` at the project root and mirrored back to `processed/<stem>.txt` on save. Excluded stems' `.txt` files are deleted belt-and-suspenders; training additionally filters to `include=True` stems when caching embeddings/latents.
5. **Train** — `trainer train` runs the LoRA loop. See `train.py` section below.
6. **Generate** — `trainer generate` loads the base + trained LoRA, enables `enable_model_cpu_offload()`, and saves `n` images under `outputs/<YYYYMMDD_HHMMSS>/000.png`, `001.png`, …. A fixed `--seed` produces a deterministic sequence.

File naming is positional: captioners write `img_path.with_suffix(".txt")`, so a `.png` and its `.txt` pair by sharing a stem. Don't rename one without the other.

## `train.py` — what every knob does

### Quality (default-on)

- **PEFT LoRA on UNet** — `target_modules=["to_k","to_q","to_v","to_out.0"]`, rank **32** alpha 32 default.
- **PEFT LoRA on both text encoders** — opt-in via `project.train_text_encoder` (GUI checkbox). When on, the training loop takes a second, parallel code path (`_build_live_dataset` + live TE encoding in `train.py`) that caches only VAE latents, keeps both CLIP text encoders resident on GPU, and re-encodes captions every step (tokens are pre-tokenized; only TE forward is hot). TE LoRA defaults — `target_modules=["q_proj","v_proj","k_proj","out_proj"]`, `te_lora_rank=8`, `te_lora_alpha=8`, `te_learning_rate=5e-5`, `te_gradient_checkpointing=True` — are lower than UNet's because TEs need less capacity and are more LR-sensitive. A preflight step (`_preflight_te_lora`) runs one forward+backward before the main loop and warns if peak VRAM tops 9.5 GB. Preflight failures are logged but non-fatal.
- **min-SNR-gamma loss weighting** (`min_snr_gamma=5.0`) — down-weights easy timesteps; free quality gain on faces.
- **Offset noise** (`offset_noise=0.05`) — improves contrast and dark backgrounds.
- **Cosine LR schedule** with `lr_warmup_steps=50`, base LR `1e-4` (UNet), `5e-5` (TE).
- **Shuffled-epoch iteration** — each n-image pass is re-shuffled via `random.Random(project.seed)`. On `--resume`, the RNG is fast-forwarded by `completed_epochs` so the shuffle matches what a fresh run would have produced.

### Memory (default-on for 10 GB)

- **Pre-compute VAE latents + text embeddings** once in `_cache_embeddings_and_latents` (split into `_cache_vae_latents` + `_cache_text_embeddings` under the hood), cache to `project/cache/latents/` and `project/cache/embeds/` respectively, then drop the VAE and both text encoders from GPU before the training loop. Without this, SDXL does not fit on 10 GB. **Exception:** when `train_text_encoder=True`, the TE-caching half is skipped and both encoders stay on GPU (they need to train); only VAE latents are cached, and VAE is still offloaded.
- **Cache invalidation marker** — `cache/cache_marker.json` stores the `(resolution, base_model_path)` pair used to generate the cache. `_validate_or_reset_cache` nukes `latents/` and `embeds/` if either changed.
- **fp16 mixed precision** via `accelerate`.
- **Gradient checkpointing** on the UNet.
- **8-bit AdamW** via `bitsandbytes` (soft fallback to `torch.optim.AdamW` with a warning if bnb is missing — do not silently remove the fallback).
- **xformers** memory-efficient attention when available, sdpa otherwise.
- **Gradient accumulation** (`gradient_accumulation_steps`) wired through `accelerate` so raising it past 1 is a straightforward VRAM↔wall-time trade. `train_batch_size` is hardcoded to 1.

### Overnight hardening

- **Signal handling.** `SIGINT` and `SIGTERM` flip `stop_requested`; the loop finishes its current step, writes a checkpoint, and exits. User expectation: Ctrl+C is safe, never data-destructive.
- **Periodic checkpoints** every `checkpointing_steps` (default 100) to `checkpoints/step_<N>/` via `accelerator.save_state`.
- **Resume** via `_find_latest_checkpoint` picks the highest `step_N` and calls `accelerator.load_state`. `start_step` becomes N; the RNG is fast-forwarded to match.
- **Tee'd log.** `_tee_stdout(log_path)` writes every training line to `logs/training_<unix_ts>.log` in addition to stdout. Don't remove; users need it to audit overnight runs. A new log file is created per run.
- **Journal.** `append_journal(project, note, extra)` writes one line to `logs/journal.txt` per run — timestamp, rank, resolution, LR, step count, `--note`. Append-only; users can hand-edit.
- **Validation previews.** If `validation_steps > 0`, every N steps the loop runs `pipe(...)` with `project.effective_validation_prompt()` and a fixed seed, saving PNGs into `logs/validation/step_<N>.png`. This is inference during training, so it temporarily moves the (CPU-resident) VAE and TEs back onto GPU — currently done by `pipe.to(device)` inside `_run_validation`. Watch VRAM carefully if you change this. Non-fatal exceptions are logged and training continues.

### SDXL-specific gotchas baked into the loop

- `added_cond_kwargs` must include `text_embeds` (pooled) and `time_ids`. Because every training image is pre-cropped to `project.resolution`, `time_ids` is a fixed `[res, res, 0, 0, res, res]`.
- `prompt_embeds` is concat of `hidden_states[-2]` from both text encoders (CLIP-L + CLIP-bigG). Pooled embeds come from `text_encoder_2`.
- Base checkpoint loading supports both Civitai-style `.safetensors` via `from_single_file` and HF directories/model-IDs via `from_pretrained` — the same branch is in `_load_sdxl_pipeline` (train.py) and `generate.py`. Type isn't validated up-front; the dispatch happens at load time.

### LoRA export

At end of training, `train.py` writes subdirectories under `lora/`:

- `lora/unet/` — always.
- `lora/text_encoder/` — only if `train_text_encoder=True`.
- `lora/text_encoder_2/` — only if `train_text_encoder=True`.

`generate.py` calls `pipe.load_lora_weights(project.lora_dir)`, which dispatches to whichever subdirs exist.

## GUI ↔ CLI contract

`gui.py` owns **no** model/training/inference logic. Every action spawns `python -m image_trainer.cli <args>` via `CLIRunner`, which tees the subprocess's stdout/stderr into a `queue.Queue`. The Tk main loop drains the queue on a 100 ms timer (`_drain_log`), so the UI thread never blocks.

The log pump also does a crude regex parse of `step N/M ...` lines to drive the training progress bar — if you change the training log format, update that parser too.

If you add a new pipeline step, add it in this order: module under `pipeline/` → `cli.py` subcommand → GUI button/tab that shells into it. Don't invert.

### Project browser

`ProjectsRoot` in `config.py` is the list/create backbone. The GUI's top bar shows the current root, the dropdown lists every subdirectory that contains `config.json`, and "New..." prompts for a name and calls `ProjectsRoot.create`. Switching projects calls `Project.load` and mirrors settings into the widgets.

### Tabs

1. **Settings (01)** — project name, trigger word, base model, LoRA rank, resolution, grad accum, xformers toggle, TE LoRA toggle, max steps, checkpointing steps, validation steps. "Save settings" writes to `config.json`.
2. **Ingest (02)** — pick source folder; dispatches `trainer prep --source <dir>`.
3. **Caption (03)** — dispatches `trainer caption`.
4. **Review (04)** — owned by `pipeline/review.py`. Listbox of stems on the left; preview + caption editor + "Include in training" + chips bar + stats + near-dup flags + notes on the right. Keyboard shortcuts: ←/→ (prev/next), I (toggle include), Ctrl+S (save). Stats + hashes are computed once on tab load; `find_near_duplicates` is O(n²) but fine for personal-scale data.
5. **Train (05)** — Start / Resume / Stop (graceful) buttons, progress bar, optional journal note field, links to logs folder / latest log / `journal.txt` / validation folder.
6. **Generate (06)** — prompt / negative / n / steps / guidance / seed; link to outputs folder.

### OOM knobs exposed in the GUI

Curated set, per user directive: **resolution**, **LoRA rank**, **gradient accumulation**, **xformers on/off**, **text-encoder LoRA on/off**. Also exposed: max steps, checkpointing steps, validation steps.

Config-only (no GUI widget — edit `config.json` directly): `train_batch_size` (hardcoded 1), `mixed_precision` (hardcoded "fp16"), `use_8bit_optim`, `min_snr_gamma`, `offset_noise`, `learning_rate`, `lr_scheduler`, `lr_warmup_steps`, `te_lora_rank`, `te_lora_alpha`, `te_learning_rate`, `te_gradient_checkpointing`, `vram_profile` (currently unused — placeholder for future).

### Review tab internals

`review.json` at the project root is the single source of truth for per-image `{include, caption, notes}`. `review.load(project)` is lenient: if no `review.json` exists (or it's corrupt), every processed image is seeded with `include=True` and the caption pulled from its `.txt`; orphaned entries (PNG deleted) are pruned. `review.save(project, review)` writes `review.json` and **re-writes each included stem's `.txt` from the edited caption**, deleting `.txt` for excluded stems. Training then keeps reading `.txt` as before — the review layer is transparent to the training loop, which additionally filters to only `include=True` stems in `_cache_embeddings_and_latents`.

`pipeline/insights.py` is PIL-only (no torch, no OpenCV): `image_stats` (brightness + Laplacian pstdev sharpness), `average_hash` + `hamming` for near-duplicate pairs, `stats_and_hash` (single-open optimization), `resolution_warning`, and `find_near_duplicates` (default threshold 6). The GUI calls it once on Review-tab reload and indexes results per stem.

`Project.prompt_chips` is a per-project editable list of quick-insert tokens for the Review tab. `append_chip(caption, chip)` in `review.py` adds to a comma-separated caption without duplicating.

## `Project` config lifecycle

- `Project.save()` / `load()` round-trip every field through `config.json`. Path fields stringify on save.
- `load()` is **forward-compatible**: unknown keys are silently dropped and missing keys fall back to dataclass defaults, so upgrading the code doesn't break an older project.
- When you add a field to `Project`, update defaults and docs here; `save/load` require nothing extra.
- `ProjectsRoot.create(name, **overrides)` is the one place a new `Project` comes into existence outside tests.
- `base_model_path` can hold either a `.safetensors` file path or a HF Hub model ID string. The distinction is handled at load time (`from_single_file` vs `from_pretrained`) — do not validate the type early.

## Environment

- Python 3.10+, install via `pip install -e .` (or `pip install -e ".[wd14]"` to unlock WD14 captioning, or `pip install -e ".[xformers]"` for xformers).
- Dependencies are declared in `pyproject.toml`, not a requirements file.
- `bitsandbytes` is core; `xformers` and WD14 extras (`onnxruntime` + `huggingface_hub`) are optional with soft-fallback code paths.
- The gitignore excludes venvs, packaging artifacts (`*.egg-info/`, `build/`, `dist/`), dev tooling caches (`.mypy_cache/`, `.pytest_cache/`, …), secrets (`.env*`), model formats (`*.safetensors`, `*.ckpt`, `*.pt`, `*.pth`, `*.bin`, `*.onnx`), `models/`, `training_data/*`, `ComfyUI/`, `kohya_ss/`, IDE files, `outputs/`, `logs/`, and the machine-specific `image_trainer.desktop` launcher. Never commit checkpoints, caches, user images, or generated outputs.
- `launch.sh` bootstraps `.venv/` on first run and launches `trainer gui`. It's portable (resolves its own path) and safe to commit.

## Conventions

- No test suite, no linter config in this repo. Don't invent one unless asked.
- All tunables live on `Project`. Don't scatter module-level constants except true protocol values (WD14 thresholds, SDXL time_ids shape, LoRA target-module sets).
- Error surface: data-prep + generation raise plain exceptions; training catches `SIGINT`/`SIGTERM` but lets other exceptions propagate. The GUI tails subprocess stdout — errors show up there.
- `trainer clean` is the safe way to reclaim disk space mid-project. Don't delete `cache/` by hand without also clearing `cache_marker.json`, or the next training run will trust stale tensors.
- Legacy `scripts/legacy_*.py` are kept for reference and should not be imported from the package.
