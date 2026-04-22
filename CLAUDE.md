# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Local end-to-end pipeline for training a personal-likeness **SDXL LoRA** and generating images with it. Nothing is uploaded. The app has two faces on the same core: a `trainer` CLI and a Tkinter wizard that shells into that CLI.

The primary hardware target is **10 GB VRAM / 16 GB RAM / 20 GB swap**, with overnight training as a first-class workflow. Quality of the final generated image is the explicit top-priority — every design trade-off was made in that direction first, then constrained by the 10 GB budget.

## Layout

```
src/image_trainer/
  config.py            # Project dataclass (per-project config.json) + ProjectsRoot helper
  cli.py               # `trainer {init,prep,caption,review,review-summary,train,generate,gui,list}`
  gui.py               # Tkinter wizard: project browser + 6-step notebook
  pipeline/
    ingest.py          # copy supported images from a source folder into project/raw/
    resize.py          # scale shortest side -> 1024, center-crop, write NNNN.png
    caption.py         # BLIP-large, prepend trigger word, write sibling .txt files
    review.py          # pre-training review model (include/caption/notes per image)
    insights.py        # PIL-only image-quality helpers (stats, near-dup perceptual hash)
    train.py           # SDXL LoRA training loop (quality + overnight hardening)
    generate.py        # load base + LoRA, run inference, save under outputs/<ts>/
scripts/
  legacy_resize_images.py   # original pre-refactor scripts, kept for reference only
  legacy_caption_images.py
```

Per-project data lives **outside the repo**, one directory per LoRA run. `Project.root` is the only path the user supplies; every other path is derived:
`raw/`, `processed/`, `cache/`, `checkpoints/`, `lora/`, `outputs/`, `logs/`, `logs/validation/`, `logs/journal.txt`, `config.json`, `review.json`.

The default **projects root** is `~/Apps/image_trainer/projects/` (see `DEFAULT_PROJECTS_ROOT` in `config.py`). The CLI resolves a bare name like `me` against that root; an absolute path is taken verbatim.

## Pipeline

Six steps, strictly ordered, each reachable through both CLI and GUI:

1. `trainer init <project_dir_or_name> [--trigger-word X] [--base <sdxl.safetensors>] [--rank N] [--resolution N]` — create the folder layout and `config.json`.
2. `trainer prep <project> [--source <folder>]` — optional ingest, then resize to `target_size` PNGs (`0000.png`, `0001.png`, …).
3. `trainer caption <project>` — BLIP captions prepended with `trigger_word`, written as sibling `.txt` files. **Requires CUDA.**
4. `trainer review <project>` — launches the GUI focused on the **Review** tab so you can include/exclude, edit the caption, and use quick-tag chips per image. Persists to `review.json`; `trainer review-summary <project>` prints counts for CI.
5. `trainer train <project> [--resume] [--max-steps N] [--rank N] [--resolution N] [--grad-accum N] [--note "..."]` — LoRA training. Training reads `review.json` and **only trains on `include=True` stems**.
6. `trainer generate <project> --prompt "..." [--n 4] [--steps 30] [--guidance 7] [--seed N]` — base + LoRA inference.

Plus `trainer gui` (launch wizard), `trainer list` (print projects), and `trainer review-summary` (print include/exclude counts).

Filenames are positional: `caption.py` writes `img_path.with_suffix(".txt")`, so a `.png` and its `.txt` pair by sharing a stem. Don't rename one without the other.

## `train.py` — what every knob does

### Quality (default-on)

- **PEFT LoRA on UNet** — `target_modules=["to_k","to_q","to_v","to_out.0"]`, rank **32** alpha 32 default.
- **PEFT LoRA on both text encoders** — opt-in via `project.train_text_encoder` (GUI checkbox). When on, the training loop takes a second, parallel code path (`_build_live_dataset` + `_encode_live` in `train.py`) that caches only VAE latents, keeps both CLIP text encoders resident on GPU, and re-encodes captions every step so LoRA weight updates take effect. TE LoRA defaults — `target_modules=["q_proj","v_proj","k_proj","out_proj"]`, `te_lora_rank=8`, `te_lora_alpha=8`, `te_learning_rate=5e-5`, `te_gradient_checkpointing=True` — are lower than UNet's because TEs need less capacity and are more LR-sensitive. A preflight step (`_preflight_te_lora`) runs one forward+backward before the main loop and warns if peak VRAM tops 9.5 GB.
- **min-SNR-gamma loss weighting** (`min_snr_gamma=5.0`) — down-weights easy timesteps; noticeable quality gain on faces for free.
- **Offset noise** (`offset_noise=0.05`) — improves contrast and dark backgrounds.
- **Cosine LR schedule** with `lr_warmup_steps=50`, base LR `1e-4`.
- **Shuffled-epoch iteration** — each "epoch" (n-image pass) is re-shuffled via `random.Random(project.seed)`, rather than `step % n`. Matters once you have >N steps per image.

### Memory (default-on for 10 GB)

- **Pre-compute VAE latents + text embeddings** once in `_cache_embeddings_and_latents` (split into `_cache_vae_latents` + `_cache_text_embeddings` under the hood), cache to `project/cache/`, then drop the VAE + both text encoders from GPU before the training loop. Without this, SDXL does not fit on 10 GB. **Exception:** when `train_text_encoder=True`, the TE-caching half is skipped and both encoders stay on GPU (they need to train); only VAE latents are cached, and VAE is still offloaded.
- **fp16 mixed precision** via `accelerate`.
- **Gradient checkpointing** on the UNet.
- **8-bit AdamW** via `bitsandbytes` (soft fallback to `torch.optim.AdamW` with a warning if bnb is missing — do not silently remove the fallback).
- **xformers** memory-efficient attention when available, sdpa otherwise.
- **Gradient accumulation** (`gradient_accumulation_steps`) wired through `accelerate` so raising it past 1 is a straightforward VRAM↔wall-time trade.

### Overnight hardening

- **Signal handling.** `SIGINT` (and `SIGTERM` where supported) flip `stop_requested`; the loop finishes its current step, writes a checkpoint, and exits. User expectation: Ctrl+C is safe, never data-destructive.
- **Periodic checkpoints** every `checkpointing_steps` (default 100) to `checkpoints/step_<N>/` via `accelerator.save_state`.
- **Resume** via `_find_latest_checkpoint` picks the highest `step_N` and calls `accelerator.load_state`. `start_step` becomes N; the step loop uses `step % n` shuffling so dataset iteration doesn't need to rewind.
- **tee'd log** — `_tee_stdout(log_path)` writes every training line to `logs/training_<ts>.log` in addition to stdout. Don't remove; users need it to audit an overnight run.
- **Validation previews.** If `validation_steps > 0`, every N steps the loop runs `pipe(...)` with `project.effective_validation_prompt()` and a fixed seed, saving PNGs into `logs/validation/step_<N>.png`. This is inference during training, so it temporarily moves the (CPU-resident) VAE and TEs back onto GPU — currently done by `pipe.to(device)` inside `_run_validation`. Watch VRAM carefully if you change this.

### SDXL-specific gotchas baked into the loop

- `added_cond_kwargs` must include `text_embeds` (pooled) and `time_ids`. Because every training image is pre-cropped to `project.resolution`, `time_ids` is a fixed `[res, res, 0, 0, res, res]`.
- `prompt_embeds` is concat of `hidden_states[-2]` from both text encoders (L + bigG). Pooled embeds come from `text_encoder_2`.
- Base checkpoint loading supports both Civitai-style `.safetensors` via `from_single_file` and HF directories via `from_pretrained` — same branch in `_load_sdxl_pipeline` and `generate`.

## GUI ↔ CLI contract

`gui.py` owns **no** model/training/inference logic. Every action spawns `python -m image_trainer.cli <args>` via `_run_cli`, which tees the subprocess's stdout into a `queue.Queue`. The Tk main loop drains the queue on a 100 ms timer (`_drain_log`), so the UI thread never blocks.

The log pump also does a crude parse of `step N/M ...` lines to drive the training progress bar — if you change the training log format, update that parser too.

If you add a new pipeline step, add it in this order: module under `pipeline/` → `cli.py` subcommand → GUI button/tab that shells into it. Don't invert.

### Project browser

`ProjectsRoot` in `config.py` is the list/create backbone. The GUI's top bar shows the current root, the dropdown lists every subdirectory that contains `config.json`, and "New..." prompts for a name and calls `ProjectsRoot.create`. Switching projects calls `Project.load` and mirrors settings into the widgets.

### OOM knobs exposed in the GUI

Curated set, per user directive: **resolution**, **LoRA rank**, **gradient accumulation**, **xformers on/off**, **text-encoder LoRA on/off**. Power-user knobs (`mixed_precision`, `use_8bit_optim`, `min_snr_gamma`, `offset_noise`, `learning_rate`, `lr_scheduler`, `lr_warmup_steps`, and the TE-LoRA tuners `te_lora_rank`, `te_lora_alpha`, `te_learning_rate`, `te_gradient_checkpointing`) stay in `config.json` only — editing that file is the documented escape hatch.

### Review tab

Owned by `pipeline/review.py`. `review.json` at the project root is the single source of truth for per-image `{include, caption, notes}`. `review.load(project)` is lenient: if no `review.json` exists, every processed image is seeded with `include=True` and the caption pulled from its `.txt`. `review.save(project, review)` writes `review.json` and **re-writes each stem's `.txt` from the edited caption** (deleting `.txt` for excluded stems belt-and-suspenders). Training then keeps reading `.txt` as before — the review layer is transparent to the training loop, which additionally filters to only `include=True` stems in `_cache_embeddings_and_latents`.

`pipeline/insights.py` is PIL-only (no torch, no OpenCV): `image_stats`, `average_hash` + `hamming` for near-duplicate pairs, and `resolution_warning`. The GUI calls `find_near_duplicates` once on Review-tab reload and indexes results per stem; O(n²) is fine for personal datasets (<500 images).

`Project.prompt_chips` is a per-project editable list of quick-insert tokens for the Review tab. `append_chip(caption, chip)` (in `review.py`) adds to a comma-separated caption without duplicating.

### Training journal

`train.append_journal(project, note, extra)` appends one line per run to `logs/journal.txt`. The Train tab has a "Journal note" field that feeds `--note` to the CLI.

## `Project` config lifecycle

- `Project.save()` / `load()` round-trip every field through `config.json`.
- `load()` is **forward-compatible**: unknown keys are silently dropped and missing keys fall back to dataclass defaults, so upgrading the code doesn't break an older project.
- When you add a field to `Project`, update defaults and docs; `save/load` require nothing extra.
- `ProjectsRoot.create(name, **overrides)` is the one place a new `Project` comes into existence. Prefer that over constructing `Project` directly outside tests.

## Environment

- Python 3.10+, install via `pip install -e .`.
- Dependencies are declared in `pyproject.toml`, not a requirements file.
- `bitsandbytes` and `xformers` are technically optional (soft-fallback code paths exist) but expected for the 10 GB budget. A `pyproject.toml` `[project.optional-dependencies]` group `xformers` is defined.
- The gitignore excludes `*.safetensors`, `*.ckpt`, `*.pt`, `models/`, all `training_data/` subfolders, `ComfyUI/`, `kohya_ss/`, venvs, and IDE files. Never commit checkpoints, caches, user images, or generated outputs.

## Conventions

- No test suite, no linter config in this repo. Don't invent one unless asked.
- All tunables live on `Project`. Don't scatter module-level constants.
- Error surface: data-prep + generation raise plain exceptions; training catches `SIGINT`/`SIGTERM` but lets other exceptions propagate. The GUI tails subprocess stdout — errors show up there.
- Legacy `scripts/legacy_*.py` are kept for reference and should not be imported from the package.
