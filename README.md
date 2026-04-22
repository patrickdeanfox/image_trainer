# image_trainer

Local end-to-end pipeline for training a personal-likeness **SDXL LoRA** and generating images with it. CLI first, Tkinter wizard on top. Everything runs on your own GPU; nothing is uploaded.

Hardware envelope: **10 GB VRAM / 16 GB RAM / 20 GB swap**. Overnight training is a first-class workflow. Linux + CUDA assumed throughout.

## Install

```bash
git clone git@github.com:patrickdeanfox/image_trainer.git
cd image_trainer
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                    # core (BLIP captioner only)
pip install -e ".[wd14]"            # + WD14 NSFW-aware tag captioner
pip install -e ".[xformers]"        # + xformers memory-efficient attention
pip install facenet-pytorch --no-deps   # + face-aware prep crop (see note below)
```

Or just run `./launch.sh` — it bootstraps the venv on first run, does `pip install -e .`, and opens the GUI. `./launch.sh --reset` wipes and reinstalls the venv.

**Why `--no-deps` on `facenet-pytorch`?** It pins `torch<2.3.0`, which would downgrade the torch + torchvision your trainer uses and break `bitsandbytes` and the rest of the pipeline. MTCNN doesn't actually need an old torch — it works fine on the current torch. `--no-deps` installs just the MTCNN code and weights without touching anything else. If face-aware crop is missing, prep silently falls back to the old centre-crop behaviour.

## Base checkpoint

You supply this — the app never downloads one. Grab any SDXL-family `.safetensors` and point the app at the file (or at a HuggingFace model ID / local directory; both `from_single_file` and `from_pretrained` are dispatched on at load time).

| Base | Strength | Where |
| --- | --- | --- |
| **Juggernaut XL** (v9+) | Photorealism, skin, lighting — safe default | Civitai |
| **RealVisXL V4.0** | Very clean portraits, SFW | HF: `SG161222/RealVisXL_V4.0` |
| **CyberRealistic XL** | Photorealism with NSFW headroom | Civitai |
| **Pony Diffusion V6 XL** | NSFW anatomy / poses; pair with WD14 tags, not BLIP | Civitai |

Convention: put bases in `~/Apps/image_trainer/models/`, keep projects in `~/Apps/image_trainer/projects/` (the default projects root).

**Use the same base at training and generation time.** A LoRA trained on Juggernaut and applied to RealVis will look subtly wrong.

## Quickstart — CLI

Each project is one LoRA-training run. Projects live outside the repo; only the name is user-visible.

```bash
# 1. Scaffold.
trainer init me \
  --base ~/Apps/image_trainer/models/juggernautXL_v9.safetensors \
  --trigger-word "ohwx person"

# 2. Import source images, resize to 1024×1024 PNGs.
#    Default: face-aware rule-of-thirds crop (requires facenet-pytorch installed --no-deps).
#    --no-face-crop falls back to the old centre-crop behaviour.
trainer prep me --source ~/Pictures/me_dataset

# 3. Caption. Default mode = "both" (BLIP sentence + WD14 tags).
trainer caption me                       # use project.captioner
trainer caption me --mode blip           # override for this run
trainer caption me --mode wd14
trainer caption me --mode both

# 4. Review (GUI). Include/exclude images, edit captions, note dupes.
trainer review me
trainer review-summary me                # scripted counts

# 5. Train. Ctrl+C is safe — it checkpoints and exits cleanly.
trainer train me --max-steps 1500 --note "first pass"
trainer train me --resume                # pick up the latest checkpoint

# 6. Generate.
trainer generate me \
  --prompt "ohwx person, studio lighting, portrait" \
  --n 4 --steps 30 --guidance 7 --seed 42
```

`trainer gui` launches the wizard on the same commands. `trainer list` prints every project under `~/Apps/image_trainer/projects/`.

### Dry-run prep

`trainer prep me --dry-run` writes 256px previews to `<project>/preview/` instead of full-resolution PNGs to `processed/`. Face-aware crop decisions run the same way, so you can audit which images will be flagged "no face detected" before committing to a real prep. Nothing is touched in `processed/` or `review.json`. The Ingest tab exposes this as a **DRY-RUN PREVIEW** button next to **IMPORT & RESIZE**.

## GUI

Six-tab notebook, one tab per pipeline step: **Settings → Ingest → Caption → Review → Train → Generate**. The GUI owns no ML logic — every action spawns `python -m image_trainer.cli <args>` and tails its stdout into the log pane.

### Header strip

Single compact row at the top: title, project combo, **NEW…**, **REFRESH**, **RECENT ▾** (MRU list of the last eight projects you opened), **ROOT…** (change the projects root), a status label, and six colour-coded status dots — one per pipeline step — that track whether that step has been run, partially run, or is clean (derived from disk state: `processed/*.png`, `.txt` pair counts, `review.json`, `checkpoints/`, `lora/unet/`, `outputs/`). Each notebook tab also carries a live badge — e.g. `02 · INGEST · 187 imgs`, `03 · CAPTION · 140/187`, `05 · TRAIN · ckpt(3)`, `06 · GENERATE · 4 runs`.

### Review tab

Two view modes — a 3-pane detail view (list / preview / editor) and a thumbnail grid toggle — with per-image include/exclude, caption editor, quick-insert chips, perceptual-hash near-duplicate flags, stats (brightness, sharpness, resolution warnings), and a notes field.

Keyboard shortcuts (scoped to the Review tab only): **←/→** prev/next, **I** toggle include, **Ctrl+S** save.

### Train tab

Start / Resume / Stop (graceful). The Stop button sends SIGINT; training finishes its current step, writes a checkpoint, and exits. Live metrics row: step counter, ETA, elapsed time, and VRAM (used/total MiB, polled from `nvidia-smi` every ~2s). A Canvas-based sparkline plots the loss curve as training progresses. Action shortcuts link to `logs/`, the latest `training_*.log`, `journal.txt`, and `logs/validation/`.

### Settings saves show a diff

Clicking **SAVE SETTINGS** computes a field-level diff against the on-disk `config.json` and shows it in a confirm dialog (`• field: old → new`). When any of `resolution`, `lora_rank`, `lora_alpha`, or `base_model_path` change, the dialog includes a warning that the cache / checkpoints will be invalidated on the next training run.

### Telemetry pane

The bottom log pane is collapsible (caret toggle on its header) and colour-tagged: `error` in red, `warn` in gold, `info` in cyan, training `step` lines in amber, and meta lines in muted grey.

### Folder fields

Every path-valued input in the GUI (source folder, base checkpoint, project root) uses a composite widget with **BROWSE…** and **OPEN** buttons, so any folder is one click away from your system file manager.

## Training knobs worth knowing

Defaults are tuned for a person LoRA on an SDXL realistic base at 10 GB VRAM.

| Knob | Default | Why |
| --- | --- | --- |
| `lora_rank` / `lora_alpha` (UNet) | 32 / 32 | Enough capacity for identity. Drop to 16 on OOM. |
| `train_text_encoder` | False | Opt-in. When True, TE LoRA runs on both CLIPs (rank 8, LR 5e-5); quality gain at higher VRAM. |
| `resolution` | 1024 | Drop to 768 on OOM. Changing resolution invalidates the cache. |
| `train_batch_size` | 1 (hardcoded) | Scale via `gradient_accumulation_steps`. |
| `gradient_accumulation_steps` | 1 | 2/4/8 are straightforward VRAM↔wall-time trades. |
| `max_train_steps` | 1500 | Person LoRA; bump for larger datasets. |
| `checkpointing_steps` | 100 | Frequency of resume-able checkpoints. |
| `validation_steps` | 200 | 0 disables validation previews. |
| `learning_rate` | 1e-4 | Cosine schedule, 50-step warmup. |
| `min_snr_gamma` | 5.0 | Down-weights easy timesteps. 0 disables. |
| `offset_noise` | 0.05 | Contrast/darks. 0 disables. |
| `mixed_precision` | fp16 | Hardcoded. |
| `use_8bit_optim` | True | Soft fallback to `torch.optim.AdamW` if bitsandbytes missing. |
| `use_xformers` | True | Soft fallback to sdpa if xformers missing. |

GUI exposes: resolution, LoRA rank, grad accum, xformers toggle, TE LoRA toggle, max steps, checkpoint steps, validation steps. Ingest tab also exposes the face-aware crop checkbox (`project.face_aware_crop`). The rest live in `config.json` — edit it by hand when needed.

### Prep knobs

| Knob | Default | Why |
| --- | --- | --- |
| `face_aware_crop` | True | Detect the subject's face and place it on a rule-of-thirds intersection (picked from the face's natural quadrant in the source). Images with no detectable face are still written but marked `include=False` in `review.json` so you can eyeball them before training. Falls back silently to centre-crop when `facenet-pytorch` isn't installed. |
| `target_size` | 1024 | Square edge length for the output PNG. SDXL's native resolution. |

## Per-project filesystem

```
~/Apps/image_trainer/projects/me/
├── config.json              # all project state
├── review.json              # {stem: {include, caption, notes}} — synced to .txt on save
├── raw/                     # original source images (idempotent copy)
├── processed/               # NNNN.png + NNNN.txt (1024² square PNGs + captions)
├── cache/
│   ├── cache_marker.json    # tracks (resolution, base_model_path) used to build cache
│   ├── latents/             # <stem>.pt — VAE latents, one per image
│   └── embeds/              # <stem>.pt — text embeds (only when TE LoRA is off)
├── checkpoints/step_N/      # accelerator state for --resume
├── lora/
│   ├── unet/                # PEFT adapter (always)
│   ├── text_encoder/        # (only if train_text_encoder=True)
│   └── text_encoder_2/
├── outputs/YYYYMMDD_HHMMSS/ # generated images, one dir per generate call
└── logs/
    ├── training_<unix>.log  # full stdout/stderr per training run
    ├── journal.txt          # append-only one-liner per run
    └── validation/step_<N>.png
```

## Reclaiming disk space

```bash
trainer clean me            # delete cache/ + checkpoints/ (regeneratable)
trainer clean me --all      # also delete raw/ + processed/ — only after the LoRA is final
```

Always preserves `lora/`, `outputs/`, `logs/`, `config.json`, `review.json`. Prompts for confirmation unless `--yes`.

## Troubleshooting

**OOM during training.** Drop resolution 1024 → 768 (invalidates cache), drop `lora_rank` 32 → 16, raise `gradient_accumulation_steps`, disable `train_text_encoder`, or keep `use_xformers=True`. The TE-LoRA preflight warns if peak VRAM tops 9.5 GB.

**OOM during validation previews.** `_run_validation` moves VAE/TEs back to GPU temporarily; drop `validation_steps` to 0 to skip them.

**Base checkpoint won't load.** The loader tries `from_single_file` for `.safetensors` and `from_pretrained` for directories / HF Hub IDs. If you have a `.ckpt`, convert it to `.safetensors` via `diffusers`' converter first.

**WD14 captioning errors.** Install the extra: `pip install -e ".[wd14]"`. For CUDA: `pip install onnxruntime-gpu huggingface_hub`. Or fall back with `--mode blip`.

**Stale cache after changing base or resolution.** Cache is auto-invalidated via `cache_marker.json`; but if you touched it by hand, run `trainer clean` and let it rebuild.

**Training crashed overnight.** Resume: `trainer train me --resume`. The highest `checkpoints/step_N/` is loaded; RNG is fast-forwarded so the epoch shuffle matches what a fresh run would have produced.

**Ctrl+C in training.** Safe. SIGINT/SIGTERM flip a flag; the loop finishes its current step, writes a checkpoint, and exits.

## What's in this repo

- `src/image_trainer/config.py` — `Project` dataclass (60+ fields, all in `config.json`) + `ProjectsRoot`.
- `src/image_trainer/cli.py` — ten subcommands.
- `src/image_trainer/gui.py` — back-compat shim that re-exports from `gui_app`.
- `src/image_trainer/gui_app.py` — main window, header, notebook, status dots, telemetry pane.
- `src/image_trainer/gui_theme.py` — frozen `Theme` dataclass with `.dark()` / `.light()` factories, platform-aware font resolver, `apply_style(root)`.
- `src/image_trainer/gui_runner.py` — `CLIRunner` (subprocess + SIGINT plumbing).
- `src/image_trainer/gui_widgets.py` — `StatusDot`, `Sparkline`, `FolderField`, `CollapsibleFrame`, `ThumbnailGrid`.
- `src/image_trainer/gui_helpers.py` — `parse_step_line`, `probe_vram`, `config_diff`, recent-projects MRU, `open_folder`/`open_file`.
- `src/image_trainer/tabs/` — one module per pipeline step (`settings_tab`, `prep_tab`, `caption_tab`, `review_tab`, `train_tab`, `generate_tab`).
- `src/image_trainer/pipeline/` — `ingest`, `resize`, `face_detect`, `caption`, `caption_wd14`, `review`, `insights`, `train`, `generate`.
- `scripts/legacy_*.py` — pre-refactor standalones; reference only, do not import.
- `launch.sh` — venv bootstrap + GUI launcher. Portable; safe to commit.
- `image_trainer.desktop` — Linux desktop entry. Machine-specific (absolute paths). Gitignored.

See `CLAUDE.md` for the detailed design notes — every `train.py` knob, the cache/resume contract, the two training code paths, and the GUI↔CLI subprocess plumbing.
