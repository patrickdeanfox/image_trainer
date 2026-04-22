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
```

Or just run `./launch.sh` — it bootstraps the venv on first run, does `pip install -e .`, and opens the GUI. `./launch.sh --reset` wipes and reinstalls the venv.

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

## GUI

Six-tab notebook, one tab per pipeline step: **Settings → Ingest → Caption → Review → Train → Generate**. The GUI owns no ML logic — every action spawns `python -m image_trainer.cli <args>` and tails its stdout into the log pane.

Review tab keyboard shortcuts: **←/→** prev/next, **I** toggle include, **Ctrl+S** save.

Train tab: Start / Resume / Stop (graceful). The Stop button sends SIGINT; training finishes its current step, writes a checkpoint, and exits. Links to `logs/`, latest `training_*.log`, `journal.txt`, and `logs/validation/`.

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

GUI exposes: resolution, LoRA rank, grad accum, xformers toggle, TE LoRA toggle, max steps, checkpoint steps, validation steps. The rest live in `config.json` — edit it by hand when needed.

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
- `src/image_trainer/gui.py` — Tkinter wizard. Owns no ML logic.
- `src/image_trainer/pipeline/` — `ingest`, `resize`, `caption`, `caption_wd14`, `review`, `insights`, `train`, `generate`.
- `scripts/legacy_*.py` — pre-refactor standalones; reference only, do not import.
- `launch.sh` — venv bootstrap + GUI launcher. Portable; safe to commit.
- `image_trainer.desktop` — Linux desktop entry. Machine-specific (absolute paths). Gitignored.

See `CLAUDE.md` for the detailed design notes — every `train.py` knob, the cache/resume contract, the two training code paths, and the GUI↔CLI subprocess plumbing.
