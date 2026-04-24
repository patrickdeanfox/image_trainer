# image_trainer

Local end-to-end pipeline for training a personal-likeness **SDXL LoRA**, generating images with it (with or without extra civitai LoRAs stacked on top), and producing image-to-video output overnight via Wan2GP. CLI first, Tkinter wizard on top. Everything runs on your own GPU; nothing is uploaded.

Hardware envelope: **10 GB VRAM / 16 GB RAM / 20 GB swap**. Overnight training is a first-class workflow. Linux + CUDA assumed throughout.

## Install

```bash
git clone git@github.com:patrickdeanfox/image_trainer.git
cd image_trainer
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                    # core (BLIP captioner only)
pip install -e ".[wd14]"            # + WD14 NSFW-aware tag captioner
pip install -e ".[prompts]"         # + compel for long-prompt support (>77 tokens)
pip install facenet-pytorch --no-deps   # + face-aware prep crop (see note below)
```

Or just run `./launch.sh` — it bootstraps the venv on first run, does `pip install -e .`, and opens the GUI. `./launch.sh --reset` wipes and reinstalls the venv.

### xformers — install carefully (not from PyPI)

xformers gives you ~15-20% faster training via memory-efficient attention. **Do NOT use `pip install xformers` from PyPI** — that wheel is built against a specific torch version and will silently overwrite your working torch install with a mismatched one. The classic symptom is `ImportError: undefined symbol: ncclCommWindowDeregister` the next time you import torch.

Install matched wheels from the official PyTorch index instead. Pick the line for your CUDA version (`nvidia-smi` top-right shows it):

```bash
# CUDA 12.1 (most common)
.venv/bin/pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 xformers==0.0.28.post1

# CUDA 12.4
.venv/bin/pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.1 torchvision==0.19.1 xformers==0.0.28.post1
```

The training loop's `use_xformers` flag soft-falls back to PyTorch SDPA when xformers is missing, so skipping this step is fine — you just lose the speedup. If you already broke your install by running `pip install xformers` from PyPI, run the matched-wheel command above with `--force-reinstall` and it'll repair the breakage.

**Why `--no-deps` on `facenet-pytorch`?** It pins `torch<2.3.0`, which would downgrade the torch + torchvision your trainer uses and break `bitsandbytes` and the rest of the pipeline. MTCNN doesn't actually need an old torch — it works fine on the current torch. `--no-deps` installs just the MTCNN code and weights without touching anything else. If face-aware crop is missing, prep silently falls back to centre-crop.

Optional system tools (only needed for the Video tab):

- `ffmpeg` — `sudo apt install ffmpeg`
- `realesrgan-ncnn-vulkan` — standalone binary from [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases)
- `rife-ncnn-vulkan` — standalone binary from [rife-ncnn-vulkan releases](https://github.com/nihui/rife-ncnn-vulkan/releases)
- `systemd-inhibit` — ships with systemd; used to keep the desktop awake during overnight runs

## Base checkpoint

You supply this — the app never downloads one. Grab any SDXL-family `.safetensors` and point the app at the file (or at a HuggingFace model ID / local directory; both `from_single_file` and `from_pretrained` are dispatched at load time).

Recommendations baked into the Generate tab's sidebar:

| Base | Strength |
| --- | --- |
| **Pony Diffusion V6 XL** | Industry standard for NSFW. Tag-driven (`score_9, score_8_up, …`). Least pushback. |
| **Lustify XL** | Pony fine-tune dialled toward photoreal NSFW. |
| **Pony Realism / Pony Real** | Other Pony fine-tunes — community variants vary in skin / anatomy bias. |
| **RealVisXL V4 / V5** | Photoreal SDXL fine-tune. Tighter on NSFW; pair with the uncensor negative preset. |
| **JuggernautXL** | Versatile; fewer Pony-isms. |
| **Illustrious-XL / NoobAI-XL** | Anime / illustration NSFW. Different tag vocabulary (`masterpiece, best quality, very aware`). |

Convention: put bases in `~/Apps/image_trainer/models/`, keep projects in `~/Apps/image_trainer/projects/` (the default projects root), and drop community LoRAs in `~/Apps/image_trainer/projects/shared_loras/` so they're discoverable across all projects.

**Use the same base at training and generation time.** A LoRA trained on Pony and applied to RealVis will look subtly wrong.

## Quickstart — CLI

Each project is one LoRA-training run. Projects live outside the repo; only the name is user-visible.

```bash
# 1. Scaffold.
trainer init me \
  --base ~/Apps/image_trainer/models/ponyDiffusionV6XL.safetensors \
  --trigger-word "ohwx person"

# 2. Import source images, resize to 1024×1024 PNGs.
#    Default: face-aware rule-of-thirds crop (requires facenet-pytorch installed --no-deps).
trainer prep me --source ~/Pictures/me_dataset
trainer prep me --source ~/Pictures/me_dataset --no-face-crop      # centre-crop fallback
trainer prep me --source ~/Pictures/me_dataset --dry-run           # 256px previews only

# 3. Caption. Default mode = "both" (BLIP sentence + WD14 tags).
trainer caption me                                  # use project.captioner
trainer caption me --mode wd14 --nsfw               # NSFW preset: lower threshold + suffix
trainer caption me --general-threshold 0.25 --extra-suffix "explicit, nsfw, soft lighting"

# 4. Review (GUI). Include/exclude images, edit captions, note dupes.
trainer review me
trainer review-summary me                # scripted counts

# 5. Train. Ctrl+C is safe — it checkpoints and exits cleanly.
trainer train me --max-steps 1500 --note "first pass"
trainer train me --resume                # pick up the latest checkpoint
trainer train me --max-steps 200 --limit-images 1   # quick sanity-check run

# 6. Generate. Trained LoRA + optional civitai extras.
trainer generate me --prompt "ohwx person, portrait" --steps 28 --sampler dpmpp_2m_karras
trainer generate me --prompt "..." --width 832 --height 1216
trainer generate me --prompt "..." --no-trained-lora                   # vanilla base
trainer generate me --prompt "..." --extra-lora ~/loras/anime.safetensors:0.7

# 7. Video post-processing. After Wan2GP produces a raw .mp4:
trainer video-post me ~/wan2gp_outputs/raw.mp4 \
  --framerate 32 --rife-multiplier 2 --upscale-scale 2
```

`trainer gui` launches the wizard on the same commands. `trainer list` prints every project under `~/Apps/image_trainer/projects/`. `trainer clean me` reclaims disk space (see "Reclaiming disk space" below).

## GUI

Eight-tab notebook: **Settings → Ingest → Caption → Review → Train → Generate → Storage → Video**. The GUI owns no ML logic — every action spawns `python -m image_trainer.cli <args>` and tails its stdout into the log pane.

Theme: warm cream parchment ground, jewel-tone accents (burgundy primary, olive secondary, slate text, soft gold highlight). Sentence case throughout. Hover any **ⓘ** glyph for ~1 second to see a one-sentence explanation of any technical field.

### Header strip

Single compact row at the top: title, project combo, **New…**, **Refresh**, **Recent ▾** (MRU list of the last eight projects you opened), **Root…** (change the projects root), six colour-coded status dots — one per pipeline step (Settings, Ingest, Caption, Review, Train, Generate) — that track whether that step has been run, partially run, or is clean. Each notebook tab also carries a live badge — e.g. `02 · Ingest · 187 imgs`, `04 · Review · 132/186`, `05 · Train · ckpt(3)`.

### Review tab (04)

Dual-column include/exclude layout — Included on the left (selection bar = olive), Excluded on the right (selection bar = burgundy). Filter bar across the top: **All / Faces / No-face / Unknown** (driven by the `face_detected` bit prep writes into `review.json`). Each row shows a face glyph (☺ / · / blank) and the stem name. Selection follows the cursor as you toggle include — items move across columns smoothly.

Two view modes — the dual-column detail view above, plus a thumbnail grid toggle for at-a-glance dupe-spotting. Per-image stats (brightness, sharpness, near-duplicate hashes) load on a background thread so the tab opens instantly even with hundreds of images.

Keyboard shortcuts (scoped to the Review tab only): **←/→** prev/next, **I** toggle include, **Ctrl+S** save.

### Train tab (05)

Start / Resume / Stop (graceful). The Stop button sends SIGINT; training finishes its current step, writes a checkpoint, and exits.

- **Images per run** picker — All / 1 / 5 / 10 / 25 / 50 / 100 (or any int). Caps how many of your included images this run trains on; great for "I have 15 minutes" sanity runs without losing the cache.
- **Free VRAM** button — lists every process holding GPU memory via `nvidia-smi`, excludes this app + the active training subprocess if running, kills the rest with SIGTERM (escalates to SIGKILL).
- **Pre-spawn TE-LoRA gate** — if Text-encoder LoRA is on AND your card has < 12 GB, the Start button shows a "won't fit" dialog with a one-click fix that disables TE LoRA, saves settings, and runs immediately.
- Live metrics row: step counter, **ETA** (computed from training-phase rate), **elapsed** (counts from subprocess launch through caching + model load), **VRAM** (used/total MiB, polled every ~2 s), Canvas-based loss sparkline, journal-note entry.

### Generate tab (06)

NSFW-focused two-column layout:

- **Quality-tag prefix picker** — Pony score_9 (real), Pony score_9 (anime), Illustrious / NoobAI tag stack, bare SDXL realism, none. Auto-prepended to your prompt at generate time.
- **Prompt template library** — eight pre-written skeletons (portrait soft window light, boudoir bedroom, nude standing, nude explicit POV, outdoor golden hour, selfie phone POV, cosplay, lingerie photoshoot). Each auto-injects the project's trigger word.
- **Negative prompt presets** — Standard quality, NSFW · uncensor (pushes against censoring artefacts), NSFW · realism push, Anime push.
- **Sampler picker** — `default / euler / euler_a / dpmpp_2m / dpmpp_2m_karras / unipc`.
- **Aspect ratio quick-pick** — five SDXL-friendly buckets (832×1216 portrait → 1216×832 landscape).
- **LoRA stack** — toggle "Use this project's trained LoRA," then multi-select from a shared library at `~/Apps/image_trainer/projects/shared_loras/` with per-LoRA weight spinboxes (0.0–2.0). Import button copies new `.safetensors` into the library.
- **Live preview** of the assembled prompt below the body so you see exactly what gets sent.
- **Recommendations sidebar** with NSFW-friendly base checkpoints and civitai LoRA categories worth searching for (detail enhancer / anatomy correction / realistic skin / lighting / pose / style).

### Storage tab (07)

Per-folder size + file count for `raw/`, `processed/`, `cache/`, `checkpoints/`, `lora/`, `outputs/`, `logs/`, `preview/`. Each row gets an Open button + a Delete button (Caution-styled red for non-regenerable, Ghost for safe-to-rebuild like cache). Bottom **Danger zone** wipes the entire project root after a two-step confirmation (typed project-name match required), drops it from Recent, and snaps back to the project picker.

### Video tab (08)

Overnight image-to-video pipeline scaffold around Wan2GP + Real-ESRGAN + RIFE + ffmpeg. Two flows:

- **Flow A · Generate (Wan2GP).** Wan2GP install/launch panel (clones the repo, sets up an isolated venv, pip-installs requirements, persists the install path in per-user settings). Pre-flight tools checklist with green/red status for ffmpeg / realesrgan-ncnn-vulkan / rife-ncnn-vulkan / systemd-inhibit. **Free VRAM** + **Inhibit sleep** buttons for overnight prep. Source image + motion prompt + Wan2GP knobs (target seconds, block swap, window frames, overlap, Lightning LoRA). **Copy generation plan** button puts a paste-ready Wan2GP plan onto your clipboard.
- **Flow B · Post-process.** Point at the raw .mp4 Wan2GP wrote, pick upscale model + scale + RIFE multiplier + final fps, click **Run post-process**. Frames are extracted via ffmpeg → upscaled by `realesrgan-ncnn-vulkan` → interpolated by `rife-ncnn-vulkan` → re-encoded into `<project>/video/<timestamp>/final.mp4`. Every intermediate is preserved on disk so a crash mid-run keeps prior phases.

Wan2GP itself runs as its own Gradio app — the GUI does not try to drive it directly. Open Wan2GP, paste the plan, click Generate, walk away. In the morning, switch to flow B and point at the .mp4.

### Caption tab (03)

Mode picker (BLIP / WD14 / BLIP+WD14), WD14 general + character threshold spinboxes, **NSFW / lewd dataset preset** (lowers general threshold to 0.25, seeds the suffix with `explicit, nsfw, detailed anatomy`), and a free-text **Extra suffix** field for stylistic anchors like `photorealistic, soft lighting`. All persisted to `Project`.

### Settings saves show a diff

Clicking **Save settings** computes a field-level diff against the on-disk `config.json` and shows it in a confirm dialog (`• field: old → new`). When any of `resolution`, `lora_rank`, `lora_alpha`, or `base_model_path` change, the dialog warns that cache + checkpoints will be invalidated on the next training run.

### Telemetry pane

The bottom log pane is collapsible (caret toggle on its header) and colour-tagged: `error` in red, `warn` in gold, `info` in cyan, training `step` lines in amber, and meta lines in muted grey.

## Training knobs worth knowing

Defaults are tuned for a person LoRA on an SDXL realistic base at 10 GB VRAM.

| Knob | Default | Why |
| --- | --- | --- |
| `lora_rank` / `lora_alpha` (UNet) | 32 / 32 | Enough capacity for identity. Drop to 16 on OOM. |
| `train_text_encoder` | False | Opt-in. Needs ~12 GB; pre-spawn gate refuses on smaller cards. |
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
| `caption_general_threshold` | 0.35 | WD14 general-tag confidence cutoff. NSFW preset → 0.25. |
| `caption_character_threshold` | 0.85 | WD14 character-tag cutoff (high to suppress). |
| `caption_extra_suffix` | "" | Tokens appended to every caption. |
| `caption_nsfw_preset` | False | Toggles the lower threshold + anatomy suffix. |
| `face_aware_crop` | True | Face-detected rule-of-thirds crop in prep; centre-crop fallback otherwise. |

GUI exposes: resolution, LoRA rank, grad accum, xformers toggle, TE LoRA toggle, max steps, checkpoint steps, validation steps. Ingest tab also exposes the face-aware crop checkbox. Caption tab exposes the captioner mode + WD14 thresholds + NSFW preset + extra suffix. The rest live in `config.json` — edit it by hand when needed.

## Per-project filesystem

```
~/Apps/image_trainer/projects/me/
├── config.json              # all project state
├── review.json              # {stem: {include, caption, notes, face_detected}}
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
├── video/YYYYMMDD_HHMMSS/   # video pipeline scratch space + final.mp4
└── logs/
    ├── training_<unix>.log  # full stdout/stderr per training run
    ├── journal.txt          # append-only one-liner per run
    └── validation/step_<N>.png
```

Sibling to the projects: `~/Apps/image_trainer/projects/shared_loras/` (community LoRAs reusable across projects) and `~/Apps/image_trainer/projects/.user_settings.json` (per-user prefs like Wan2GP install path).

## Reclaiming disk space

```bash
trainer clean me            # delete cache/ + checkpoints/ (regeneratable)
trainer clean me --all      # also delete raw/ + processed/ — only after the LoRA is final
```

Always preserves `lora/`, `outputs/`, `logs/`, `config.json`, `review.json`. Prompts for confirmation unless `--yes`. The Storage tab in the GUI gives the same fine-grained control with size readouts and a danger-zone "delete entire project" option.

## Troubleshooting

**OOM during training.** Already mitigated — the UNet stays in fp16 on GPU (the base weights are frozen for LoRA training; only PEFT's small fp32 LoRA adapters need to train). Earlier code mistakenly upcast the UNet to fp32 which is ~10 GB of weights alone — won't fit on a 10 GB card. If you still OOM, drop resolution 1024 → 768 (invalidates cache), drop `lora_rank` 32 → 16, raise `gradient_accumulation_steps`, disable `train_text_encoder`, or keep `use_xformers=True`.

**`ImportError: undefined symbol: ncclCommWindowDeregister`.** Your torch + NCCL stack is mismatched, almost always because something pip-installed `xformers` (or another torch-dependent package) and quietly pulled a different torch version. Recovery is in the **xformers — install carefully** section above: pip uninstall the entire torch + nvidia-* stack, then reinstall a matched triad from PyTorch's `--index-url`.

**`ValueError: infer_schema(func): Parameter q has unsupported type torch.Tensor`.** Your diffusers is too new for torch 2.4.1. Diffusers 0.33+ introduced `attention_dispatch.py` using PEP 604 union syntax that older torch doesn't parse. Fix: `pip install "diffusers>=0.30,<0.32"`. The pyproject already pins this range, so a fresh `pip install -e .` produces the right version.

**Generate fails with `Error no file named pytorch_lora_weights.bin found`.** Old issue from before training learned to write the diffusers-flat LoRA format. Re-run training; the loop now writes both `lora/pytorch_lora_weights.safetensors` (what generate loads) AND `lora/unet/adapter_model.safetensors` (PEFT format for `--resume`). If you only have the PEFT-format directory from a stale training run, re-run training to regenerate the flat file.

**Generated images look smooth / airbrushed / ignore half the prompt.** Almost always SDXL CLIP's 77-token cap silently truncating your prompt. Check the CLI output for `WARNING: prompt is ~N tokens; CLIP's 77-token cap…`. Fix: install compel via `pip install -e ".[prompts]"`. Generate then chunks long prompts and your full description reaches the model.

**TE-LoRA refuses to start.** Intentional — the pre-spawn gate refuses TE LoRA on cards under 12 GB because it OOMs reliably. Override (not recommended): `IMAGE_TRAINER_FORCE_TE_LORA=1 trainer train me`.

**OOM during validation previews.** `_run_validation` moves VAE/TEs back to GPU temporarily; drop `validation_steps` to 0 to skip them.

**Base checkpoint won't load.** The loader tries `from_single_file` for `.safetensors` and `from_pretrained` for directories / HF Hub IDs. If you have a `.ckpt`, convert it to `.safetensors` via `diffusers`' converter first.

**WD14 captioning errors.** Install the extra: `pip install -e ".[wd14]"`. For CUDA: `pip install onnxruntime-gpu huggingface_hub`. Or fall back with `--mode blip`.

**Stale cache after changing base or resolution.** Cache is auto-invalidated via `cache_marker.json`; if you touched it by hand, run `trainer clean` and let it rebuild.

**Training crashed overnight.** Resume: `trainer train me --resume`. The highest `checkpoints/step_N/` is loaded; RNG is fast-forwarded so the epoch shuffle matches what a fresh run would have produced.

**Custom civitai LoRA fails to load.** Some LoRAs use unusual key prefixes diffusers doesn't auto-recognise. The error in the Telemetry pane lists the bad keys; usually the fix is to use that LoRA's recommended workflow on its civitai page.

**Wan2GP launch button is greyed out.** The status line tells you what's missing. Run **Install / update** first; it clones the repo + sets up the venv. Model weights download from inside Wan2GP on first generation.

**Face filter shows 0 faces despite obvious faces.** Project was prepped before the `face_detected` field existed. The Review tab now back-fills the bit on load when `face_aware_crop=True` — just open Review and it self-heals. Future preps write the bit directly.

## What's in this repo

- `src/image_trainer/config.py` — `Project` dataclass + `ProjectsRoot`. Adds `caption_*`, `face_aware_crop`, etc.
- `src/image_trainer/cli.py` — eleven subcommands (`init`, `prep`, `caption`, `review`, `review-summary`, `train`, `generate`, `video-post`, `gui`, `list`, `clean`).
- `src/image_trainer/gui.py` — back-compat shim that re-exports from `gui_app`.
- `src/image_trainer/gui_app.py` — main window, header, notebook, status dots, telemetry pane.
- `src/image_trainer/gui_theme.py` — frozen `Theme` dataclass with `.parchment()` (default) / `.dark()` factories.
- `src/image_trainer/gui_runner.py` — `CLIRunner` (subprocess + SIGINT plumbing).
- `src/image_trainer/gui_widgets.py` — `StatusDot`, `Sparkline`, `FolderField`, `CollapsibleFrame`, `ThumbnailGrid`, `Tooltip`, `info_icon`.
- `src/image_trainer/gui_helpers.py` — `parse_step_line`, `probe_vram`, `list_gpu_processes`, `kill_processes`, `folder_size_and_count`, recent-projects MRU, user-settings, shared-LoRA helpers.
- `src/image_trainer/wan2gp_installer.py` — clone / venv / pip-install / launch helpers for Wan2GP.
- `src/image_trainer/tabs/` — one module per pipeline step: `settings_tab`, `prep_tab`, `caption_tab`, `review_tab`, `train_tab`, `generate_tab`, `storage_tab`, `video_tab`.
- `src/image_trainer/pipeline/` — `ingest`, `resize`, `face_detect`, `caption`, `caption_wd14`, `review`, `insights`, `train`, `generate`, `video`.
- `scripts/legacy_*.py` — pre-refactor standalones; reference only, do not import.
- `launch.sh` — venv bootstrap + GUI launcher. Portable; safe to commit.
- `image_trainer.desktop` — Linux desktop entry. Machine-specific (absolute paths). Gitignored.

See `CLAUDE.md` for the detailed design notes — every `train.py` knob, the cache/resume contract, the two training code paths, and the GUI↔CLI subprocess plumbing.
