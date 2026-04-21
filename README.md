# image_trainer

Local end-to-end pipeline for training a personal-likeness **SDXL LoRA** and generating images with it — CLI first, Tkinter wizard on top. Everything runs on your own GPU; nothing is uploaded.

Built for a **10 GB VRAM / 16 GB RAM** budget with overnight training in mind, but the same defaults scale up cleanly.

---

## Install

```bash
git clone <this repo>
cd image_trainer
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Strongly recommended on a 10 GB GPU
pip install xformers bitsandbytes
```

Python 3.10+, a CUDA build of PyTorch, and an SDXL `.safetensors` base checkpoint you have downloaded locally.

## Base checkpoint

You supply this. Download any SDXL-family `.safetensors` you want to build on top of (e.g. a realistic-photography variant from Civitai) and point the app at the file. The app never downloads checkpoints for you.

---

## Quickstart — CLI

Each "project" is one LoRA-training run. Every project gets its own folder and its own `config.json`.

```bash
# 1. Create a project.
trainer init me --base ~/models/mySDXL.safetensors --trigger-word "ohwx person"

# 2. Import source images and resize to 1024x1024.
trainer prep me --source ~/Pictures/me_dataset

# 3. Auto-caption with BLIP.
trainer caption me

# 4. Review each image: pick which to include, edit captions, add quick tags.
#    This opens the GUI focused on the Review tab. Persists to review.json.
trainer review me
#    (optional: print counts for scripting)
trainer review-summary me

# 5. Train overnight (Ctrl+C is safe — saves a checkpoint and exits).
trainer train me --max-steps 1500 --note "rank 32, first pass"

#    Next morning, continue from the last checkpoint:
trainer train me --resume

# 6. Generate images.
trainer generate me \
  --prompt "ohwx person, soft window light, 35mm photo" \
  --n 4
```

`me` is resolved as `~/Apps/image_trainer/projects/me/` by default (or an absolute path, if you pass one).

`trainer list` prints all projects under the default root.

## Quickstart — GUI

```bash
trainer gui
```

The wizard has a **project browser** at the top (create / switch / refresh) and six step tabs:

1. **Settings** — trigger word, base checkpoint, and the curated OOM / quality knobs: **resolution**, **LoRA rank**, **gradient accumulation**, **xformers on/off**, **text-encoder LoRA on/off**, plus training length + checkpoint frequency + validation frequency.
2. **Import & Resize** — pick a source folder, run the copy + resize.
3. **Caption** — run BLIP.
4. **Review** — step through each processed image: include/exclude toggle, editable caption, quick-tag chips, per-image resolution/brightness/sharpness stats, and near-duplicate warnings. Keyboard: `←`/`→` navigate, `I` toggle include, `Ctrl+S` save. Training only uses `include=True` images.
5. **Train** — Start / Resume / Stop (graceful), live progress bar, optional journal note, shortcut to open the `validation/` folder so you can watch quality trend over the night.
6. **Generate** — prompt, negative (pre-filled from `default_negative_prompt`), N, steps, guidance, seed, and an "Open outputs folder" button.

All heavy work is run through the CLI subprocess, so anything you can do in the GUI you can script.

---

## What you get per project

```
~/Apps/image_trainer/projects/me/
  config.json         # settings
  review.json         # per-image include/caption/notes (Review tab)
  raw/                # your source images
  processed/          # 1024x1024 PNGs + paired .txt captions
  cache/              # pre-computed VAE latents + text embeddings (huge VRAM save)
  checkpoints/        # step_100, step_200, ...  (safe resume points)
  lora/               # final trained LoRA weights (diffusers PEFT format)
  outputs/            # generated images, grouped by timestamp
  logs/
    training_<ts>.log # tee'd stdout from each training run
    journal.txt       # one line per training run (rank, res, lr, note, ...)
    validation/       # validation_step_000200.png, ... (quality progression)
```

---

## Quality tuning

Good defaults for a person LoRA on an SDXL realistic base:

| Knob | Default | Why |
| --- | --- | --- |
| `lora_rank` | 32 | Higher = more face detail; 32 fits on 10 GB with the other tricks. |
| `min_snr_gamma` | 5.0 | Standard quality win (Ephraim et al. 2023). Negligible cost. |
| `offset_noise` | 0.05 | Better contrast / dark scenes. |
| `lr_scheduler` | cosine | Smoother convergence than constant. |
| `learning_rate` | 1e-4 | Sensible SDXL LoRA default. |
| `validation_steps` | 200 | Produces a reference image every 200 steps under `logs/validation/`. |
| `max_train_steps` | 1500 | Enough for ~20 images; bump to 2000–3000 for larger sets. |

Tips that consistently improve final-image quality:

- **Caption what varies, not what's invariant.** BLIP gives generic captions; skim `processed/*.txt` and delete descriptors that are *always true of your subject* (hair color, eye color, face shape). The trigger word carries those, so captioning them fights the LoRA.
- **Include diverse framings.** Closeups + half-body + full-body, multiple lighting conditions, a few angles. 20–40 images is a sweet spot.
- **Use the same base checkpoint for training and generation.** LoRAs trained on checkpoint A can look off when applied to checkpoint B.
- **Negative prompt matters.** Start with `low quality, blurry, extra fingers, deformed` and iterate.
- **Watch the validation images.** If the subject starts melting after step N, you've likely overtrained — shorten `max_train_steps` or lower the LR.

---

## Overnight workflow

Designed to be reliable across an 8-hour run without babysitting:

1. **Ctrl+C is graceful.** The training loop traps SIGINT/SIGTERM, finishes the current step, writes a checkpoint, and exits. Restart with `--resume` to continue from that exact step.
2. **Periodic checkpoints.** `checkpointing_steps=100` by default. Even a power loss or OOM late in the run only costs up to 99 steps.
3. **Full log on disk.** Every run tees stdout to `logs/training_<timestamp>.log`. Scroll back in the morning to find the loss trend and any warnings.
4. **Validation previews.** `validation_steps=200` generates a reference image with a fixed seed+prompt every 200 steps. Open `logs/validation/` in a file manager and scrub through to see quality progression.
5. **Close your browser and screensaver-blockers.** Nothing about this training is "networked" — once it starts, it only uses local GPU + disk.

---

## OOM fallback ladder

If `trainer train` crashes with CUDA OOM on 10 GB:

1. Lower `resolution` to **768** in Settings, retrain from scratch.
2. Bump `gradient_accumulation_steps` to 2 or 4 (trades wall time for memory).
3. Drop `lora_rank` from 32 to 16 (marginal quality hit).
4. Disable `xformers` if a version mismatch is the culprit — the loop falls back to PyTorch sdpa.
5. Make sure nothing else is using the GPU (browser hardware accel, another ML process).
6. As a last resort, switch `base_model_path` to an SD 1.5 checkpoint — the pipeline code is the same, but your LoRA will be SD1.5-compatible, not SDXL-compatible.

---

## Troubleshooting

- **"caption_dataset requires a CUDA GPU"** — captioning loads BLIP in fp16 on cuda with no CPU fallback on purpose. Run on a CUDA machine.
- **Base checkpoint won't load** — the loader tries `from_single_file` for `.safetensors` and `from_pretrained` for directories. If you pass a `.ckpt`, convert it to `.safetensors` first (e.g. using `diffusers`' converter).
- **Training hangs at "Caching latents + text embeddings"** — that step runs once per image on first training run. After that it's nearly instant. Delete `cache/` if you've changed `resolution` or captions.
- **Validation images are black or noise** — usually a broken VAE in fp16 on some cards; the generation loop uses the full pipeline, so retry inference (step 5) with `guidance 5–6` and more steps.
- **LoRA "doesn't fire" in ComfyUI/A1111** — weights are saved in diffusers PEFT format under `lora/`. Load them with diffusers' `pipe.load_lora_weights(...)`. A kohya/A1111-compatible converter is a nice-to-have not yet in this repo.

---

## Legacy scripts

The pre-refactor scripts (`resize_images.py`, `caption_images.py`) live under `scripts/legacy_*.py` for reference only. New work belongs in `src/image_trainer/`.
