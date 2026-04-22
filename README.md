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

# Optional but recommended for NSFW / adult person LoRAs — enables the WD14
# booru-style tagger, which is much more candid than BLIP about body,
# anatomy, clothing, and pose. See "Captioning" below.
pip install -e ".[wd14]"          # or: pip install onnxruntime-gpu huggingface_hub
```

Python 3.10+, a CUDA build of PyTorch, and an SDXL `.safetensors` base checkpoint you have downloaded locally.

## Base checkpoint

You supply this. Download any SDXL-family `.safetensors` you want to build on top of and point the app at the file. The app never downloads checkpoints for you.

Good starting points for realistic person LoRAs:

| Base | Strength | Notes |
| --- | --- | --- |
| **Juggernaut XL** | Photorealism, skin, lighting. | SFW-leaning variants and NSFW-tuned variants exist — pick deliberately. |
| **RealVisXL V4.0** | Very realistic portraits. | Clean SFW base; fine with moderate NSFW after a LoRA. |
| **CyberRealistic XL** | Photorealism with NSFW headroom. | Handles nudity without fighting the prompt. |
| **Pony Diffusion V6 XL** | NSFW anatomy / poses. | Different aesthetic; best paired with WD14 tags, not BLIP sentences. |

Once you've downloaded one, either pass `--base /path/to/file.safetensors` to `trainer init` or set it on the **Settings** tab. For best quality, **use the same base at training time and generation time** — LoRAs trained on base A generally look off when applied to base B.

---

## Quickstart — CLI

Each "project" is one LoRA-training run. Every project gets its own folder and its own `config.json`.

```bash
# 1. Create a project.
trainer init me --base ~/models/mySDXL.safetensors --trigger-word "ohwx person"

# 2. Import source images and resize to 1024x1024.
trainer prep me --source ~/Pictures/me_dataset

# 3. Auto-caption. Default is "both" = BLIP sentence + WD14 tags.
#    Override for a run with --mode blip | wd14 | both.
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

`trainer clean me` deletes the regeneratable `cache/` and `checkpoints/` after you've finalised a LoRA — can easily reclaim 20–100 GB. Add `--all` to also drop `raw/` and `processed/`. The trained `lora/`, generated `outputs/`, `config.json`, and `logs/` are always preserved.

## Quickstart — GUI

```bash
trainer gui
```

The wizard has a **project browser** at the top (create / switch / refresh) and six step tabs:

1. **Settings** — trigger word, base checkpoint, and the curated OOM / quality knobs: **resolution**, **LoRA rank**, **gradient accumulation**, **xformers on/off**, **text-encoder LoRA on/off** (see [Text-encoder LoRA](#text-encoder-lora) below), plus training length + checkpoint frequency + validation frequency.
2. **Import & Resize** — pick a source folder, run the copy + resize.
3. **Caption** — run the configured captioner (BLIP / WD14 / both).
4. **Review** — step through each processed image: include/exclude toggle, editable caption, quick-tag chips, per-image resolution/brightness/sharpness stats, and near-duplicate warnings. Keyboard: `←`/`→` navigate, `I` toggle include, `Ctrl+S` save. Training only uses `include=True` images.
5. **Train** — Start / Resume / Stop (graceful), live progress bar, optional journal note, plus shortcuts to open the `logs/` folder, the latest `training_<ts>.log`, the `logs/journal.txt` training diary, and the `validation/` folder so you can watch quality trend over the night.
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

## Captioning

Step 3 writes one `.txt` caption next to each `processed/NNNN.png`. Training reads those captions, so caption quality is a direct lever on LoRA quality.

Three modes, selected via `Project.captioner` (override per-run with `trainer caption <project> --mode ...`):

| Mode | Output format | Best for |
| --- | --- | --- |
| `blip` | English sentence from BLIP-large. | Fully-clothed subjects; SFW lifestyle / portrait LoRAs. |
| `wd14` | Danbooru-style tag list from WD14 (SmilingWolf). | Anime bases, NSFW subjects where anatomy/pose/clothing tags matter more than sentences. |
| `both` (default) | BLIP sentence **+** WD14 tags concatenated after the trigger word. | Realistic person LoRAs, SFW **or** NSFW — broadest coverage. |

Why WD14 for NSFW: BLIP was trained on a sanitised web corpus and tends to be vague or euphemistic about nudity, anatomy, poses, and adult body content. WD14 was trained on a booru corpus and produces candid tags like `1girl, long hair, topless, large breasts, spread legs, bedroom`. For a person LoRA trained on an adult dataset, those tags are what the model actually needs to learn.

After running step 3, **skim `processed/*.txt`** and:

- Remove anything that's *always true of your subject* (hair colour, eye colour, distinctive face shape). Those belong to the trigger word; captioning them forces the LoRA to fight itself.
- Keep anything that *varies across the dataset* (clothing, pose, lighting, setting).
- The **Review** tab (step 4) is the intended place to do this; you can edit any caption and tick/untick images to include. Training only trains on `include=True` rows.

### Quick-insert chips

`Project.prompt_chips` is a per-project list of one-click caption tokens shown on the Review tab. Defaults cover the common axes for a person LoRA: framing (`close-up`, `full body`), poses (`standing`, `sitting`), lighting (`natural light`, `studio lighting`), clothing (`casual clothing`, `dress`, `swimwear`), and explicit tags (`nude`, `topless`, `lingerie`, …). Edit the list on the Review tab or in `config.json` to add project-specific tags.

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

### Text-encoder LoRA

Off by default because it's the single biggest VRAM-and-quality lever in the pipeline. Flip the **Text-encoder LoRA** checkbox in the Train tab (or set `"train_text_encoder": true` in `config.json`) to enable it.

What changes when it's on:

- Both CLIP text encoders stay resident on GPU for the whole run; captions are re-encoded every step so the LoRA weight updates take effect. The cached-embedding path is bypassed.
- Only VAE latents are cached; the VAE is still offloaded (VAE doesn't train, so this is pure VRAM savings).
- A preflight step runs one forward+backward before the main loop and prints peak VRAM usage. If it's over 9.5 GB, you'll get a warning suggesting which knobs to drop before the real run wastes four hours.
- The exported LoRA now includes `text_encoder/` and `text_encoder_2/` subdirectories under `<project>/lora/` alongside `unet/`. `trainer generate` loads all three automatically; no extra flags.

Quality tradeoff: noticeably better subject identity (face, micro-features) and prompt-following on subject-adjacent concepts. VRAM tradeoff: roughly +2 GB peak on the 10 GB budget. If it OOMs, drop `resolution` to 768 first; that always fits.

TE-LoRA-specific tunables, all in `config.json` (no GUI — power-user knobs):

| Field | Default | Notes |
| --- | --- | --- |
| `te_lora_rank` | 8 | TEs need less capacity than the UNet; going higher rarely helps. |
| `te_lora_alpha` | 8 | Keep equal to `te_lora_rank`. |
| `te_learning_rate` | 5e-5 | ~½ the UNet LR; TEs are more sensitive and can destabilize at 1e-4. |
| `te_gradient_checkpointing` | true | Required to fit on 10 GB. Disable on 16 GB+ for ~10% speedup. |

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

1. If Text-encoder LoRA is on, that's the likely culprit — drop `resolution` to 768, or turn TE LoRA off to confirm.
2. Lower `resolution` to **768** in Settings, retrain from scratch.
3. Bump `gradient_accumulation_steps` to 2 or 4 (trades wall time for memory).
4. Drop `lora_rank` from 32 to 16 (marginal quality hit).
5. Disable `xformers` if a version mismatch is the culprit — the loop falls back to PyTorch sdpa.
6. Make sure nothing else is using the GPU (browser hardware accel, another ML process).
7. As a last resort, switch `base_model_path` to an SD 1.5 checkpoint — the pipeline code is the same, but your LoRA will be SD1.5-compatible, not SDXL-compatible.

---

## Troubleshooting

- **"caption_dataset requires a CUDA GPU"** — BLIP captioning loads in fp16 on CUDA with no CPU fallback on purpose. Run on a CUDA machine, or set `captioner = "wd14"` to skip BLIP entirely (WD14 will run on CPU via `onnxruntime` if no CUDA EP is available).
- **"WD14 captioning needs onnxruntime"** — install the WD14 extra: `pip install -e ".[wd14]"` (or `pip install onnxruntime-gpu huggingface_hub`). Alternatively set `captioner = "blip"` in `config.json`.
- **Running low on disk** — `trainer clean <project>` drops `cache/` (pre-computed latents + text embeddings — regenerated on next train) and `checkpoints/` (intermediate `step_N/` dirs — only needed for `--resume`). The final `lora/` is always preserved.
- **Base checkpoint won't load** — the loader tries `from_single_file` for `.safetensors` and `from_pretrained` for directories. If you pass a `.ckpt`, convert it to `.safetensors` first (e.g. using `diffusers`' converter).
- **Training hangs at "Caching latents + text embeddings"** — that step runs once per image on first training run. After that it's nearly instant. Delete `cache/` if you've changed `resolution` or captions.
- **Validation images are black or noise** — usually a broken VAE in fp16 on some cards; the generation loop uses the full pipeline, so retry inference (step 5) with `guidance 5–6` and more steps.
- **LoRA "doesn't fire" in ComfyUI/A1111** — weights are saved in diffusers PEFT format under `lora/`. Load them with diffusers' `pipe.load_lora_weights(...)`. A kohya/A1111-compatible converter is a nice-to-have not yet in this repo.

---

## Legacy scripts

The pre-refactor scripts (`resize_images.py`, `caption_images.py`) live under `scripts/legacy_*.py` for reference only. New work belongs in `src/image_trainer/`.
