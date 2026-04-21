# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Preprocessing utilities for a LoRA/fine-tuning image training pipeline. The two scripts prepare a dataset of photos of a person for training: `resize_images.py` normalizes images to 1024×1024, then `caption_images.py` generates a `.txt` caption file alongside each image using BLIP, with a trigger word prepended.

The actual training/inference tooling (ComfyUI, kohya_ss) lives in sibling directories that are explicitly git-ignored — this repo contains only the data prep scripts.

## Pipeline

The scripts are designed to be run in order on a fixed directory layout rooted at `~/Apps/image_trainer/training_data/`:

1. Drop source images into `training_data/raw/` (`.jpg`, `.jpeg`, `.png`, `.webp`).
2. `python resize_images.py` — scales the shortest side to 1024, center-crops to 1024×1024, and writes sequentially numbered PNGs (`0000.png`, `0001.png`, …) to `training_data/processed/`.
3. `python caption_images.py` — runs BLIP (`Salesforce/blip-image-captioning-large`) on each `.png` in `processed/` and writes a sibling `.txt` containing `"{TRIGGER_WORD}, {caption}"`. The current trigger word is `ohwx person` (hardcoded at the top of the file).

Paths, target size, and trigger word are all module-level constants — edit the source when changing datasets/subjects rather than adding CLI flags unless asked.

## Environment

- Python with `torch`, `transformers`, `Pillow`. No `requirements.txt` is checked in.
- `caption_images.py` requires a CUDA GPU: the model is loaded in `float16` and `.to("cuda")`. It will fail on CPU-only machines — do not silently fall back; surface the issue.
- The input/output paths use `os.path.expanduser("~/Apps/image_trainer/...")`, so the repo is expected to be cloned at that location (or the paths edited).

## Conventions

- Output PNGs are renamed to zero-padded indices (`{i:04d}.png`) so captions and images pair up positionally; don't change the naming scheme without also considering downstream training configs that may reference these filenames.
- No test suite, linter config, or build step exists. Don't invent one unless asked.
- Large artifacts (`*.safetensors`, `*.ckpt`, `*.pt`, `models/`, all `training_data/` subfolders, `ComfyUI/`, `kohya_ss/`) are git-ignored by design — never commit them back in.
