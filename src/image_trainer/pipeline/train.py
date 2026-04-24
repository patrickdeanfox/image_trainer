"""SDXL LoRA training loop.

Designed for a 10 GB VRAM budget with uncompromised final-image quality as the
goal, and overnight-friendly behavior (resumable, graceful Ctrl+C, logged to
disk, validation previews so you can see quality trend in the morning).

Quality features:
- PEFT LoRA, rank 32 by default, on UNet.
- min-SNR-gamma loss weighting (Ephraim et al. 2023).
- Offset noise.
- Cosine LR schedule with warmup.
- Shuffled epoch iteration (not just step % n).

Memory features (all default-on for 10 GB):
- Pre-computed VAE latents and text embeddings cached to disk; VAE + text
  encoders are then unloaded from GPU entirely for the training loop.
- fp16 mixed precision via accelerate.
- UNet gradient checkpointing.
- 8-bit AdamW via bitsandbytes (soft-fallback to torch.optim.AdamW).
- xformers memory-efficient attention when available.

Overnight features:
- `--resume` picks up from the highest `checkpoints/step_N/` directory; the
  RNG is also advanced by the completed-epoch count so shuffling is consistent.
- SIGINT / SIGTERM are caught: the current step finishes, a checkpoint is
  written, then the process exits cleanly.
- Full stdout is tee'd to `logs/training_<timestamp>.log` (one file per run).
- `project.validation_steps` > 0 writes PNG previews to `logs/validation/`.
- The cache layer has a marker file to catch silent invalidation when
  `resolution` or `base_model_path` changes.
"""

from __future__ import annotations

import json
import random
import shutil
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

from ..config import Project

ProgressCb = Callable[[int, int], None]

_CACHE_MARKER = "cache_marker.json"


def append_journal(project: Project, note: str, extra: Optional[dict] = None) -> None:
    """Append a single-line entry to `logs/journal.txt` so users can keep a
    record of what they tried and what worked."""
    project.ensure_dirs()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    bits = [
        f"[{ts}]",
        f"rank={project.lora_rank}",
        f"res={project.resolution}",
        f"lr={project.learning_rate}",
        f"steps={project.max_train_steps}",
    ]
    if extra:
        for k, v in extra.items():
            bits.append(f"{k}={v}")
    if note:
        bits.append(f"note={note!r}")
    line = " ".join(bits) + "\n"
    with (project.logs_dir / "journal.txt").open("a") as f:
        f.write(line)


# ---------- tee logging ----------

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


@contextmanager
def _tee_stdout(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", buffering=1)  # fresh file per run (filename is timestamped)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, f)
    sys.stderr = _Tee(orig_stderr, f)
    try:
        yield
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        f.close()


# ---------- base model + cache ----------

def _load_sdxl_pipeline(base_model_path: Path):
    import torch
    from diffusers import StableDiffusionXLPipeline

    if base_model_path.suffix == ".safetensors" and base_model_path.is_file():
        return StableDiffusionXLPipeline.from_single_file(
            str(base_model_path), torch_dtype=torch.float16
        )
    return StableDiffusionXLPipeline.from_pretrained(
        str(base_model_path), torch_dtype=torch.float16
    )


def _validate_or_reset_cache(project: Project) -> None:
    """Cached latents + embeds are only valid for the (resolution, base_model)
    combo they were produced from. If either changed, nuke the cache so we
    regenerate rather than silently train on stale tensors."""
    project.cache_dir.mkdir(parents=True, exist_ok=True)
    marker_path = project.cache_dir / _CACHE_MARKER
    current = {
        "resolution": project.resolution,
        "base_model_path": str(project.base_model_path) if project.base_model_path else None,
    }
    if marker_path.exists():
        try:
            prev = json.loads(marker_path.read_text())
        except Exception as e:
            print(
                f"Warning: {marker_path} is not valid JSON ({e}); "
                f"treating the cache as stale and rebuilding.",
                flush=True,
            )
            prev = None
        if prev != current:
            print(
                f"Cache marker mismatch (resolution or base model changed); "
                f"clearing {project.cache_dir}",
                flush=True,
            )
            for sub in ("latents", "embeds"):
                d = project.cache_dir / sub
                if d.exists():
                    shutil.rmtree(d)
    marker_path.write_text(json.dumps(current, indent=2))


def _select_training_pngs(
    project: Project, limit: Optional[int] = None
) -> list[Path]:
    """List processed PNGs, filtered by the pre-training review.

    ``limit`` (optional positive int) further caps the result to the first
    ``N`` PNGs after the review filter — used by the GUI's "Images per run"
    picker to make short runs possible. The cap is applied AFTER the review
    filter so the user's include/exclude decisions are always respected.

    Raises:
        RuntimeError: if the project has no processed images or every image
            was excluded in the Review tab.
    """
    from . import review as review_mod

    all_pngs = sorted(project.processed_dir.glob("*.png"))
    if not all_pngs:
        raise RuntimeError(
            f"No .png files in {project.processed_dir}. Run prep + caption first."
        )
    review = review_mod.load(project)
    included = set(review.stems_for_training())
    pngs = [p for p in all_pngs if p.stem in included]
    if not pngs:
        raise RuntimeError(
            f"Review excluded every image in {project.processed_dir}. "
            f"Open the Review tab and mark some as included."
        )
    excluded = len(all_pngs) - len(pngs)
    full_count = len(pngs)
    if limit is not None and limit > 0 and limit < full_count:
        pngs = pngs[:limit]
        print(
            f"Review: training on {len(pngs)}/{full_count} included images "
            f"(limit-images={limit}; {excluded} excluded total).",
            flush=True,
        )
    elif excluded > 0:
        print(
            f"Review: training on {len(pngs)}/{len(all_pngs)} images "
            f"({excluded} excluded).",
            flush=True,
        )
    return pngs


def _cache_vae_latents(project: Project, pipe, pngs: list[Path], device: str) -> dict[str, Path]:
    """Encode each training image through the VAE once, cache to disk, then
    move the VAE to CPU. Used in both the cached-embeddings path and the
    live-encoders path — VAE never trains, so latents are always cacheable.

    Returns a mapping ``{stem: latent_path}``.
    """
    import torch
    from PIL import Image
    from torchvision import transforms

    (project.cache_dir / "latents").mkdir(parents=True, exist_ok=True)

    image_tf = transforms.Compose(
        [
            transforms.Resize(
                project.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(project.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    vae = pipe.vae.to(device, dtype=torch.float32)
    latent_paths: dict[str, Path] = {}
    total = len(pngs)
    for idx, png in enumerate(pngs, start=1):
        stem = png.stem
        latent_path = project.cache_dir / "latents" / f"{stem}.pt"
        if not latent_path.exists():
            img = Image.open(png).convert("RGB")
            tensor = image_tf(img).unsqueeze(0).to(device, dtype=torch.float32)
            with torch.no_grad():
                latent = vae.encode(tensor).latent_dist.sample() * vae.config.scaling_factor
            torch.save(latent.squeeze(0).cpu(), latent_path)
            print(f"caching latents {idx}/{total}: {png.name}", flush=True)
        latent_paths[stem] = latent_path

    pipe.vae.to("cpu")
    del vae
    torch.cuda.empty_cache()
    return latent_paths


def _cache_text_embeddings(project: Project, pipe, pngs: list[Path], device: str) -> dict[str, Path]:
    """Encode each caption through both text encoders once, cache to disk,
    then move the TEs to CPU. Only called when ``train_text_encoder=False`` —
    with TE LoRA on, captions have to be re-encoded every step against the
    current LoRA weights.

    Returns a mapping ``{stem: embed_path}``.
    """
    import torch

    (project.cache_dir / "embeds").mkdir(parents=True, exist_ok=True)

    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    te1 = pipe.text_encoder.to(device)
    te2 = pipe.text_encoder_2.to(device)

    embed_paths: dict[str, Path] = {}
    total = len(pngs)
    for idx, png in enumerate(pngs, start=1):
        stem = png.stem
        embed_path = project.cache_dir / "embeds" / f"{stem}.pt"
        if not embed_path.exists():
            txt_path = png.with_suffix(".txt")
            caption = txt_path.read_text().strip() if txt_path.exists() else project.trigger_word

            tok1 = tokenizer(
                caption,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            tok2 = tokenizer_2(
                caption,
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out1 = te1(tok1.input_ids, output_hidden_states=True)
                out2 = te2(tok2.input_ids, output_hidden_states=True)
                hidden1 = out1.hidden_states[-2]
                hidden2 = out2.hidden_states[-2]
                prompt_embeds = torch.cat([hidden1, hidden2], dim=-1)
                pooled = out2[0]
            torch.save(
                {
                    "prompt_embeds": prompt_embeds.squeeze(0).cpu(),
                    "pooled": pooled.squeeze(0).cpu(),
                },
                embed_path,
            )
            print(f"caching embeds {idx}/{total}: {png.name}", flush=True)
        embed_paths[stem] = embed_path

    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    del te1, te2
    torch.cuda.empty_cache()
    return embed_paths


def _cache_embeddings_and_latents(
    project: Project, pipe, device: str = "cuda",
    limit_images: Optional[int] = None,
) -> list[dict]:
    """Cached-embeddings path: latents *and* text embeds pre-computed once,
    then both VAE and TEs offloaded to CPU. Used when ``train_text_encoder``
    is False (the default); returns a list of
    ``{"latent": Path, "embed": Path, "stem": str}`` ready for the training
    loop's per-step ``torch.load`` calls.
    """
    _validate_or_reset_cache(project)
    pngs = _select_training_pngs(project, limit=limit_images)
    latent_paths = _cache_vae_latents(project, pipe, pngs, device)
    embed_paths = _cache_text_embeddings(project, pipe, pngs, device)
    return [
        {"latent": latent_paths[p.stem], "embed": embed_paths[p.stem], "stem": p.stem}
        for p in pngs
    ]


def _build_live_dataset(
    project: Project, pipe, device: str,
    limit_images: Optional[int] = None,
) -> list[dict]:
    """Live-encoders path: cache only VAE latents (VAE still doesn't train),
    but keep both text encoders resident on GPU and pre-tokenize every
    caption. The training loop encodes captions fresh every step so TE LoRA
    weight updates take effect; tokenization itself is cheap and only needs
    to happen once.

    Returns a list of dicts with keys:
        ``"latent"`` -> Path to cached latent .pt
        ``"tokens1"`` -> torch.LongTensor [max_length] for text_encoder
        ``"tokens2"`` -> torch.LongTensor [max_length] for text_encoder_2
        ``"caption"`` -> str (kept for debug; not used per-step)
        ``"stem"``    -> image stem
    """
    _validate_or_reset_cache(project)
    pngs = _select_training_pngs(project, limit=limit_images)
    latent_paths = _cache_vae_latents(project, pipe, pngs, device)

    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    entries: list[dict] = []
    for png in pngs:
        txt_path = png.with_suffix(".txt")
        caption = txt_path.read_text().strip() if txt_path.exists() else project.trigger_word
        tok1 = tokenizer(
            caption,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tok2 = tokenizer_2(
            caption,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        entries.append(
            {
                "latent": latent_paths[png.stem],
                "tokens1": tok1.input_ids.squeeze(0),
                "tokens2": tok2.input_ids.squeeze(0),
                "caption": caption,
                "stem": png.stem,
            }
        )
    return entries


def _encode_live(te1, te2, tokens1, tokens2):
    """Run both text encoders on already-tokenized captions to produce the
    tensors the SDXL UNet forward expects.

    ``tokens1`` / ``tokens2`` are [B, max_length] LongTensors on the same
    device as the encoders. Returns ``(prompt_embeds, pooled)``:
        ``prompt_embeds`` -> [B, 77, 2048]  (concat of penultimate hidden
                                             states, features last)
        ``pooled``        -> [B, 1280]      (text_encoder_2's pooled output)

    Unlike the cache path's `_cache_text_embeddings`, this function runs
    under autograd: gradients flow back into the wrapped TE LoRA modules.
    """
    import torch

    out1 = te1(tokens1, output_hidden_states=True)
    out2 = te2(tokens2, output_hidden_states=True)
    hidden1 = out1.hidden_states[-2]
    hidden2 = out2.hidden_states[-2]
    prompt_embeds = torch.cat([hidden1, hidden2], dim=-1)
    pooled = out2[0]
    return prompt_embeds, pooled


def _wrap_text_encoders_with_lora(pipe, project: Project):
    """Install PEFT LoRA adapters onto both text encoders in-place on ``pipe``.

    Uses the HF-standard CLIP LoRA target set. Separate (lower) rank from
    UNet because TEs need less capacity and tend to overfit faster.
    Optionally enables gradient checkpointing on each encoder to fit 10 GB.

    Mutates ``pipe.text_encoder`` and ``pipe.text_encoder_2`` so validation
    inference (which goes through the diffusers pipeline) sees the wrapped
    encoders automatically. Returns ``(te1, te2)`` — the same objects.
    """
    from peft import LoraConfig, get_peft_model

    te_config = LoraConfig(
        r=project.te_lora_rank,
        lora_alpha=project.te_lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    )
    te1 = get_peft_model(pipe.text_encoder, te_config)
    te2 = get_peft_model(pipe.text_encoder_2, te_config)
    pipe.text_encoder = te1
    pipe.text_encoder_2 = te2

    if project.te_gradient_checkpointing:
        for name, te in (("text_encoder", te1), ("text_encoder_2", te2)):
            base = getattr(te, "base_model", te)
            target = getattr(base, "model", base)  # PEFT wraps .base_model.model
            enable = getattr(target, "gradient_checkpointing_enable", None)
            if callable(enable):
                try:
                    enable()
                except Exception as e:
                    print(
                        f"  {name}: gradient_checkpointing_enable failed "
                        f"({e}); continuing without.",
                        flush=True,
                    )
            else:
                print(
                    f"  {name}: gradient_checkpointing_enable not available "
                    f"on this transformers version; skipping.",
                    flush=True,
                )

    print("TE LoRA trainable parameters:", flush=True)
    te1.print_trainable_parameters()
    te2.print_trainable_parameters()
    return te1, te2


def _find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    # Only consider directories whose name parses as "step_<int>". This skips
    # hand-renamed dirs like "step_final" instead of crashing with ValueError.
    valid: list[tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        if not (p.is_dir() and p.name.startswith("step_")):
            continue
        try:
            step = int(p.name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        valid.append((step, p))
    if not valid:
        return None
    return max(valid, key=lambda pair: pair[0])[1]


# ---------- loss helpers ----------

def _compute_snr(timesteps, noise_scheduler):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus = (1.0 - alphas_cumprod[timesteps]) ** 0.5
    return (sqrt_alphas_cumprod / sqrt_one_minus) ** 2


def _min_snr_loss_weights(timesteps, noise_scheduler, gamma: float):
    import torch

    snr = _compute_snr(timesteps, noise_scheduler)
    return torch.stack([snr, gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr


# ---------- validation previews ----------

def _run_validation(project: Project, pipe, trained_unet, step: int, device: str) -> Optional[Path]:
    """Generate one preview image with the current LoRA weights.

    The key subtleties (both caught by code review):
    - `get_peft_model(unet, ...)` returns a *new* wrapper object. `pipe.unet`
      still points at the un-wrapped UNet, so calling `pipe(...)` directly
      would *not* reflect training progress. We temporarily swap `pipe.unet`
      with the unwrapped-from-accelerator trained model.
    - The VAE is moved to CPU in both training paths. For validation
      inference we move it back to GPU, then back to CPU afterwards to
      preserve the training-time memory budget.
    - The text encoders stay on GPU when ``train_text_encoder`` is on (they
      must be resident for the per-step forwards), so we skip the VAE-style
      ping-pong for them in that mode. When TE LoRA is off, they were moved
      to CPU during cache building and we bring them back for validation
      just like the VAE.
    """
    import torch

    te_on_gpu = project.train_text_encoder
    original_unet = pipe.unet
    try:
        pipe.unet = trained_unet
        pipe.vae.to(device)
        if not te_on_gpu:
            pipe.text_encoder.to(device)
            pipe.text_encoder_2.to(device)

        prompt = project.effective_validation_prompt()
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                num_inference_steps=25,
                guidance_scale=6.5,
                generator=torch.Generator(device=device).manual_seed(project.seed),
            ).images[0]
        out = project.validation_dir / f"step_{step:06d}.png"
        image.save(out)
        print(f"  validation preview -> {out}", flush=True)
        return out
    except Exception:
        print("  validation failed (non-fatal):", flush=True)
        print(traceback.format_exc(), flush=True)
        return None
    finally:
        # Return to the training-time memory layout.
        pipe.unet = original_unet
        pipe.vae.to("cpu")
        if not te_on_gpu:
            pipe.text_encoder.to("cpu")
            pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()


# ---------- TE LoRA preflight ----------

def _preflight_te_lora(
    project: Project,
    unet,
    te1,
    te2,
    noise_scheduler,
    base_time_ids,
    sample_entry: dict,
    device,
) -> None:
    """One forward+backward on a real entry to measure peak VRAM before the
    training loop starts. Prints the peak, warns if >9.5 GB, does not
    auto-change settings — the user asked for explicit knobs, not surprise
    fallbacks.
    """
    import torch

    torch.cuda.reset_peak_memory_stats(device)
    unet.train()
    te1.train()
    te2.train()

    latent = torch.load(sample_entry["latent"]).to(device, dtype=torch.float32).unsqueeze(0)
    tokens1 = sample_entry["tokens1"].to(device).unsqueeze(0)
    tokens2 = sample_entry["tokens2"].to(device).unsqueeze(0)
    prompt_embeds, pooled = _encode_live(te1, te2, tokens1, tokens2)
    bsz = latent.shape[0]
    time_ids = base_time_ids.unsqueeze(0).expand(bsz, -1)

    noise = torch.randn_like(latent)
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
    ).long()
    noisy = noise_scheduler.add_noise(latent, noise, timesteps)

    pred = unet(
        noisy,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
    ).sample
    loss = ((pred.float() - noise.float()) ** 2).mean()
    loss.backward()

    # Clear the grads so the real first step starts from a clean slate.
    for p in unet.parameters():
        if p.grad is not None:
            p.grad = None
    for p in te1.parameters():
        if p.grad is not None:
            p.grad = None
    for p in te2.parameters():
        if p.grad is not None:
            p.grad = None

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print(
        f"TE LoRA preflight: peak VRAM ~{peak_gb:.2f} GB of {total_gb:.2f} GB total.",
        flush=True,
    )
    if peak_gb > 9.5:
        print(
            "  WARNING: peak VRAM is close to / over 10 GB. If this run OOMs, "
            "try in config.json: drop `resolution` to 768, drop `lora_rank` to 16, "
            "or set `te_gradient_checkpointing` to true if it's off.",
            flush=True,
        )
    torch.cuda.empty_cache()


# ---------- main ----------

def train_lora(
    project: Project,
    resume: bool = False,
    max_steps_override: Optional[int] = None,
    progress_cb: Optional[ProgressCb] = None,
    note: str = "",
    limit_images: Optional[int] = None,
) -> Path:
    """Run LoRA training. Returns the path to the exported LoRA directory.

    ``limit_images`` (optional, positive int) caps how many included images
    this run trains on. The first ``N`` stems from the alphabetical
    included list are used; everything else is ignored for this run only.
    Does not affect the cache — entries for the N stems are reused if they
    exist, otherwise they get cached fresh.
    """
    import torch
    from accelerate import Accelerator
    from diffusers import DDPMScheduler
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig, get_peft_model

    if project.base_model_path is None:
        raise ValueError(
            "No base checkpoint configured. Set `base_model_path` in config.json, "
            "pass `trainer train <project> --base <sdxl.safetensors>`, or fill it "
            "in the GUI Settings tab."
        )
    if not project.base_model_path.exists():
        raise FileNotFoundError(
            f"base_model_path does not exist: {project.base_model_path}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("train_lora requires a CUDA GPU.")

    # ---- VRAM gate: TE LoRA needs ~12 GB; on smaller cards it OOMs at
    # the first ``unet.to(device)`` because both text encoders are already
    # resident. Catch this BEFORE we burn time loading the base checkpoint
    # (which on a cold cache means downloading several GB of HF assets).
    # Override with ``IMAGE_TRAINER_FORCE_TE_LORA=1`` if you want to live
    # dangerously (e.g. someone with 11 GB on a Linux card with no display
    # compositor stealing 1 GB).
    import os
    if project.train_text_encoder and not os.environ.get("IMAGE_TRAINER_FORCE_TE_LORA"):
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_gb < 12.0:
            raise RuntimeError(
                f"Text-encoder LoRA needs ~12 GB of VRAM and your card has "
                f"{total_gb:.1f} GB. The run would OOM at unet.to(device) "
                f"before training even starts.\n\n"
                f"Fix one of these in the GUI Settings tab:\n"
                f"  • Uncheck 'Text-encoder LoRA' (recommended for 10 GB cards)\n"
                f"  • Drop resolution to 768 (less help; TEs are still resident)\n"
                f"  • Drop LoRA rank to 16\n\n"
                f"Override (not recommended): set IMAGE_TRAINER_FORCE_TE_LORA=1 in your environment."
            )

    project.ensure_dirs()
    log_path = project.logs_dir / f"training_{int(time.time())}.log"
    append_journal(project, note, extra={"resume": int(bool(resume))})

    with _tee_stdout(log_path):
        print(f"=== training start, logging to {log_path} ===", flush=True)

        accelerator = Accelerator(
            mixed_precision=project.mixed_precision,
            gradient_accumulation_steps=project.gradient_accumulation_steps,
        )
        device = accelerator.device

        pipe = _load_sdxl_pipeline(project.base_model_path)
        if project.use_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"xformers unavailable ({e}); falling back to sdpa.", flush=True)

        # Dispatch on `train_text_encoder`. The UNet-only path caches both
        # latents AND text embeds once and offloads everything but the UNet;
        # the TE-LoRA path caches only latents and keeps both text encoders
        # resident (and trainable) on GPU.
        if project.train_text_encoder:
            print(
                "TE LoRA enabled: caching latents only, keeping text encoders on GPU...",
                flush=True,
            )
            entries = _build_live_dataset(
                project, pipe, device=str(device), limit_images=limit_images,
            )
        else:
            print("Caching latents + text embeddings (one-time per image)...", flush=True)
            entries = _cache_embeddings_and_latents(
                project, pipe, device=str(device), limit_images=limit_images,
            )
        n = len(entries)
        if n == 0:
            raise RuntimeError("No training entries after caching; aborting.")

        # ---- safe UNet upcast (10 GB VRAM friendly) ----
        # The naive ``unet.to(device, dtype=torch.float32)`` does device + dtype
        # in one step. PyTorch can't do that in-place when the dtype changes,
        # so it allocates the fp32 destination on GPU while the fp16 source is
        # still resident. On a 10 GB card with caching residue this trips OOM
        # before the first training step.
        #
        # Robust path: belt-and-braces offload the encoders we just used (the
        # offload calls inside ``_cache_*`` do this too, but PyTorch's caching
        # allocator can hold fragments), force a gc + empty_cache, then cast
        # the UNet to fp32 ON CPU first. The CPU dtype-conversion hits the
        # user's plentiful RAM, and the eventual move to GPU is a clean
        # single allocation.
        import gc
        if not project.train_text_encoder:
            # In the live-encoders path the TEs are intentionally resident on
            # GPU; only offload them when we're not training them.
            pipe.text_encoder.to("cpu")
            pipe.text_encoder_2.to("cpu")
        pipe.vae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        unet = pipe.unet
        unet.to("cpu", dtype=torch.float32)   # cast in CPU RAM, not on GPU
        gc.collect()
        torch.cuda.empty_cache()
        unet.to(device)                        # then a single clean GPU alloc
        unet.enable_gradient_checkpointing()

        lora_config = LoraConfig(
            r=project.lora_rank,
            lora_alpha=project.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

        # TE LoRA adapters (only on the live-encoders path). Move TEs to
        # fp32 on GPU first — same convention as the UNet; accelerator handles
        # mixed-precision autocast around the fp32 master weights.
        te1_lora = te2_lora = None
        if project.train_text_encoder:
            pipe.text_encoder.to(device, dtype=torch.float32)
            pipe.text_encoder_2.to(device, dtype=torch.float32)
            te1_lora, te2_lora = _wrap_text_encoders_with_lora(pipe, project)

        # Two param groups so UNet LoRA and TE LoRA can have different LRs.
        # TEs are more sensitive, so `te_learning_rate` defaults to ~½ of the
        # UNet LR.
        param_groups = [
            {
                "params": [p for p in unet.parameters() if p.requires_grad],
                "lr": project.learning_rate,
            }
        ]
        if te1_lora is not None:
            param_groups.append(
                {
                    "params": [p for p in te1_lora.parameters() if p.requires_grad]
                    + [p for p in te2_lora.parameters() if p.requires_grad],
                    "lr": project.te_learning_rate,
                }
            )

        if project.use_8bit_optim:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(param_groups, lr=project.learning_rate)
            except ImportError:
                print("bitsandbytes missing; using torch.optim.AdamW.", flush=True)
                optimizer = torch.optim.AdamW(param_groups, lr=project.learning_rate)
        else:
            optimizer = torch.optim.AdamW(param_groups, lr=project.learning_rate)

        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        max_steps = max_steps_override or project.max_train_steps

        lr_scheduler = get_scheduler(
            project.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=project.lr_warmup_steps,
            num_training_steps=max_steps,
        )

        # Pass TEs through accelerator.prepare too so mixed-precision wrapping
        # and save_state/load_state cover them automatically.
        if te1_lora is not None:
            unet, te1_lora, te2_lora, optimizer, lr_scheduler = accelerator.prepare(
                unet, te1_lora, te2_lora, optimizer, lr_scheduler
            )
            # prepare() returns new wrapper objects; keep the pipe pointers in
            # sync so validation inference hits the prepared modules.
            pipe.text_encoder = te1_lora
            pipe.text_encoder_2 = te2_lora
        else:
            unet, optimizer, lr_scheduler = accelerator.prepare(
                unet, optimizer, lr_scheduler
            )

        # Resume
        start_step = 0
        if resume:
            latest = _find_latest_checkpoint(project.checkpoints_dir)
            if latest is not None:
                accelerator.load_state(str(latest))
                start_step = int(latest.name.split("_")[1])
                # accelerate *does* restore scheduler state in modern versions, but
                # be defensive: if last_epoch drifts from start_step, force-align.
                try:
                    if getattr(lr_scheduler, "last_epoch", None) != start_step:
                        lr_scheduler.last_epoch = start_step
                except Exception:
                    pass
                print(f"Resumed from {latest} (step {start_step}).", flush=True)
            else:
                print("No checkpoint found; starting fresh.", flush=True)

        # Fixed time_ids because every sample is pre-cropped to `resolution`.
        # Expanded to current batch size inside the loop.
        res = project.resolution
        base_time_ids = torch.tensor(
            [res, res, 0, 0, res, res], device=device, dtype=torch.float32
        )

        # Graceful shutdown on SIGINT/SIGTERM.
        stop_requested = {"flag": False}

        def _handle_signal(signum, frame):
            print(
                f"\nCaught signal {signum}; will checkpoint after current step and exit.",
                flush=True,
            )
            stop_requested["flag"] = True

        # Save originals so we can restore them on normal return. Exception
        # paths skip the restore, but the handler only sets a flag, so a
        # lingering handler is benign. In the CLI case the process exits
        # right after train_lora() anyway; this matters for nested callers
        # (tests, batch runners) that invoke train_lora more than once.
        _orig_sigint = signal.signal(signal.SIGINT, _handle_signal)
        try:
            _orig_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
        except Exception:
            _orig_sigterm = None  # Windows doesn't support SIGTERM the same way.

        # Shuffled-epoch iteration. On resume, fast-forward the RNG by the number
        # of completed epochs so the next epoch's shuffle matches what a fresh
        # run would have produced at this step.
        rng = random.Random(project.seed)
        order: list[int] = list(range(n))
        completed_epochs = start_step // n
        for _ in range(completed_epochs + 1):
            rng.shuffle(order)
        cursor = start_step % n

        def _save_checkpoint(step: int) -> Path:
            ckpt_path = project.checkpoints_dir / f"step_{step}"
            accelerator.save_state(str(ckpt_path))
            print(f"  saved checkpoint: {ckpt_path}", flush=True)
            return ckpt_path

        unet.train()
        if te1_lora is not None:
            te1_lora.train()
            te2_lora.train()

        # Preflight VRAM check when TE LoRA is on: do one forward+backward on
        # a synthetic batch, report peak memory, and warn loudly if we're
        # over budget. Cheap insurance against a 4-hour run that OOMs at
        # step 1.
        if te1_lora is not None and start_step == 0:
            try:
                _preflight_te_lora(
                    project, unet, te1_lora, te2_lora, noise_scheduler,
                    base_time_ids, entries[0], device,
                )
            except Exception as e:
                print(
                    f"TE LoRA preflight check failed ({e}); continuing anyway.",
                    flush=True,
                )

        step = start_step
        t0 = time.time()
        while step < max_steps:
            entry = entries[order[cursor]]
            cursor += 1
            if cursor >= n:
                rng.shuffle(order)
                cursor = 0

            latent = torch.load(entry["latent"]).to(device, dtype=torch.float32).unsqueeze(0)
            if te1_lora is not None:
                # Live-encoders path: re-encode captions every step so TE
                # LoRA weight updates take effect. Tokens were cached at
                # dataset-build time (cheap); only the TE forward is hot.
                tokens1 = entry["tokens1"].to(device).unsqueeze(0)
                tokens2 = entry["tokens2"].to(device).unsqueeze(0)
                prompt_embeds, pooled = _encode_live(
                    te1_lora, te2_lora, tokens1, tokens2
                )
            else:
                embed_blob = torch.load(entry["embed"])
                prompt_embeds = embed_blob["prompt_embeds"].to(
                    device, dtype=torch.float32
                ).unsqueeze(0)
                pooled = embed_blob["pooled"].to(
                    device, dtype=torch.float32
                ).unsqueeze(0)
                # Drop the dict so its CPU-resident source tensors can be
                # reclaimed immediately; on a 10 GB budget every bit of
                # fragmentation headroom helps over long runs.
                del embed_blob
            bsz = latent.shape[0]
            time_ids = base_time_ids.unsqueeze(0).expand(bsz, -1)

            noise = torch.randn_like(latent)
            if project.offset_noise > 0:
                noise = noise + project.offset_noise * torch.randn(
                    (bsz, latent.shape[1], 1, 1), device=device
                )

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latent, noise, timesteps)

            accum_targets = [unet]
            if te1_lora is not None:
                accum_targets.extend([te1_lora, te2_lora])
            with accelerator.accumulate(*accum_targets):
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
                ).sample

                if project.min_snr_gamma > 0:
                    weights = _min_snr_loss_weights(
                        timesteps, noise_scheduler, project.min_snr_gamma
                    )
                    per_sample = (
                        (model_pred.float() - noise.float()) ** 2
                    ).mean(dim=[1, 2, 3])
                    loss = (per_sample * weights).mean()
                else:
                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % 10 == 0 or step == max_steps:
                elapsed = time.time() - t0
                sps = (step - start_step) / max(elapsed, 1e-6)
                eta_s = (max_steps - step) / max(sps, 1e-6)
                print(
                    f"step {step}/{max_steps} loss={loss.item():.4f} "
                    f"lr={lr_scheduler.get_last_lr()[0]:.2e} "
                    f"sps={sps:.2f} eta={eta_s/60:.1f}min",
                    flush=True,
                )
            if progress_cb is not None:
                progress_cb(step, max_steps)

            do_ckpt = (
                step % project.checkpointing_steps == 0
                or step == max_steps
                or stop_requested["flag"]
            )
            if do_ckpt:
                _save_checkpoint(step)

            do_val = (
                project.validation_steps > 0
                and (step % project.validation_steps == 0 or step == max_steps)
            )
            if do_val:
                _run_validation(project, pipe, unet, step, str(device))

            if stop_requested["flag"]:
                print("Graceful stop after checkpoint. Use --resume to continue.", flush=True)
                break

        # Export LoRA weights in the diffusers multi-component layout:
        #     <lora_dir>/unet/
        #     <lora_dir>/text_encoder/        (only if TE LoRA was trained)
        #     <lora_dir>/text_encoder_2/
        # `pipe.load_lora_weights(lora_dir)` in generate.py dispatches
        # automatically across whichever subdirectories exist, so no
        # generate-side code change is needed.
        if project.lora_dir.exists():
            shutil.rmtree(project.lora_dir)
        project.lora_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(str(project.lora_dir / "unet"))
        print(f"UNet LoRA exported to {project.lora_dir / 'unet'}", flush=True)

        if te1_lora is not None:
            unwrapped_te1 = accelerator.unwrap_model(te1_lora)
            unwrapped_te2 = accelerator.unwrap_model(te2_lora)
            unwrapped_te1.save_pretrained(str(project.lora_dir / "text_encoder"))
            unwrapped_te2.save_pretrained(str(project.lora_dir / "text_encoder_2"))
            print(
                f"TE LoRA exported to {project.lora_dir / 'text_encoder'} "
                f"and {project.lora_dir / 'text_encoder_2'}",
                flush=True,
            )

        # Restore the signal handlers we installed above, so nested callers
        # (tests, batch runners) don't inherit our shutdown trap.
        signal.signal(signal.SIGINT, _orig_sigint)
        if _orig_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, _orig_sigterm)
            except Exception:
                pass

        return project.lora_dir
