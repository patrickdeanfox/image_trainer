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
        except Exception:
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


def _cache_embeddings_and_latents(project: Project, pipe, device: str = "cuda") -> list[dict]:
    import torch
    from PIL import Image
    from torchvision import transforms

    _validate_or_reset_cache(project)
    (project.cache_dir / "latents").mkdir(parents=True, exist_ok=True)
    (project.cache_dir / "embeds").mkdir(parents=True, exist_ok=True)

    pngs = sorted(project.processed_dir.glob("*.png"))
    if not pngs:
        raise RuntimeError(
            f"No .png files in {project.processed_dir}. Run prep + caption first."
        )

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
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    te1 = pipe.text_encoder.to(device)
    te2 = pipe.text_encoder_2.to(device)

    entries: list[dict] = []
    total = len(pngs)
    for idx, png in enumerate(pngs, start=1):
        stem = png.stem
        latent_path = project.cache_dir / "latents" / f"{stem}.pt"
        embed_path = project.cache_dir / "embeds" / f"{stem}.pt"
        did_work = False

        if not latent_path.exists():
            img = Image.open(png).convert("RGB")
            tensor = image_tf(img).unsqueeze(0).to(device, dtype=torch.float32)
            with torch.no_grad():
                latent = vae.encode(tensor).latent_dist.sample() * vae.config.scaling_factor
            torch.save(latent.squeeze(0).cpu(), latent_path)
            did_work = True

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
            did_work = True

        entries.append({"latent": latent_path, "embed": embed_path, "stem": stem})
        if did_work:
            print(f"caching {idx}/{total}: {png.name}", flush=True)

    # Free VAE + TEs from GPU; the training loop doesn't need them.
    pipe.vae.to("cpu")
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    del vae, te1, te2
    torch.cuda.empty_cache()
    return entries


def _find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    candidates = [p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("step_")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.name.split("_")[1]))


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
    - VAE + text encoders were moved to CPU in `_cache_embeddings_and_latents`.
      For validation inference we need them on GPU. We move them back for the
      duration of this call and return them to CPU afterward so the training
      loop's memory budget is preserved.
    """
    import torch

    original_unet = pipe.unet
    try:
        pipe.unet = trained_unet
        pipe.vae.to(device)
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
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()


# ---------- main ----------

def train_lora(
    project: Project,
    resume: bool = False,
    max_steps_override: Optional[int] = None,
    progress_cb: Optional[ProgressCb] = None,
) -> Path:
    """Run LoRA training. Returns the path to the exported LoRA directory."""
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
    if project.train_text_encoder:
        # Check early so we never allocate text-encoder LoRA weights on GPU.
        raise NotImplementedError(
            "train_text_encoder=True is not wired up: it requires re-encoding "
            "captions every step, which is incompatible with the cached-embeddings "
            "strategy used in this training loop. Leave it off for now."
        )

    project.ensure_dirs()
    log_path = project.logs_dir / f"training_{int(time.time())}.log"

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

        print("Caching latents + text embeddings (one-time per image)...", flush=True)
        entries = _cache_embeddings_and_latents(project, pipe, device=str(device))
        n = len(entries)
        if n == 0:
            raise RuntimeError("No training entries after caching; aborting.")

        unet = pipe.unet
        unet.to(device, dtype=torch.float32)
        unet.enable_gradient_checkpointing()

        lora_config = LoraConfig(
            r=project.lora_rank,
            lora_alpha=project.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

        trainable_params = [p for p in unet.parameters() if p.requires_grad]

        if project.use_8bit_optim:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(trainable_params, lr=project.learning_rate)
            except ImportError:
                print("bitsandbytes missing; using torch.optim.AdamW.", flush=True)
                optimizer = torch.optim.AdamW(trainable_params, lr=project.learning_rate)
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=project.learning_rate)

        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        max_steps = max_steps_override or project.max_train_steps

        lr_scheduler = get_scheduler(
            project.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=project.lr_warmup_steps,
            num_training_steps=max_steps,
        )

        unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)

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

        signal.signal(signal.SIGINT, _handle_signal)
        try:
            signal.signal(signal.SIGTERM, _handle_signal)
        except Exception:
            pass  # Windows doesn't support SIGTERM the same way.

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
        step = start_step
        t0 = time.time()
        while step < max_steps:
            entry = entries[order[cursor]]
            cursor += 1
            if cursor >= n:
                rng.shuffle(order)
                cursor = 0

            latent = torch.load(entry["latent"]).to(device, dtype=torch.float32).unsqueeze(0)
            embed_blob = torch.load(entry["embed"])
            prompt_embeds = embed_blob["prompt_embeds"].to(device, dtype=torch.float32).unsqueeze(0)
            pooled = embed_blob["pooled"].to(device, dtype=torch.float32).unsqueeze(0)
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

            with accelerator.accumulate(unet):
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

        # Export LoRA weights (portable diffusers PEFT format).
        if project.lora_dir.exists():
            shutil.rmtree(project.lora_dir)
        project.lora_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(unet)
        unwrapped.save_pretrained(str(project.lora_dir))
        print(f"LoRA exported to {project.lora_dir}", flush=True)
        return project.lora_dir
