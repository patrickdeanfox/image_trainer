"""Inference step: load base SDXL, optionally stack LoRAs, generate images.

Step 6 of the pipeline. Uses :meth:`StableDiffusionXLPipeline.enable_model_cpu_offload`
to sequence sub-module loads across GPU and CPU so the whole pipeline fits
on a 10 GB card. Outputs are grouped under ``outputs/<timestamp>/`` so
separate generate calls don't overwrite each other.

LoRA stacking model:

- ``use_trained_lora=True`` loads the project's own trained LoRA from
  ``project/lora/`` (the PEFT-format export written by training).
- ``extra_loras`` is a list of ``(path, weight)`` for community LoRAs from
  e.g. civitai. Each is loaded as its own adapter and combined with the
  trained one. Use this to mix style packs ("anime", "photo realism", a
  specific lighting LoRA) on top of your likeness LoRA.

Adapters are combined via ``set_adapters`` with explicit weights so the
relative strengths are stable across runs. Without this, diffusers' default
behaviour of "last loaded wins" is hard to reason about.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, Optional

from ..config import Project


def generate(
    project: Project,
    prompt: str,
    negative: str = "",
    n: int = 4,
    steps: int = 30,
    guidance: float = 7.0,
    seed: int | None = None,
    use_trained_lora: bool = True,
    extra_loras: Optional[Iterable[tuple[Path, float]]] = None,
    width: int = 1024,
    height: int = 1024,
    sampler: str = "default",
) -> list[Path]:
    """Generate `n` images with the project's base checkpoint and optional LoRAs.

    Args:
        project: Loaded project; ``base_model_path`` must exist.
        prompt: Positive text prompt. Include the project's trigger word when
            ``use_trained_lora=True``; omit it for a vanilla base-model render.
        negative: Negative prompt (``""`` disables it).
        n: Number of images to generate.
        steps: Inference denoising steps. 25-40 is the usual useful range.
        guidance: Classifier-free guidance scale. 5-7 for photorealistic
            checkpoints, 7-9 for stylised ones.
        seed: Optional integer seed. ``None`` = non-deterministic.
        use_trained_lora: If True, load the project's trained LoRA from
            ``project.lora_dir``. If False, render with just the base
            checkpoint + any extra LoRAs (vanilla text-to-image).
        extra_loras: Optional iterable of ``(safetensors_path, weight)`` tuples
            for community LoRAs. Each is loaded as a separate adapter and
            combined with the trained one (when present) via ``set_adapters``.

    Returns:
        Paths to the saved PNGs under ``outputs/<timestamp>/``.

    Raises:
        ValueError: if the project has no base checkpoint configured.
        FileNotFoundError: if ``use_trained_lora=True`` but the project's
            ``lora/`` directory is empty (i.e. training hasn't run yet),
            or if any extra LoRA path doesn't exist.
    """
    import torch
    from diffusers import StableDiffusionXLPipeline

    if project.base_model_path is None:
        raise ValueError(
            "No base checkpoint configured. Set `base_model_path` in config.json "
            "or fill it in the GUI Settings tab before running `trainer generate`."
        )

    extras = list(extra_loras or [])
    for p, _w in extras:
        if not Path(p).exists():
            raise FileNotFoundError(f"Extra LoRA not found: {p}")

    if use_trained_lora and (
        not project.lora_dir.exists() or not any(project.lora_dir.iterdir())
    ):
        raise FileNotFoundError(
            f"No trained LoRA found at {project.lora_dir}. Run `trainer train` "
            f"first, or pass --no-trained-lora to render with the base model only."
        )

    base = project.base_model_path
    if base.suffix == ".safetensors" and base.is_file():
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(base), torch_dtype=torch.float16
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(base), torch_dtype=torch.float16
        )

    # ---- LoRA stacking ----
    # Load adapters one at a time with distinct names, then activate them
    # together with their relative weights. This is more stable than the
    # implicit "last loaded wins" behaviour and keeps the output reproducible
    # across runs that use the same set.
    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    if use_trained_lora:
        pipe.load_lora_weights(
            str(project.lora_dir), adapter_name="trained",
        )
        adapter_names.append("trained")
        adapter_weights.append(1.0)
        print(f"Loaded trained LoRA: {project.lora_dir}", flush=True)
    for i, (lora_path, weight) in enumerate(extras):
        # adapter_name has to be a valid identifier — sanitise the stem.
        stem = Path(lora_path).stem
        safe = "".join(c if c.isalnum() else "_" for c in stem) or f"extra_{i}"
        # Avoid collision with "trained" if someone literally named a LoRA that.
        if safe == "trained":
            safe = f"trained_extra_{i}"
        pipe.load_lora_weights(
            str(Path(lora_path).parent),
            weight_name=Path(lora_path).name,
            adapter_name=safe,
        )
        adapter_names.append(safe)
        adapter_weights.append(float(weight))
        print(f"Loaded extra LoRA: {lora_path} @ weight {weight}", flush=True)

    if adapter_names:
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
    elif not use_trained_lora and not extras:
        # Pure base-model render — log so the user is sure we did what they asked.
        print("Base-model render (no LoRAs).", flush=True)

    # Optional sampler swap. Diffusers' default for SDXL is EulerDiscrete
    # but DPM++ 2M and UniPC are widely preferred for portrait realism +
    # NSFW work because they converge faster (good output at 20-25 steps
    # vs 30-40 for Euler). Each scheduler is constructed from the pipe's
    # current scheduler config so SDXL-specific timestep settings carry
    # over.
    if sampler and sampler != "default":
        from diffusers import (
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            UniPCMultistepScheduler,
        )
        scheduler_map = {
            "euler": EulerDiscreteScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
            "dpmpp_2m": DPMSolverMultistepScheduler,
            "dpmpp_2m_karras": DPMSolverMultistepScheduler,
            "unipc": UniPCMultistepScheduler,
        }
        cls = scheduler_map.get(sampler)
        if cls is not None:
            kwargs = {}
            if sampler == "dpmpp_2m_karras":
                kwargs["use_karras_sigmas"] = True
            pipe.scheduler = cls.from_config(pipe.scheduler.config, **kwargs)
            print(f"Sampler set to {sampler}", flush=True)
        else:
            print(f"Unknown sampler {sampler!r}; keeping default.", flush=True)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.enable_model_cpu_offload()

    out_dir = project.outputs_dir / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Empty string disables the negative prompt — make the intent explicit
    # rather than relying on truthy-to-None coercion.
    negative_prompt = negative if negative else None

    results: list[Path] = []
    for i in range(n):
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            width=width,
            height=height,
        ).images[0]
        out_path = out_dir / f"{i:03d}.png"
        image.save(out_path)
        print(f"Saved {out_path}", flush=True)
        results.append(out_path)

    return results
