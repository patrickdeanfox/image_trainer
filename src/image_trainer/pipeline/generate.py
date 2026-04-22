"""Inference step: load base SDXL + trained LoRA and generate images.

Step 6 of the pipeline. Uses :meth:`StableDiffusionXLPipeline.enable_model_cpu_offload`
to sequence sub-module loads across GPU and CPU so the whole pipeline fits
on a 10 GB card. Outputs are grouped under ``outputs/<timestamp>/`` so
separate generate calls don't overwrite each other.

LoRA weights are loaded from the project's ``lora/`` directory (the PEFT-
format export written by :func:`pipeline.train.train_lora`). Use the same
base checkpoint the LoRA was trained on — mixing families produces
surprising results.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from ..config import Project


def generate(
    project: Project,
    prompt: str,
    negative: str = "",
    n: int = 4,
    steps: int = 30,
    guidance: float = 7.0,
    seed: int | None = None,
) -> list[Path]:
    """Generate `n` images with the project's base checkpoint + trained LoRA.

    Args:
        project: Loaded project; ``base_model_path`` and ``lora_dir`` must exist.
        prompt: Positive text prompt. Include the project's trigger word.
        negative: Negative prompt (``""`` disables it). The GUI pre-fills this
            from :attr:`Project.default_negative_prompt`.
        n: Number of images to generate. Each is run sequentially with the
            same seed generator for determinism.
        steps: Inference denoising steps. 25-40 is the usual useful range.
        guidance: Classifier-free guidance scale. 5-7 for photorealistic
            checkpoints, 7-9 for more stylized ones.
        seed: Optional integer seed. ``None`` = non-deterministic.

    Returns:
        Paths to the saved PNGs under ``outputs/<timestamp>/``.

    Raises:
        ValueError: if the project has no base checkpoint configured.
        FileNotFoundError: if the project's ``lora/`` directory is empty
            (i.e. :func:`train_lora` hasn't run yet).
    """
    import torch
    from diffusers import StableDiffusionXLPipeline

    if project.base_model_path is None:
        raise ValueError(
            "No base checkpoint configured. Set `base_model_path` in config.json "
            "or fill it in the GUI Settings tab before running `trainer generate`."
        )
    if not project.lora_dir.exists() or not any(project.lora_dir.iterdir()):
        raise FileNotFoundError(
            f"No trained LoRA found at {project.lora_dir}. Run `trainer train` first."
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

    pipe.load_lora_weights(str(project.lora_dir))
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

    results: list[Path] = []
    for i in range(n):
        image = pipe(
            prompt=prompt,
            negative_prompt=negative or None,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]
        out_path = out_dir / f"{i:03d}.png"
        image.save(out_path)
        print(f"Saved {out_path}", flush=True)
        results.append(out_path)

    return results
