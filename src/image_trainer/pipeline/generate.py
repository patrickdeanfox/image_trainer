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


def _encode_long_prompt(pipe, prompt: str, negative: Optional[str]):
    """Encode prompts of any length using ``compel``, bypassing CLIP's 77-token cap.

    SDXL's CLIP encoders truncate at 77 tokens. With the Pony score-stack
    opener already eating ~10 tokens, that leaves ~60 for actual content —
    nowhere near enough for the descriptive prompts NSFW work needs. The
    `compel` library chunks long prompts into 77-token windows and
    concatenates the resulting embeddings before they hit the UNet, which
    is the standard fix for this in the diffusers ecosystem.

    Returns ``(prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds,
    negative_pooled_prompt_embeds)`` ready to pass to the pipeline. Returns
    ``None`` if compel isn't installed — caller falls back to the truncated
    raw-string path.
    """
    try:
        from compel import Compel, ReturnedEmbeddingsType  # type: ignore
    except ImportError:
        return None

    # NOTE: variable name is `encoder` not `compel` to avoid shadowing the
    # `compel` module — confusing for readers + future-proof if anyone adds
    # a `from compel import something_else` to this function.
    encoder = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
    )
    pos_embeds, pos_pooled = encoder(prompt)
    neg_embeds, neg_pooled = encoder(negative or "")
    # Both conditioning tensors have to share the same sequence length —
    # compel pads them up to whichever is longer.
    pos_embeds, neg_embeds = encoder.pad_conditioning_tensors_to_same_length(
        [pos_embeds, neg_embeds]
    )
    return pos_embeds, pos_pooled, neg_embeds, neg_pooled


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
    output_name: str = "",
    compare_stacks: bool = False,
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

    # ---- Sampler swap (BEFORE LoRA load) ----
    # Diffusers' default for SDXL is EulerDiscrete but DPM++ 2M and UniPC
    # are widely preferred for portrait realism + NSFW work because they
    # converge faster (good output at 20-25 steps vs 30-40 for Euler).
    # Each scheduler is constructed from the pipe's current scheduler
    # config so SDXL-specific timestep settings carry over.
    #
    # We swap the scheduler BEFORE loading LoRA adapters so any scheduler-
    # specific state initialisation that the LoRA loader's
    # set_adapters() depends on sees a clean baseline.
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

    # ---- LoRA stacking ----
    # Load adapters one at a time with distinct names, then activate them
    # together with their relative weights. This is more stable than the
    # implicit "last loaded wins" behaviour and keeps the output reproducible
    # across runs that use the same set.
    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    if use_trained_lora:
        # Be EXPLICIT about which file to load. train.py writes both:
        #   <lora_dir>/pytorch_lora_weights.safetensors  (diffusers flat format)
        #   <lora_dir>/unet/adapter_model.safetensors    (PEFT subdir format)
        # Without weight_name, diffusers' loader has historically dispatched
        # ambiguously between these — which is exactly the
        # "no file named pytorch_lora_weights.bin" error the user hit even
        # though the file existed. Naming the file explicitly removes any
        # heuristic from the equation.
        pipe.load_lora_weights(
            str(project.lora_dir),
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="trained",
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

    # (Sampler swap moved to BEFORE LoRA load above.)

    # ---- Prompt encoding (BEFORE offload setup) ----
    # Empty string disables the negative prompt — make the intent explicit
    # rather than relying on truthy-to-None coercion.
    negative_prompt = negative if negative else None

    # ---- Prompt encoding (all of it, BEFORE enable_model_cpu_offload) ----
    #
    # Compel must run BEFORE enable_model_cpu_offload. The offload hook
    # wraps the text encoders with accelerate magic that manages device
    # placement per-forward; compel bypasses that path and puts its own
    # tokens on the encoder's naive `.device`, which collides with the
    # hook mid-forward and throws:
    #   RuntimeError: Expected all tensors to be on the same device
    #
    # In normal mode that means encoding the single prompt here; in
    # compare-stacks mode it means encoding the prompt for EVERY stack
    # upfront (one set of embeddings per stack). The text encoders go to
    # CPU once after all encoding is done, then enable_model_cpu_offload
    # takes over for the UNet + VAE inference loop. Re-engaging the text
    # encoders later is a known-broken path under offload — once the hook
    # is on, the TEs aren't truly movable anymore.
    if torch.cuda.is_available():
        try:
            pipe.text_encoder.to("cuda")
            pipe.text_encoder_2.to("cuda")
        except Exception:
            pass

    # `precomputed` holds either:
    #   - normal mode: a single embed tuple (pos, pooled, neg, neg_pooled)
    #     OR None when compel isn't installed (raw-string fallback).
    #   - compare mode: a list of (stack_label, full_prompt, embed_tuple_or_None).
    precomputed: object = None  # populated below
    if compare_stacks:
        from ..prompt_presets import stacks_for_compare
        stack_iter = stacks_for_compare()
        compare_entries: list[tuple[str, str, object]] = []
        for stack_label, stack_prefix in stack_iter:
            full_prompt = f"{stack_prefix}{prompt}"
            embeds = _encode_long_prompt(pipe, full_prompt, negative_prompt)
            compare_entries.append((stack_label, full_prompt, embeds))
        precomputed = compare_entries
        # Whether compel is active determined by the FIRST entry — if compel
        # is missing they're all None, if compel works they're all populated.
        compel_active = compare_entries[0][2] is not None if compare_entries else False
        if compel_active:
            print(
                f"Compare-stacks: encoded {len(compare_entries)} prompts via compel.",
                flush=True,
            )
    else:
        long_prompt = _encode_long_prompt(pipe, prompt, negative_prompt)
        if long_prompt is None:
            # Fallback path — warn if the prompt is going to truncate.
            try:
                tok_count = len(pipe.tokenizer.tokenize(prompt))
            except Exception:
                tok_count = 0
            if tok_count > 75:
                print(
                    f"WARNING: prompt is ~{tok_count} tokens; CLIP's 77-token cap "
                    f"means everything past ~token 77 will be silently dropped. "
                    f"Install 'compel' for long-prompt support: pip install compel",
                    flush=True,
                )
            compel_active = False
        else:
            print(
                f"Long-prompt encoding via compel "
                f"(prompt embeds shape: {tuple(long_prompt[0].shape)})",
                flush=True,
            )
            compel_active = True
        precomputed = long_prompt  # tuple-or-None

    # ---- Offload setup (AFTER ALL prompt encoding) ----
    # Move text encoders off GPU + enable offload exactly once.
    import gc as _gc
    try:
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
    except Exception:
        pass
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.enable_model_cpu_offload()

    # Output dir naming. By default we get an unambiguous, sortable
    # YYYYMMDD_HHMMSS folder. When the caller passes ``output_name``, it's
    # sanitised (filesystem-safe) and prefixed onto the same timestamp so
    # the user can scan their outputs/ folder by purpose ("photoshoot",
    # "comparison_run", etc.) while keeping a unique per-run suffix.
    import random
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if compare_stacks:
        folder = f"stack_compare_{timestamp}"
    elif output_name and output_name.strip():
        safe = "".join(
            c if (c.isalnum() or c in "_-") else "_"
            for c in output_name.strip()
        ).strip("_")
        folder = f"{safe}_{timestamp}" if safe else timestamp
    else:
        folder = timestamp
    out_dir = project.outputs_dir / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []
    # Per-image seed tracking so filenames include the seed and run_info.txt
    # can list them. When the user doesn't pass a seed we draw a random
    # uint32 per image — random-but-known, so the filename still carries it
    # and the image is reproducible by feeding that seed back.
    image_seeds: list[int] = []

    if compare_stacks:
        # ``precomputed`` is the list of (label, full_prompt, embed_tuple)
        # populated above (BEFORE offload) — encoding can no longer happen
        # at this point because the text encoders are wrapped by accelerate.
        compare_entries = precomputed  # type: ignore[assignment]
        # Compare mode uses one shared seed so the ONLY variable between
        # outputs is the stack itself.
        shared_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        print(
            f"Compare-stacks render: {len(compare_entries)} stacks, "
            f"shared seed {shared_seed}",
            flush=True,
        )
        for idx, (stack_label, full_prompt, embeds) in enumerate(compare_entries):
            generator = torch.Generator(device="cuda").manual_seed(shared_seed)
            if embeds is not None:
                se, sp, ne, np_ = embeds
                image = pipe(
                    prompt_embeds=se, pooled_prompt_embeds=sp,
                    negative_prompt_embeds=ne, negative_pooled_prompt_embeds=np_,
                    num_inference_steps=steps, guidance_scale=guidance,
                    generator=generator, width=width, height=height,
                ).images[0]
            else:
                # Compel not installed — raw-string fallback per stack.
                image = pipe(
                    prompt=full_prompt, negative_prompt=negative_prompt,
                    num_inference_steps=steps, guidance_scale=guidance,
                    generator=generator, width=width, height=height,
                ).images[0]
            safe_label = "".join(
                c if (c.isalnum() or c in "_-") else "_" for c in stack_label
            ).strip("_")
            out_path = out_dir / f"{idx:02d}_{safe_label}_seed{shared_seed}.png"
            image.save(out_path)
            print(
                f"[{idx+1}/{len(compare_entries)}] {stack_label} → {out_path.name}",
                flush=True,
            )
            results.append(out_path)
            image_seeds.append(shared_seed)

        _write_run_info(
            out_dir=out_dir, mode="compare_stacks",
            project=project, base_model_path=project.base_model_path,
            body_or_prompt=prompt, negative=negative_prompt or "",
            sampler=sampler, steps=steps, guidance=guidance,
            width=width, height=height,
            use_trained_lora=use_trained_lora, extras=extras,
            seeds=image_seeds,
            stack_labels=[lbl for lbl, _, _ in compare_entries],
            compel_active=compel_active,
        )
        return results

    # Normal (non-compare) path. Unpack the single-prompt tuple.
    if precomputed is None:
        prompt_embeds = pooled_embeds = neg_embeds = neg_pooled = None
    else:
        prompt_embeds, pooled_embeds, neg_embeds, neg_pooled = precomputed  # type: ignore[misc]

    # Pre-compute the seed list so filenames embed the seed + run_info.txt
    # can list them.
    if seed is None:
        image_seeds = [random.randint(0, 2**32 - 1) for _ in range(n)]
    else:
        image_seeds = [seed + i for i in range(n)]

    for i in range(n):
        # Per-image generator with the seed we picked above. Fresh generator
        # per iteration so rerunning with the same seed reproduces every
        # image, not just image 0.
        s = image_seeds[i]
        generator = torch.Generator(device="cuda").manual_seed(s)
        if prompt_embeds is not None:
            image = pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                negative_prompt_embeds=neg_embeds,
                negative_pooled_prompt_embeds=neg_pooled,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                width=width,
                height=height,
            ).images[0]
        else:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                width=width,
                height=height,
            ).images[0]
        out_path = out_dir / f"{i:03d}_seed{s}.png"
        image.save(out_path)
        print(f"Saved {out_path}", flush=True)
        results.append(out_path)

    _write_run_info(
        out_dir=out_dir, mode="generate",
        project=project, base_model_path=project.base_model_path,
        body_or_prompt=prompt, negative=negative_prompt or "",
        sampler=sampler, steps=steps, guidance=guidance,
        width=width, height=height,
        use_trained_lora=use_trained_lora, extras=extras,
        seeds=image_seeds, stack_labels=None,
        compel_active=prompt_embeds is not None,
    )
    return results


def _write_run_info(
    *,
    out_dir: Path,
    mode: str,
    project: Project,
    base_model_path: Optional[Path],
    body_or_prompt: str,
    negative: str,
    sampler: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    use_trained_lora: bool,
    extras: list,
    seeds: list[int],
    stack_labels: Optional[list[str]],
    compel_active: bool,
) -> None:
    """Write ``run_info.txt`` into the output directory so the user can
    reference the exact settings that produced a set of images.

    Format is human-readable (grep-friendly) rather than JSON so a user
    can scan dozens of outputs with `less` / `head`. One line per setting,
    one line per output image with its seed, and the full prompt + negative
    in dedicated blocks at the end (which are the longest values).
    """
    lines: list[str] = []
    lines.append(f"run_info")
    lines.append(f"==========")
    lines.append(f"mode              : {mode}")
    lines.append(f"timestamp         : {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"project           : {project.root.name}  ({project.root})")
    lines.append(f"base checkpoint   : {base_model_path}")
    lines.append(f"sampler           : {sampler}")
    lines.append(f"steps             : {steps}")
    lines.append(f"guidance (CFG)    : {guidance}")
    lines.append(f"dimensions        : {width} x {height}")
    lines.append(f"use trained LoRA  : {use_trained_lora}")
    lines.append(f"long-prompt compel: {'yes' if compel_active else 'no (raw-string fallback)'}")
    if extras:
        lines.append("extra LoRAs       :")
        for path, weight in extras:
            lines.append(f"  - {path}  @ weight {weight}")
    else:
        lines.append("extra LoRAs       : (none)")

    lines.append("")
    lines.append("outputs")
    lines.append("-------")
    saved_pngs = sorted(p for p in out_dir.iterdir() if p.suffix == ".png")
    if stack_labels is not None:
        # compare_stacks mode — pair filenames with the stack that made them.
        for idx, p in enumerate(saved_pngs):
            label = stack_labels[idx] if idx < len(stack_labels) else "?"
            lines.append(f"  {p.name}  ·  stack: {label}")
    else:
        for i, p in enumerate(saved_pngs):
            s = seeds[i] if i < len(seeds) else "?"
            lines.append(f"  {p.name}  ·  seed: {s}")

    lines.append("")
    lines.append("prompt")
    lines.append("------")
    lines.append(body_or_prompt)
    lines.append("")
    lines.append("negative")
    lines.append("--------")
    lines.append(negative or "(none)")
    lines.append("")

    info_path = out_dir / "run_info.txt"
    try:
        info_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {info_path}", flush=True)
    except OSError as e:
        print(f"WARNING: couldn't write run_info.txt ({e})", flush=True)
