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
import json as _json
import re as _re
import struct as _struct
from pathlib import Path
from typing import Iterable, Optional

from ..config import Project


# ---- LoRA architecture pre-flight --------------------------------------------
#
# Diffusers' `pipe.load_lora_weights` fails loudly but late on non-SDXL or
# malformed LoRAs — after a 10–15 s base-checkpoint load, sometimes halfway
# through a multi-LoRA stack. The error messages PEFT surfaces ("Target
# modules … not found in the base model" / "Invalid LoRA checkpoint") don't
# tell the user WHY the file is wrong, just that it is.
#
# This pre-flight reads only the safetensors header (≪1 MB, ~1 ms per file),
# inspects the tensor-key naming + metadata, and blocks incompatible files
# before we commit any meaningful time. Three classes of rejection:
#
#   * architecture mismatch — FLUX, SD3, Anima, DiT — hard-incompatible with SDXL.
#   * hybrid malformed format — e.g. mostly-kohya keys but with a handful of
#     ComfyUI-flattened `diffusion_model_*` keys that diffusers chokes on.
#     Empirically observed on at least one civitai detail-enhancer LoRA.
#   * unknown format — no recognizable SDXL markers AND no incompatible markers.
#     Passed through with a warning; diffusers may still accept or reject it.

_DIT_FLUX        = _re.compile(r"^diffusion_model\.(?:double_blocks|single_blocks)\.")
_DIT_GENERIC     = _re.compile(r"^diffusion_model\.layers\.\d+\.(?:adaLN_modulation|attention|w\d)")
_SD3_JOINT       = _re.compile(r"^joint_blocks\.")
_ANIMA           = _re.compile(r"^lora_unet_blocks_\d+_adaln_modulation")
_SDXL_KOHYA_UNET = _re.compile(r"^lora_unet_")
_SDXL_KOHYA_TE   = _re.compile(r"^lora_te[12]?_")
_SDXL_DIFFUSERS  = _re.compile(r"^unet\.")
# ComfyUI-flattened raw names (underscore-joined, no `lora_` prefix). Diffusers
# can't parse these; presence even in small numbers breaks the whole load.
_COMFY_FLAT      = _re.compile(r"^diffusion_model_")


def _read_safetensors_header(path: Path) -> dict:
    """Read a safetensors file's JSON header without loading any tensors.

    The safetensors format is: 8-byte little-endian header length, then that
    many bytes of UTF-8 JSON, then the raw tensor bytes. We only need the
    JSON, which is usually under 1 MB and lists every tensor's name/shape
    plus an optional ``__metadata__`` blob written by the training script.
    """
    with open(path, "rb") as f:
        header_len = _struct.unpack("<Q", f.read(8))[0]
        return _json.loads(f.read(header_len).decode("utf-8"))


def _classify_lora(path: Path) -> tuple[str, str]:
    """Return ``(verdict, reason)`` for an extra LoRA, for SDXL compatibility.

    Verdict is one of:
      * ``"ok"``            — SDXL-compatible kohya or diffusers format.
      * ``"incompatible"``  — wrong architecture or malformed hybrid format;
        refuse to load.
      * ``"warn"``          — unknown format or header unreadable; let the
        load proceed and surface whatever diffusers says.

    The reason is a short human-readable explanation for the GUI/CLI user.
    """
    try:
        header = _read_safetensors_header(path)
    except Exception as e:
        return ("warn", f"couldn't read safetensors header ({e}); letting load proceed")

    meta = header.get("__metadata__") or {}
    keys = [k for k in header.keys() if k != "__metadata__"]

    # Fast metadata-level architecture check. Both `modelspec.architecture`
    # (sai_model_spec) and `ss_base_model_version` (kohya sd-scripts) can
    # carry the info; some files set one, some the other, some neither.
    arch_meta = " ".join(
        s.lower() for s in (
            meta.get("modelspec.architecture"),
            meta.get("ss_base_model_version"),
        ) if s
    )
    if "flux" in arch_meta:
        return ("incompatible",
                f"FLUX LoRA (metadata: {arch_meta!r}); SDXL base cannot load it")
    if "sd3" in arch_meta or "stable-diffusion-3" in arch_meta:
        return ("incompatible",
                f"SD3 LoRA (metadata: {arch_meta!r}); SDXL base cannot load it")
    if arch_meta.strip() == "anima":
        return ("incompatible",
                "Anima architecture LoRA (Lenovo transformer); SDXL base cannot load it")

    # Key-level architecture check. This catches files with missing/misleading
    # metadata — e.g. FLUX LoRAs with no modelspec entry. We only need one
    # diagnostic hit; if we see a DiT-family key, we're done.
    hits = {"flux": 0, "dit": 0, "sd3": 0, "anima": 0,
            "sdxl_unet": 0, "sdxl_te": 0, "sdxl_diff": 0, "comfy": 0}
    first_bad_key = {}
    for k in keys:
        if _DIT_FLUX.match(k):
            hits["flux"] += 1; first_bad_key.setdefault("flux", k)
        elif _DIT_GENERIC.match(k):
            hits["dit"] += 1;  first_bad_key.setdefault("dit", k)
        if _SD3_JOINT.match(k):
            hits["sd3"] += 1;  first_bad_key.setdefault("sd3", k)
        if _ANIMA.match(k):
            hits["anima"] += 1; first_bad_key.setdefault("anima", k)
        if _SDXL_KOHYA_UNET.match(k):
            hits["sdxl_unet"] += 1
        if _SDXL_KOHYA_TE.match(k):
            hits["sdxl_te"] += 1
        if _SDXL_DIFFUSERS.match(k):
            hits["sdxl_diff"] += 1
        # Only count as comfy-flat when it's NOT the kohya `lora_unet_`
        # prefix, i.e. the raw `diffusion_model_xxx` underscore form.
        if _COMFY_FLAT.match(k) and not _SDXL_KOHYA_UNET.match(k):
            hits["comfy"] += 1; first_bad_key.setdefault("comfy", k)

    if hits["flux"]:
        return ("incompatible",
                f"FLUX architecture detected (key: {first_bad_key['flux']!r})")
    if hits["dit"]:
        return ("incompatible",
                f"DiT architecture detected (key: {first_bad_key['dit']!r})")
    if hits["sd3"]:
        return ("incompatible",
                f"SD3 architecture detected (key: {first_bad_key['sd3']!r})")
    if hits["anima"]:
        return ("incompatible",
                f"Anima architecture detected (key: {first_bad_key['anima']!r})")

    # Hybrid/malformed format: some valid SDXL keys mixed with raw ComfyUI
    # `diffusion_model_*` keys. Diffusers rejects the whole file even when
    # it's mostly SDXL. Block it with a specific message.
    if hits["comfy"]:
        return ("incompatible",
                f"contains {hits['comfy']} non-standard 'diffusion_model_*' "
                f"key(s) (e.g. {first_bad_key['comfy']!r}) that diffusers "
                f"cannot parse. This is usually a ComfyUI-only LoRA variant; "
                f"find a kohya-format version of the same model")

    # At this point, no incompatible markers — does it look SDXL at all?
    if hits["sdxl_unet"] or hits["sdxl_te"] or hits["sdxl_diff"]:
        return ("ok", "SDXL")

    return ("warn",
            "no recognized SDXL / FLUX / SD3 markers in key names; format is "
            "unknown, letting diffusers try to load it")


def _preflight_extra_loras(extras: list) -> None:
    """Validate existence + architecture of every extra LoRA before the base
    pipe load. Raises :class:`ValueError` (aggregated) for hard-incompatible
    files; prints a warning for unknown-format files and returns.

    Aggregating failures into a single ValueError means the user sees every
    problem at once instead of fix-one-rerun-find-next.
    """
    problems: list[tuple[Path, str]] = []
    warnings_: list[tuple[Path, str]] = []
    for lora_path, _weight in extras:
        p = Path(lora_path)
        if not p.exists():
            problems.append((p, "file not found"))
            continue
        verdict, reason = _classify_lora(p)
        if verdict == "incompatible":
            problems.append((p, reason))
        elif verdict == "warn":
            warnings_.append((p, reason))

    for p, r in warnings_:
        print(f"WARNING: {p.name}: {r}", flush=True)

    if problems:
        lines = ["Cannot load one or more extra LoRAs:"]
        for p, r in problems:
            lines.append(f"  - {p.name}: {r}")
        lines.append("")
        lines.append(
            "Remove the incompatible LoRA(s) from --extra-lora, or replace "
            "them with SDXL versions from Civitai."
        )
        raise ValueError("\n".join(lines))


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
    compare_loras: bool = False,
    lora_recipes: Optional[list[tuple[str, list[tuple[Path, float]]]]] = None,
    compare_stacks_subset: Optional[list[str]] = None,
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
    # Validate existence + SDXL compatibility of every extra LoRA BEFORE we
    # spend ~15 s loading the base checkpoint. Reads each file's safetensors
    # header only (tiny). See `_classify_lora` above for the rules.
    _preflight_extra_loras(extras)

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
    #
    # In compare_loras mode the loop is different: we load EVERY unique
    # adapter that appears in ANY recipe, and defer the `set_adapters`
    # call to the per-render loop below (where each recipe activates its
    # own subset). Reason: diffusers can load adapters dynamically but
    # un-loading or re-loading mid-pipe-offload is brittle — load-once,
    # switch-via-set_adapters is the stable pattern.
    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    # Map from path-stem → sanitised adapter_name, used by compare_loras
    # to look up adapters when assembling per-recipe set_adapters calls.
    stem_to_adapter: dict[str, str] = {}

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

    # Build the list of unique extra-LoRA paths to load. In normal mode
    # that's just `extras`. In compare_loras mode it's the union of
    # every (path, weight) across every recipe — load once, weight-switch
    # at render time.
    if compare_loras and lora_recipes:
        seen_paths: set[Path] = set()
        all_recipe_loras: list[tuple[Path, float]] = []
        for _label, recipe in lora_recipes:
            for path, weight in recipe:
                p = Path(path)
                if p in seen_paths:
                    continue
                seen_paths.add(p)
                # Use weight 1.0 here — the per-recipe weight is set later
                # by set_adapters; the load-time weight is irrelevant.
                all_recipe_loras.append((p, 1.0))
        loras_to_load = all_recipe_loras
        # Re-run the existence + compat pre-flight against the union set
        # so we fail fast if any recipe references an incompatible LoRA.
        _preflight_extra_loras(loras_to_load)
    else:
        loras_to_load = extras

    for i, (lora_path, weight) in enumerate(loras_to_load):
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
        stem_to_adapter[stem] = safe
        # In normal mode we also build the active-adapter list. In
        # compare_loras mode that list stays empty here — set_adapters
        # is called per-recipe in the render loop.
        if not compare_loras:
            adapter_names.append(safe)
            adapter_weights.append(float(weight))
            print(f"Loaded extra LoRA: {lora_path} @ weight {weight}", flush=True)
        else:
            print(f"Loaded extra LoRA (compare_loras pool): {lora_path}", flush=True)

    if compare_loras:
        # In compare_loras mode the render loop drives set_adapters per
        # recipe; nothing to activate here yet. The trained LoRA is the
        # only adapter loaded via the use_trained_lora path above; we
        # still want it active for any recipe that includes it, so we
        # cache its name for the recipe-activation step below.
        pass
    elif adapter_names:
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
    #   - compare-stacks mode: a list of (stack_label, full_prompt, embed_tuple_or_None).
    #   - compare-loras mode: a list of (stack_label, full_prompt, embed_tuple_or_None)
    #     for the stack subset selected by the caller (just one entry if
    #     cross-with-stacks is off).
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
    elif compare_loras:
        # Encode one prompt-embedding per quality-stack in the chosen
        # subset. If no subset was supplied (or empty), default to a
        # single empty-prefix entry so every recipe still gets rendered
        # with the user's already-prepended prompt.
        from ..prompt_presets import stack_label_to_prefix
        labels = compare_stacks_subset or ["(current)"]
        compare_entries = []
        for stack_label in labels:
            prefix = stack_label_to_prefix(stack_label) if stack_label != "(current)" else ""
            full_prompt = f"{prefix}{prompt}"
            embeds = _encode_long_prompt(pipe, full_prompt, negative_prompt)
            compare_entries.append((stack_label, full_prompt, embeds))
        precomputed = compare_entries
        compel_active = (
            compare_entries[0][2] is not None if compare_entries else False
        )
        if compel_active:
            print(
                f"Compare-loras: encoded {len(compare_entries)} stack prompts via compel.",
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
    elif compare_loras:
        folder = f"lora_compare_{timestamp}"
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

    if compare_loras:
        # Outer loop: stacks (one or many depending on cross-with-stacks).
        # Inner loop: recipes (each defines a subset of loaded adapters
        # plus their weights). Trained LoRA is implicitly active in
        # every recipe when use_trained_lora=True.
        compare_entries = precomputed  # type: ignore[assignment]
        recipes = lora_recipes or []
        if not recipes:
            raise ValueError(
                "compare_loras=True requires lora_recipes to be a non-empty "
                "list of (label, [(path, weight), ...]) tuples."
            )
        shared_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        total = len(compare_entries) * len(recipes)
        print(
            f"Compare-loras render: {len(recipes)} recipes × "
            f"{len(compare_entries)} stack(s) = {total} images, "
            f"shared seed {shared_seed}",
            flush=True,
        )
        rendered_idx = 0
        # Keep run_info recipe-stack pairing in order so users can map
        # filenames back to settings. Each entry is a stable index.
        recipe_stack_pairs: list[tuple[str, str]] = []
        for stack_idx, (stack_label, full_prompt, embeds) in enumerate(compare_entries):
            for recipe_idx, (recipe_label, recipe_loras) in enumerate(recipes):
                # Build the active-adapter set for this recipe. Trained
                # LoRA is included if the user toggle is on (it's already
                # loaded under adapter_name "trained" above).
                active_names: list[str] = []
                active_weights: list[float] = []
                if use_trained_lora:
                    active_names.append("trained")
                    active_weights.append(1.0)
                for path, weight in recipe_loras:
                    stem = Path(path).stem
                    aname = stem_to_adapter.get(stem)
                    if aname is None:
                        # Should never happen — pre-flight should have
                        # ensured every recipe path is loaded — but log
                        # and skip rather than crash.
                        print(
                            f"WARNING: recipe {recipe_label!r} references "
                            f"{path} which wasn't pre-loaded; skipping.",
                            flush=True,
                        )
                        continue
                    active_names.append(aname)
                    active_weights.append(float(weight))

                if active_names:
                    pipe.set_adapters(active_names, adapter_weights=active_weights)
                else:
                    # "no LoRAs" recipe — set every loaded adapter to
                    # weight 0 so the render is pure base-model. Calling
                    # set_adapters([]) is unsupported by some diffusers
                    # versions; passing all-loaded-with-zero-weight is
                    # the safe alternative.
                    every_loaded = list(stem_to_adapter.values())
                    if use_trained_lora:
                        every_loaded = ["trained"] + every_loaded
                    if every_loaded:
                        pipe.set_adapters(
                            every_loaded,
                            adapter_weights=[0.0] * len(every_loaded),
                        )

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
                    image = pipe(
                        prompt=full_prompt, negative_prompt=negative_prompt,
                        num_inference_steps=steps, guidance_scale=guidance,
                        generator=generator, width=width, height=height,
                    ).images[0]

                safe_recipe = "".join(
                    c if (c.isalnum() or c in "_-") else "_" for c in recipe_label
                ).strip("_")[:60]
                safe_stack = "".join(
                    c if (c.isalnum() or c in "_-") else "_" for c in stack_label
                ).strip("_")[:40]
                # Filename layout: NN_<recipe>__<stack>_seed<N>.png
                # The double underscore separates recipe from stack so
                # users can ls-sort by recipe across stacks.
                out_path = out_dir / (
                    f"{rendered_idx:02d}_{safe_recipe}__{safe_stack}"
                    f"_seed{shared_seed}.png"
                )
                image.save(out_path)
                print(
                    f"[{rendered_idx+1}/{total}] {recipe_label} × {stack_label} "
                    f"→ {out_path.name}",
                    flush=True,
                )
                results.append(out_path)
                image_seeds.append(shared_seed)
                recipe_stack_pairs.append((recipe_label, stack_label))
                rendered_idx += 1

        _write_run_info(
            out_dir=out_dir, mode="compare_loras",
            project=project, base_model_path=project.base_model_path,
            body_or_prompt=prompt, negative=negative_prompt or "",
            sampler=sampler, steps=steps, guidance=guidance,
            width=width, height=height,
            use_trained_lora=use_trained_lora,
            extras=[],  # extras are recipe-specific; the recipes list is below
            seeds=image_seeds,
            # Stuff the recipe×stack pairing into the stack_labels slot
            # so _write_run_info can emit it without a schema change.
            stack_labels=[f"{r} × {s}" for r, s in recipe_stack_pairs],
            compel_active=compel_active,
        )
        # Append a recipes block to run_info.txt so users can map the
        # numbered files back to exact (path, weight) lists.
        try:
            recipes_lines = ["", "recipes", "-------"]
            for r_label, r_loras in recipes:
                recipes_lines.append(f"  {r_label}:")
                if not r_loras:
                    recipes_lines.append("    (no extra LoRAs)")
                for path, weight in r_loras:
                    recipes_lines.append(f"    - {Path(path).name}  @ {weight}")
            recipes_lines.append("")
            recipes_lines.append("stacks tested")
            recipes_lines.append("-------------")
            for lbl in (compare_stacks_subset or ["(current)"]):
                recipes_lines.append(f"  - {lbl}")
            with open(out_dir / "run_info.txt", "a", encoding="utf-8") as f:
                f.write("\n".join(recipes_lines))
        except OSError as e:
            print(
                f"WARNING: couldn't append recipes to run_info.txt ({e})",
                flush=True,
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
