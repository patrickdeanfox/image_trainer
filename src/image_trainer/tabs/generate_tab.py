"""06 · Generate tab — base + (optional) trained LoRA + extras → images.

Built around the user's NSFW workflow with their own trained likeness LoRA.
Surfaces the levers that actually matter for explicit-content generation
without burying them in advanced menus:

- Quality-tag preset (Pony score_9 stack, Illustrious tag stack, plain SDXL)
- Prompt-template library (poses, framing, scenarios)
- NSFW negative-prompt preset
- Aspect-ratio quick-pick (SDXL-friendly buckets)
- Sampler picker (DPM++ 2M Karras / UniPC for fast convergence)
- Stack of community LoRAs from <projects_root>/shared_loras/
- "Use trained LoRA" toggle for vanilla-vs-LoRA comparisons
- An NSFW model-recommendation sidebar for which base + LoRAs work best

Nothing here generates content — the tab assembles the prompt + flags and
hands off to ``trainer generate``. All persistence is in-memory for the
session; presets are static lists so the Generate tab loads instantly.
"""

from __future__ import annotations

import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

from .. import gui_helpers, gui_theme
from ..gui_widgets import info_icon

if TYPE_CHECKING:
    from ..gui_app import TrainerGUI


# ---------- Preset content ----------
#
# These are the actual recipes the tab proposes. Kept as data tables at the
# top of the file so they're easy to tune without touching layout code.

#: Quality-tag stacks. Prepended to the user's prompt when picked.
#: Empty string = no prefix (the default).
QUALITY_STACKS: list[tuple[str, str, str]] = [
    (
        "(none)", "",
        "No quality prefix — your prompt stands alone. Use with non-Pony bases "
        "where the score_X tags do nothing.",
    ),
    (
        "Pony score_9 stack",
        "score_9, score_8_up, score_7_up, source_real, ",
        "The Pony Diffusion XL canonical 'photoreal' opener. "
        "Pony was trained with score_X tags as quality anchors — using them "
        "in your prompt is essentially mandatory for getting the best output. "
        "score_9 = highest-rated bucket; the 'up' variants extend the range.",
    ),
    (
        "Pony score_9 stack (anime)",
        "score_9, score_8_up, score_7_up, source_anime, ",
        "Same as above but with source_anime — pulls Pony toward its "
        "stylised-art training distribution. Use for anime / hentai / "
        "illustrated NSFW with a Pony-derived checkpoint.",
    ),
    (
        "Illustrious / NoobAI tag stack",
        "masterpiece, best quality, very aware, newest, ",
        "Tag stack for Illustrious-XL / NoobAI-XL family checkpoints. These "
        "use SD-style 'masterpiece, best quality' anchors plus 'very aware' "
        "(NoobAI's anatomy-conscious tag). Use with an Illustrious base.",
    ),
    (
        "Bare SDXL realism prefix",
        "raw photo, photograph, sharp focus, detailed skin, ",
        "Realism-tilted opener for vanilla SDXL or RealVis-family checkpoints "
        "that aren't trained with explicit quality tags. Steers toward "
        "photographic look.",
    ),
]


#: Prompt templates. The {trigger} token is substituted with the project's
#: trigger word so the trained LoRA is summoned. {body} marks where any
#: existing user prompt lands when the template is applied as a wrapper.
PROMPT_TEMPLATES: list[tuple[str, str, str]] = [
    (
        "Portrait · soft window light",
        "{trigger}, headshot portrait, soft natural window light, "
        "shallow depth of field, intimate framing, looking at viewer, "
        "{body}",
        "Tight portrait with cinematic lighting. Best for testing likeness; "
        "keep prompt short so the LoRA does the heavy lifting.",
    ),
    (
        "Boudoir · bedroom",
        "{trigger}, lying on bed, lingerie, bedroom, warm lamp light, "
        "tousled sheets, looking at viewer, suggestive pose, {body}",
        "Classic boudoir setup. Pair with Pony score_9 stack + a 'lingerie' "
        "or 'soft erotic' style LoRA for best results. Implicit/teasing, not "
        "explicit.",
    ),
    (
        "Nude · standing",
        "{trigger}, nude, standing, full body, soft studio lighting, "
        "neutral grey background, natural skin texture, detailed anatomy, "
        "looking at viewer, {body}",
        "Studio nude. Pony / Lustify-family bases handle this cleanly with "
        "the score stack. Add a 'detail enhancer' LoRA for skin micro-detail.",
    ),
    (
        "Nude · explicit POV",
        "{trigger}, nude, pov, intimate angle, soft natural light, "
        "detailed anatomy, explicit, looking at viewer, {body}",
        "First-person framing. Needs a Pony / NSFW-trained base + the "
        "'explicit' tag to bypass implicit framing. Some bases want 'rating "
        "explicit' instead of bare 'explicit' — try both.",
    ),
    (
        "Outdoor · golden hour",
        "{trigger}, outdoor, golden hour, warm sunlight, casual outfit, "
        "candid pose, depth of field background, {body}",
        "Outdoor SFW-ish framing. Use this as a baseline test of whether "
        "your LoRA generalises beyond bedroom shots — if the face stays "
        "consistent in different lighting, the LoRA is well-trained.",
    ),
    (
        "Selfie · phone POV",
        "{trigger}, selfie, holding phone, mirror, bedroom, looking at "
        "phone screen, casual outfit, instagram aesthetic, {body}",
        "Mimics the OnlyFans/Instagram aesthetic the source data was likely "
        "drawn from — closest to the LoRA's training distribution, so this "
        "produces the most reliable likeness.",
    ),
    (
        "Cosplay · stylised",
        "{trigger}, cosplay, dramatic pose, studio lighting, stylised "
        "outfit, {body}",
        "Skeleton for a cosplay shot. Stack with a character/style LoRA "
        "(e.g. 'sailor uniform LoRA') from civitai for the wardrobe.",
    ),
    (
        "Lingerie photoshoot",
        "{trigger}, lingerie, professional photoshoot, studio lighting, "
        "looking at viewer, confident pose, detailed fabric, {body}",
        "Polished lingerie set. Add 'detail enhancer' LoRA + a soft-skin "
        "LoRA at low weight (0.3-0.5) for magazine-style output.",
    ),
]


#: Negative-prompt presets. The user can pick one or chain them.
NEGATIVE_PRESETS: list[tuple[str, str, str]] = [
    (
        "(none)", "",
        "No negative prompt. Some samplers + bases work fine without one.",
    ),
    (
        "Standard quality",
        "low quality, blurry, deformed, extra fingers, bad anatomy, "
        "watermark, signature, jpeg artifacts, ugly",
        "Generic quality-suppression negatives. Safe baseline for any base.",
    ),
    (
        "NSFW · uncensor",
        "censored, mosaic, bar censor, pixelated, black bars, clothing, "
        "covered, low quality, blurry, deformed, extra fingers, bad anatomy, "
        "watermark, signature",
        "Pushes against the model's tendency to add censoring artefacts (mosaic, "
        "black bars) when generating explicit content. Pair with the score_9 "
        "Pony stack on a Pony / Lustify base for best results.",
    ),
    (
        "NSFW · realism push",
        "anime, illustration, drawing, painting, cartoon, 3d render, "
        "doll, plastic, low quality, blurry, deformed, extra fingers, "
        "bad anatomy, watermark, signature, censored, mosaic",
        "Suppresses stylised output to push toward photographic realism. "
        "Use with photoreal bases (RealVisXL, Lustify, JuggernautXL).",
    ),
    (
        "Anime / illustration push",
        "photo, photograph, photorealistic, 3d, realistic, low quality, "
        "blurry, deformed, extra fingers, bad anatomy, watermark, signature",
        "Inverse of the realism push — suppresses photo qualities to keep "
        "output drawn / illustrated. Use with Pony anime mode or "
        "Illustrious-XL.",
    ),
]


#: SDXL-friendly aspect ratio buckets. SDXL is happiest at one of its
#: training resolutions; off-bucket sizes work but produce more artefacts.
ASPECT_RATIOS: list[tuple[str, int, int]] = [
    ("Portrait 832×1216", 832, 1216),
    ("Portrait 896×1152", 896, 1152),
    ("Square 1024×1024", 1024, 1024),
    ("Landscape 1152×896", 1152, 896),
    ("Landscape 1216×832", 1216, 832),
]


#: NSFW-friendly base checkpoint families with one-line guidance.
BASE_RECOMMENDATIONS: list[tuple[str, str]] = [
    (
        "Pony Diffusion V6 XL",
        "Industry standard for NSFW. Tag-driven (score_9 stack required). "
        "Works for anime + photoreal via source_real / source_anime. Least "
        "censorship pushback of any common base.",
    ),
    (
        "Lustify XL",
        "Pony fine-tune dialled toward photoreal NSFW. Same prompt vocabulary. "
        "Better skin / lighting than vanilla Pony for photoreal subjects.",
    ),
    (
        "Pony Realism / Pony Real",
        "Another Pony fine-tune family. Try several — community-trained "
        "variants vary wildly in skin tone, anatomy bias, lighting style.",
    ),
    (
        "RealVisXL V4 / V5",
        "SDXL fine-tune for photoreal. Tighter on NSFW than Pony — needs the "
        "uncensor negative + 'rating explicit' style hints.",
    ),
    (
        "JuggernautXL",
        "Versatile SDXL fine-tune. Good photoreal baseline if you want fewer "
        "Pony-isms. Less explicit content out of the box.",
    ),
    (
        "Illustrious-XL / NoobAI-XL",
        "Anime/illustration NSFW. Different prompt vocabulary "
        "(masterpiece / very aware / etc.). Excellent for stylised work.",
    ),
]


#: Civitai LoRA categories worth searching for (no direct links —
#: civitai LoRAs come and go).
LORA_RECOMMENDATIONS: list[tuple[str, str]] = [
    (
        "Detail enhancer",
        "Search civitai for 'add detail XL' / 'detail tweaker XL'. Adds skin "
        "micro-texture, fabric weave, hair strands. Use at 0.3-0.6 weight; "
        "higher = noisy output.",
    ),
    (
        "Anatomy correction",
        "Search 'perfect anatomy' / 'better anatomy XL'. Fixes hand/foot "
        "deformity at the cost of some style. Use at 0.4-0.7.",
    ),
    (
        "Realistic skin",
        "Search 'realistic skin XL' / 'detailed skin XL'. Helps when your "
        "base or trained LoRA produces plastic-looking skin.",
    ),
    (
        "Lighting LoRAs",
        "Search 'cinematic lighting XL' / 'natural lighting XL'. Stack at "
        "0.4-0.7 to push specific lighting setups (golden hour, neon, "
        "studio key/fill).",
    ),
    (
        "Pose pack",
        "Search by pose name ('lying down XL', 'sitting XL', etc.). Pose "
        "LoRAs are the best way to get reliable composition without ControlNet.",
    ),
    (
        "Style LoRAs",
        "Search by aesthetic ('glamour XL', 'film grain XL', 'vintage 90s "
        "XL'). Stack at 0.3-0.6 to push the overall look without "
        "overpowering the trained likeness LoRA.",
    ),
]


# ---------- Builder ----------

def build(gui: "TrainerGUI") -> None:
    PAD = gui_theme.PAD
    f = gui.tab_generate
    f.columnconfigure(0, weight=1)

    gui.prompt_var = tk.StringVar(value="")
    gui.negative_var = tk.StringVar(value="")
    gui.n_var = tk.StringVar(value="4")
    gui.steps_var = tk.StringVar(value="28")
    gui.guidance_var = tk.StringVar(value="6.5")
    gui.seed_var = tk.StringVar(value="")
    gui.use_trained_lora_var = tk.BooleanVar(value=True)
    gui.sampler_var = tk.StringVar(value="dpmpp_2m_karras")
    gui.aspect_var = tk.StringVar(value="Portrait 832×1216")
    gui.quality_stack_var = tk.StringVar(value="(none)")

    state = _GenerateState(gui)
    gui.generate_state = state

    ttk.Label(
        f, text="Generate · prompt → images",
        style="Header.TLabel",
    ).grid(row=0, column=0, sticky="w", pady=(0, PAD))

    # ---- main two-column layout: form on the left, recommendations on the right.
    main = ttk.Frame(f)
    main.grid(row=1, column=0, sticky="nswe")
    main.columnconfigure(0, weight=2)
    main.columnconfigure(1, weight=1)

    form = ttk.Frame(main)
    form.grid(row=0, column=0, sticky="nswe", padx=(0, PAD))
    form.columnconfigure(0, weight=1)

    rec = ttk.Frame(main)
    rec.grid(row=0, column=1, sticky="nswe")
    rec.columnconfigure(0, weight=1)

    state.build_form(form)
    state.build_recommendations(rec)


class _GenerateState:
    """Holds the LoRA-stack rows + handlers for the Generate tab."""

    def __init__(self, gui: "TrainerGUI") -> None:
        self.gui = gui
        self.lora_rows: dict[str, dict] = {}
        self.lora_table: tk.Widget | None = None
        # The actual prompt body the user types (separated from any preset
        # template wrapping or quality-stack prefix).
        self.body_var = tk.StringVar(value="portrait, natural lighting")

    # ---- left column: form ----

    def build_form(self, root: ttk.Frame) -> None:
        PAD = gui_theme.PAD

        # Quality stack picker
        qs_box = ttk.LabelFrame(root, text="Quality-tag prefix", padding=PAD)
        qs_box.grid(row=0, column=0, sticky="we", pady=(0, PAD))
        qs_box.columnconfigure(1, weight=1)
        ttk.Label(qs_box, text="Stack:").grid(row=0, column=0, sticky="w")
        info_icon(
            qs_box,
            "Prepended to your prompt automatically. Pony bases require "
            "score_9, score_8_up etc. — without them output quality drops "
            "noticeably. Match this to your base checkpoint family.",
        ).grid(row=0, column=2, sticky="w")
        qs_combo = ttk.Combobox(
            qs_box,
            textvariable=self.gui.quality_stack_var,
            values=[label for label, _, _ in QUALITY_STACKS],
            state="readonly",
        )
        qs_combo.grid(row=0, column=1, sticky="we", padx=PAD)
        # Hover description below the combobox so users see what each stack does.
        self.qs_hint_var = tk.StringVar(value=QUALITY_STACKS[0][2])
        ttk.Label(
            qs_box, textvariable=self.qs_hint_var, style="Status.TLabel",
            wraplength=520, justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        qs_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_quality_stack_change(),
        )

        # Prompt template picker
        tpl_box = ttk.LabelFrame(root, text="Prompt template", padding=PAD)
        tpl_box.grid(row=1, column=0, sticky="we", pady=(0, PAD))
        tpl_box.columnconfigure(1, weight=1)
        ttk.Label(tpl_box, text="Apply:").grid(row=0, column=0, sticky="w")
        info_icon(
            tpl_box,
            "Pre-written prompt skeletons. The trigger word for THIS project "
            "is filled in for you, and your prompt body is appended after the "
            "template. Click 'Apply' to overwrite the body with the template.",
        ).grid(row=0, column=2, sticky="w")
        self.tpl_var = tk.StringVar(value=PROMPT_TEMPLATES[0][0])
        tpl_combo = ttk.Combobox(
            tpl_box,
            textvariable=self.tpl_var,
            values=[label for label, _, _ in PROMPT_TEMPLATES],
            state="readonly",
        )
        tpl_combo.grid(row=0, column=1, sticky="we", padx=PAD)
        self.tpl_hint_var = tk.StringVar(value=PROMPT_TEMPLATES[0][2])
        ttk.Label(
            tpl_box, textvariable=self.tpl_hint_var, style="Status.TLabel",
            wraplength=520, justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        tpl_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_template_change())
        ttk.Button(
            tpl_box, text="Apply template to prompt", style="Ghost.TButton",
            command=self._apply_template,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(PAD // 2, 0))

        # Prompt body
        prompt_box = ttk.LabelFrame(root, text="Prompt body", padding=PAD)
        prompt_box.grid(row=2, column=0, sticky="we", pady=(0, PAD))
        prompt_box.columnconfigure(0, weight=1)
        ttk.Label(
            prompt_box,
            text=(
                "What you'd type yourself. Quality stack is prepended; trigger "
                "word is auto-injected by the template. Edit freely."
            ),
            style="Status.TLabel", wraplength=520, justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.prompt_text = tk.Text(
            prompt_box, height=4, wrap="word",
            background=gui_theme.THEME.BG_INPUT,
            foreground=gui_theme.THEME.TEXT_PRIMARY,
            font=gui_theme.THEME.FONT_BODY, relief="flat", borderwidth=0,
            highlightthickness=1,
            highlightbackground=gui_theme.THEME.DIVIDER,
            highlightcolor=gui_theme.THEME.FOCUS_RING,
        )
        self.prompt_text.insert("1.0", self.body_var.get())
        self.prompt_text.grid(row=1, column=0, sticky="we")
        self.assembled_var = tk.StringVar(value="")
        ttk.Label(
            prompt_box, textvariable=self.assembled_var,
            style="Mono.TLabel", wraplength=520, justify="left",
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))

        # Negative prompt
        neg_box = ttk.LabelFrame(root, text="Negative prompt", padding=PAD)
        neg_box.grid(row=3, column=0, sticky="we", pady=(0, PAD))
        neg_box.columnconfigure(1, weight=1)
        ttk.Label(neg_box, text="Preset:").grid(row=0, column=0, sticky="w")
        info_icon(
            neg_box,
            "Pre-built negative prompts. 'NSFW · uncensor' actively pushes "
            "against censoring artefacts. You can layer your own additions "
            "into the entry below.",
        ).grid(row=0, column=2, sticky="w")
        self.neg_preset_var = tk.StringVar(value=NEGATIVE_PRESETS[1][0])
        neg_combo = ttk.Combobox(
            neg_box,
            textvariable=self.neg_preset_var,
            values=[label for label, _, _ in NEGATIVE_PRESETS],
            state="readonly",
        )
        neg_combo.grid(row=0, column=1, sticky="we", padx=PAD)
        neg_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_neg_preset_change())
        ttk.Entry(neg_box, textvariable=self.gui.negative_var).grid(
            row=1, column=0, columnspan=3, sticky="we", pady=(PAD // 2, 0),
        )
        # Seed initial negative.
        self.gui.negative_var.set(NEGATIVE_PRESETS[1][1])

        # Numeric / sampler row
        nums_box = ttk.LabelFrame(root, text="Sampler & dimensions", padding=PAD)
        nums_box.grid(row=4, column=0, sticky="we", pady=(0, PAD))
        nums_box.columnconfigure(0, weight=1)

        nums = ttk.Frame(nums_box)
        nums.grid(row=0, column=0, sticky="we")
        ttk.Label(nums, text="Sampler:").pack(side="left")
        info_icon(
            nums,
            "Diffusers scheduler. dpmpp_2m_karras and unipc converge fastest "
            "(good output at 20-25 steps); euler_a is the classic ancestral "
            "for varied output; default = whatever the pipeline picked.",
        ).pack(side="left")
        ttk.Combobox(
            nums, textvariable=self.gui.sampler_var,
            values=["default", "euler", "euler_a", "dpmpp_2m", "dpmpp_2m_karras", "unipc"],
            state="readonly", width=18,
        ).pack(side="left", padx=(PAD // 2, PAD * 2))

        ttk.Label(nums, text="Aspect:").pack(side="left")
        info_icon(
            nums,
            "SDXL is happiest at its training resolutions. Portrait 832×1216 "
            "for vertical shots, square 1024×1024 for centred subjects, "
            "landscape 1216×832 for wider compositions.",
        ).pack(side="left")
        ttk.Combobox(
            nums, textvariable=self.gui.aspect_var,
            values=[label for label, _w, _h in ASPECT_RATIOS],
            state="readonly", width=22,
        ).pack(side="left", padx=(PAD // 2, 0))

        nums2 = ttk.Frame(nums_box)
        nums2.grid(row=1, column=0, sticky="we", pady=(PAD // 2, 0))
        ttk.Label(nums2, text="N images:").pack(side="left")
        ttk.Entry(nums2, textvariable=self.gui.n_var, width=5).pack(
            side="left", padx=(PAD // 2, PAD),
        )
        ttk.Label(nums2, text="Steps:").pack(side="left")
        info_icon(
            nums2,
            "Denoising steps. With dpmpp_2m_karras / unipc, 20-28 is plenty. "
            "Bare euler benefits from 30-40. More than 40 is rarely worth the "
            "wall-clock.",
        ).pack(side="left")
        ttk.Entry(nums2, textvariable=self.gui.steps_var, width=5).pack(
            side="left", padx=(PAD // 2, PAD),
        )
        ttk.Label(nums2, text="Guidance:").pack(side="left")
        info_icon(
            nums2,
            "Classifier-free guidance scale. 5-7 for photoreal Pony work, "
            "7-9 for stylised Illustrious / anime. Higher = more literal "
            "prompt-following but more 'fried' look at extremes.",
        ).pack(side="left")
        ttk.Entry(nums2, textvariable=self.gui.guidance_var, width=5).pack(
            side="left", padx=(PAD // 2, PAD),
        )
        ttk.Label(nums2, text="Seed (blank=random):").pack(side="left")
        info_icon(
            nums2,
            "Fix a seed for reproducible output across prompt tweaks. Useful "
            "when iterating: lock the seed, change one word, see exactly what "
            "that word did.",
        ).pack(side="left")
        ttk.Entry(nums2, textvariable=self.gui.seed_var, width=12).pack(
            side="left", padx=(PAD // 2, 0),
        )

        # LoRA stack
        lora_box = ttk.LabelFrame(root, text="LoRA stack", padding=PAD)
        lora_box.grid(row=5, column=0, sticky="we", pady=(0, PAD))
        lora_box.columnconfigure(0, weight=1)
        self._build_lora_block(lora_box)

        # Action row
        btns = ttk.Frame(root)
        btns.grid(row=6, column=0, sticky="w", pady=(PAD, 0))
        ttk.Button(
            btns, text="Generate", style="Primary.TButton",
            command=self._on_generate,
        ).pack(side="left")
        ttk.Button(
            btns, text="Open outputs", style="Ghost.TButton",
            command=self._open_outputs,
        ).pack(side="left", padx=PAD)

        # Live preview of assembled prompt as the user types/picks.
        self.prompt_text.bind("<KeyRelease>", lambda _e: self._refresh_assembled())
        self.gui.quality_stack_var.trace_add("write", lambda *_: self._refresh_assembled())
        self._refresh_assembled()

    # ---- right column: NSFW guidance + civitai pointers ----

    def build_recommendations(self, root: ttk.Frame) -> None:
        PAD = gui_theme.PAD

        ttk.Label(
            root, text="Recommended setup",
            style="SubHeader.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, PAD // 2))

        bases_box = ttk.LabelFrame(root, text="Best NSFW base checkpoints", padding=PAD)
        bases_box.grid(row=1, column=0, sticky="we", pady=(0, PAD))
        bases_box.columnconfigure(0, weight=1)
        for r, (name, hint) in enumerate(BASE_RECOMMENDATIONS):
            ttk.Label(
                bases_box, text=f"• {name}",
                style="Mono.TLabel",
            ).grid(row=r * 2, column=0, sticky="w")
            ttk.Label(
                bases_box, text=hint,
                style="Status.TLabel", wraplength=320, justify="left",
            ).grid(row=r * 2 + 1, column=0, sticky="w", padx=(12, 0), pady=(0, 4))

        loras_box = ttk.LabelFrame(root, text="Civitai LoRAs to layer in", padding=PAD)
        loras_box.grid(row=2, column=0, sticky="we", pady=(0, PAD))
        loras_box.columnconfigure(0, weight=1)
        for r, (name, hint) in enumerate(LORA_RECOMMENDATIONS):
            ttk.Label(
                loras_box, text=f"• {name}",
                style="Mono.TLabel",
            ).grid(row=r * 2, column=0, sticky="w")
            ttk.Label(
                loras_box, text=hint,
                style="Status.TLabel", wraplength=320, justify="left",
            ).grid(row=r * 2 + 1, column=0, sticky="w", padx=(12, 0), pady=(0, 4))

        tips_box = ttk.LabelFrame(root, text="Quick tips", padding=PAD)
        tips_box.grid(row=3, column=0, sticky="we")
        tips_box.columnconfigure(0, weight=1)
        ttk.Label(
            tips_box,
            text=(
                "• Start with the trained LoRA at 1.0 + a style LoRA at 0.4-0.7. "
                "The trained LoRA carries likeness; the style LoRA carries vibe.\n"
                "• If the face drifts away from training, drop steps from 30 to "
                "22-25 — over-denoising can blur learned identity.\n"
                "• Lock the seed when iterating prompt wording so you can see "
                "exactly what each token contributes.\n"
                "• Generate at the SDXL aspect bucket nearest your final crop. "
                "Off-bucket sizes produce extra limbs / faces."
            ),
            style="Status.TLabel", wraplength=320, justify="left",
        ).grid(row=0, column=0, sticky="w")

    # ---- LoRA library ----

    def _build_lora_block(self, root: ttk.LabelFrame) -> None:
        PAD = gui_theme.PAD

        toggle_row = ttk.Frame(root)
        toggle_row.grid(row=0, column=0, sticky="we", pady=(0, PAD // 2))
        ttk.Checkbutton(
            toggle_row,
            text="Use this project's trained LoRA",
            variable=self.gui.use_trained_lora_var,
        ).pack(side="left")
        info_icon(
            toggle_row,
            "Tick to apply the LoRA you trained for THIS project. Untick for "
            "vanilla text-to-image with just the base checkpoint — useful to "
            "sanity-check the base or compare 'with vs without LoRA'. "
            "Either way, you can still stack extras below.",
        ).pack(side="left")

        lib_row = ttk.Frame(root)
        lib_row.grid(row=1, column=0, sticky="we", pady=(0, PAD // 2))
        ttk.Label(lib_row, text="Extra LoRAs:").pack(side="left")
        info_icon(
            lib_row,
            "Drop civitai .safetensors LoRAs into the shared library, refresh, "
            "tick to use, dial weight (1.0 = full strength). Usual mix: "
            "trained likeness LoRA at 1.0 + a style LoRA at 0.4-0.7 + "
            "optionally a detail/lighting LoRA at 0.3-0.5.",
        ).pack(side="left")
        ttk.Button(
            lib_row, text="Refresh", style="Ghost.TButton",
            command=self.refresh_lora_list,
        ).pack(side="right")
        ttk.Button(
            lib_row, text="Open library", style="Ghost.TButton",
            command=self.open_library,
        ).pack(side="right", padx=(0, PAD // 2))
        ttk.Button(
            lib_row, text="Import…", style="Ghost.TButton",
            command=self.import_safetensors,
        ).pack(side="right", padx=(0, PAD // 2))

        self.lora_table = ttk.Frame(root)
        self.lora_table.grid(row=2, column=0, sticky="we")
        self.refresh_lora_list()

    def refresh_lora_list(self) -> None:
        if self.lora_table is None:
            return
        for child in self.lora_table.winfo_children():
            child.destroy()

        files = gui_helpers.list_shared_loras(self.gui.projects_root.root)
        if not files:
            ttk.Label(
                self.lora_table,
                text=(
                    "No extras yet. Use 'Import…' or drop .safetensors files into "
                    f"{gui_helpers.shared_loras_dir(self.gui.projects_root.root)}."
                ),
                style="Status.TLabel", wraplength=520, justify="left",
            ).pack(anchor="w", pady=4)
            return

        new_rows: dict[str, dict] = {}
        for path in files:
            prior = self.lora_rows.get(path.stem, {})
            selected = prior.get("selected_var") or tk.BooleanVar(value=False)
            weight = prior.get("weight_var") or tk.StringVar(value="0.7")
            row = ttk.Frame(self.lora_table)
            row.pack(fill="x", pady=2)
            ttk.Checkbutton(row, variable=selected).pack(side="left", padx=(8, 8))
            ttk.Label(row, text=path.name).pack(side="left")
            ttk.Spinbox(
                row, from_=0.0, to=2.0, increment=0.05,
                textvariable=weight, width=6,
            ).pack(side="right", padx=(0, gui_theme.PAD))
            new_rows[path.stem] = {
                "path": path,
                "selected_var": selected,
                "weight_var": weight,
            }
        self.lora_rows = new_rows

    def open_library(self) -> None:
        d = gui_helpers.shared_loras_dir(self.gui.projects_root.root)
        gui_helpers.open_folder(d)

    def import_safetensors(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Pick one or more .safetensors LoRAs to import",
            filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")],
        )
        if not paths:
            return
        dest = gui_helpers.shared_loras_dir(self.gui.projects_root.root)
        copied = 0
        for src in paths:
            srcp = Path(src)
            try:
                shutil.copy2(srcp, dest / srcp.name)
                copied += 1
            except Exception as e:
                messagebox.showerror("Import failed", f"{srcp.name}: {e}")
        if copied:
            self.gui.log_queue.put(f"[Imported {copied} LoRA(s) into {dest}]\n")
            self.refresh_lora_list()

    def selected_extras(self) -> list[tuple[Path, float]]:
        out: list[tuple[Path, float]] = []
        for stem, row in self.lora_rows.items():
            if not row["selected_var"].get():
                continue
            try:
                w = float(row["weight_var"].get())
            except ValueError:
                w = 1.0
            out.append((row["path"], w))
        return out

    # ---- preset handlers ----

    def _on_quality_stack_change(self) -> None:
        label = self.gui.quality_stack_var.get()
        for name, _prefix, hint in QUALITY_STACKS:
            if name == label:
                self.qs_hint_var.set(hint)
                break
        self._refresh_assembled()

    def _on_template_change(self) -> None:
        label = self.tpl_var.get()
        for name, _tmpl, hint in PROMPT_TEMPLATES:
            if name == label:
                self.tpl_hint_var.set(hint)
                break

    def _on_neg_preset_change(self) -> None:
        label = self.neg_preset_var.get()
        for name, body, _hint in NEGATIVE_PRESETS:
            if name == label:
                self.gui.negative_var.set(body)
                break

    def _apply_template(self) -> None:
        if not self.gui.current_project:
            messagebox.showerror("No project", "Open a project first.")
            return
        label = self.tpl_var.get()
        template = ""
        for name, tmpl, _hint in PROMPT_TEMPLATES:
            if name == label:
                template = tmpl
                break
        if not template:
            return
        body = self.prompt_text.get("1.0", "end").strip()
        trigger = self.gui.current_project.trigger_word
        new_body = template.format(trigger=trigger, body=body).strip(", \n ")
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", new_body)
        self._refresh_assembled()

    def _refresh_assembled(self) -> None:
        body = self.prompt_text.get("1.0", "end").strip()
        prefix = ""
        for name, p, _hint in QUALITY_STACKS:
            if name == self.gui.quality_stack_var.get():
                prefix = p
                break
        full = f"{prefix}{body}".strip()
        if not full:
            self.assembled_var.set("(empty prompt)")
        else:
            display = full if len(full) <= 240 else full[:237] + "…"
            self.assembled_var.set(f"→ sending: {display}")

    # ---- generate ----

    def _on_generate(self) -> None:
        project = self.gui.require_project()
        if not project:
            return
        self.gui.save_settings_silent()

        body = self.prompt_text.get("1.0", "end").strip()
        prefix = ""
        for name, p, _hint in QUALITY_STACKS:
            if name == self.gui.quality_stack_var.get():
                prefix = p
                break
        prompt = f"{prefix}{body}".strip()
        if not prompt:
            messagebox.showerror("Empty prompt", "Type a prompt first.")
            return

        # Resolve aspect ratio.
        width, height = 1024, 1024
        for label, w, h in ASPECT_RATIOS:
            if label == self.gui.aspect_var.get():
                width, height = w, h
                break

        args = [
            "generate",
            str(project.root),
            "--prompt", prompt,
            "--n", self.gui.n_var.get() or "4",
            "--steps", self.gui.steps_var.get() or "28",
            "--guidance", self.gui.guidance_var.get() or "6.5",
            "--width", str(width),
            "--height", str(height),
            "--sampler", self.gui.sampler_var.get() or "default",
        ]
        neg = self.gui.negative_var.get().strip()
        if neg:
            args += ["--negative", neg]
        seed = self.gui.seed_var.get().strip()
        if seed:
            args += ["--seed", seed]
        if not self.gui.use_trained_lora_var.get():
            args.append("--no-trained-lora")
        extras = self.selected_extras()
        for path, weight in extras:
            args += ["--extra-lora", f"{path}:{weight}"]

        if not self.gui.use_trained_lora_var.get() and not extras:
            ok = messagebox.askokcancel(
                "Render with base only",
                "No trained LoRA + no extra LoRAs selected. This will be a "
                "vanilla base render. Proceed?",
            )
            if not ok:
                return
        self.gui.spawn(args)

    def _open_outputs(self) -> None:
        if not self.gui.current_project:
            return
        gui_helpers.open_folder(self.gui.current_project.outputs_dir)
