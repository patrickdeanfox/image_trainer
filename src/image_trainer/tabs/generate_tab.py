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
from ..gui_widgets import CollapsibleFrame, ScrollableFrame, info_icon

if TYPE_CHECKING:
    from ..gui_app import TrainerGUI


# ---------- Preset content ----------
#
# These are the actual recipes the tab proposes. Kept as data tables at the
# top of the file so they're easy to tune without touching layout code.

#: Quality-tag stacks. Prepended to the user's prompt when picked.
#: Empty string = no prefix (default — your prompt stands alone).
#:
#: IMPORTANT: SDXL CLIP encoders cap at 77 tokens per chunk. Without compel
#: installed, anything past ~77 tokens gets silently dropped. Stacks below
#: are kept short (5-10 tokens) so the user has plenty of budget for actual
#: prompt content. Stylistic anchors that USED to live here (e.g.
#: "professional nude photography, detailed anatomy") have moved into the
#: PROMPT_TEMPLATES so they're optional + visible.
QUALITY_STACKS: list[tuple[str, str, str]] = [
    (
        "(none)", "",
        "No quality prefix — your prompt stands alone. Pick this when you "
        "want full control over the prompt text.",
    ),
    (
        "Pony · photoreal NSFW (recommended)",
        "score_9, score_8_up, score_7_up, source_real, rating_explicit, ",
        "Canonical Pony photoreal NSFW opener — short, ~10 tokens. "
        "score_X tags are mandatory for Pony quality; rating_explicit "
        "unlocks the explicit content range; source_real biases toward "
        "photographic output. Best default for personal-likeness LoRAs "
        "trained on photo source material.",
    ),
    (
        "Pony · photoreal SFW",
        "score_9, score_8_up, score_7_up, source_real, rating_safe, ",
        "Same Pony short stack but with rating_safe instead of "
        "rating_explicit. Use for SFW portrait / lifestyle work.",
    ),
    (
        "Pony · explicit (heavy)",
        "score_9, score_8_up, score_7_up, score_6_up, rating_explicit, "
        "rating_questionable, source_real, ",
        "Heavier Pony NSFW opener — ~13 tokens. Adds score_6_up + "
        "rating_questionable to widen the explicit range. Use when the "
        "base is producing tasteful / implied output and you want it to "
        "lean more explicit. Costs more token budget.",
    ),
    (
        "Pony · anime / illustrated",
        "score_9, score_8_up, score_7_up, source_anime, rating_explicit, ",
        "Pony in anime/hentai mode. source_anime pulls toward Pony's "
        "illustrated-art training distribution. Combine with anime-style "
        "LoRAs from the library.",
    ),
    (
        "Illustrious / NoobAI XL",
        "masterpiece, best quality, very aware, newest, absurdres, ",
        "Tag stack for Illustrious-XL / NoobAI-XL family checkpoints. "
        "Different vocabulary from Pony — masterpiece + best quality + "
        "very aware (NoobAI's anatomy anchor) + newest (recency bias).",
    ),
    (
        "RealVis / Juggernaut · photoreal",
        "raw photo, dslr, photorealistic, ",
        "Short realism-tilted opener for RealVisXL / JuggernautXL / "
        "vanilla SDXL fine-tunes that don't use score_X tags. Pair with "
        "the NSFW uncensor negative preset for explicit work.",
    ),
    (
        "Lustify / Pony Realism",
        "score_9, score_8_up, score_7_up, source_real, rating_explicit, "
        "professional photography, ",
        "Tuned for Lustify XL and Pony Realism. Pony score stack + "
        "professional-photography anchor for the magazine look these "
        "fine-tunes were heavily trained on.",
    ),
]


#: Prompt templates. The {trigger} token is substituted with the project's
#: trigger word so the trained LoRA is summoned. {body} marks where any
#: existing user prompt lands when the template is applied as a wrapper.
#:
#: These are written in the Pony Diffusion vocabulary because Pony is the
#: recommended NSFW base. Most tags transfer cleanly to other Pony-derived
#: fine-tunes (Lustify, Pony Realism). Use the QUALITY_STACKS picker to add
#: the score_X / rating_X anchors — those are kept separate so you can swap
#: bases without rewriting templates.
PROMPT_TEMPLATES: list[tuple[str, str, str]] = [
    (
        "Full-body nude · blonde (default)",
        "{trigger}, 1girl, beautiful blonde woman, long wavy blonde hair, "
        "blue eyes, natural makeup, fully nude, full body visible, "
        "standing, contrapposto pose, arms relaxed at sides, looking at "
        "viewer, face visible, breasts visible, pubic area visible, "
        "thighs visible, legs visible, anatomically correct, natural "
        "body proportions, detailed anatomy, soft natural lighting, "
        "neutral background, raw photo, professional nude photography, "
        "85mm lens, shallow depth of field, {body}",
        "Full-body fully-nude blonde — head-to-toe visible, anatomy "
        "anchors that suppress Pony's tendency to crop or 'tasteful drape'. "
        "Default template for the NSFW workflow. Pair with Pony · "
        "photoreal NSFW quality stack and the NSFW · uncensor negative "
        "preset; replace 'blonde' with your subject's hair colour for the "
        "trained-LoRA case.",
    ),
    (
        "Portrait · headshot test",
        "{trigger}, beautiful woman, head and shoulders portrait, "
        "looking at viewer, soft natural window light, shallow depth of "
        "field, sharp focus on eyes, detailed skin texture, freckles, "
        "natural makeup, intimate framing, candid expression, "
        "professional photography, 85mm lens, {body}",
        "Likeness sanity check. Tight headshot, neutral lighting, no body. "
        "If the face doesn't look like your training subject here, your "
        "LoRA either needs more training or your prompt is fighting the "
        "model. Best diagnostic template.",
    ),
    (
        "Boudoir · bedroom intimate",
        "{trigger}, beautiful woman, lying on bed, lacy lingerie, "
        "bedroom interior, warm bedside lamp light, tousled white sheets, "
        "looking at viewer, parted lips, suggestive pose, one hand "
        "behind head, soft shadows, shallow depth of field, "
        "intimate atmosphere, {body}",
        "Classic boudoir setup. Implicit / teasing, not explicit. Pair "
        "with the Pony photoreal SFW stack for tasteful output, or "
        "photoreal NSFW stack to push it further. The lighting + sheet "
        "details give Pony specific anchors that improve composition.",
    ),
    (
        "Nude · standing studio",
        "{trigger}, beautiful nude woman, full body, standing pose, hand "
        "on hip, slight contrapposto, soft studio lighting, neutral grey "
        "seamless background, natural skin texture, subsurface scattering, "
        "detailed anatomy, anatomically correct breasts, looking at viewer, "
        "professional nude photography, {body}",
        "Studio nude. The 'anatomically correct' anchor is the Pony "
        "vocabulary that most reliably suppresses extra limbs / deformed "
        "body parts. Pair with Pony · photoreal NSFW stack and an anatomy "
        "correction LoRA at 0.5.",
    ),
    (
        "Nude · explicit close-up",
        "{trigger}, nude, explicit, looking at viewer, intimate framing, "
        "detailed anatomy, anatomically correct, perfect breasts, "
        "natural body, pubic hair, soft natural light, shallow depth of "
        "field, raw photo, professional erotic photography, "
        "high detail skin, {body}",
        "Heaviest explicit framing. Use with Pony · explicit NSFW (heavy) "
        "stack + NSFW · uncensor negative preset. The 'natural body' + "
        "'pubic hair' anchors prevent the excessive idealisation Pony "
        "drifts toward without them.",
    ),
    (
        "POV · first-person intimate",
        "{trigger}, pov, first person view, looking up at viewer, "
        "lying down, nude, soft warm light, intimate close-up, "
        "detailed face, parted lips, eye contact, raw photo, "
        "amateur photograph aesthetic, {body}",
        "POV framing — first-person camera angle, subject lying back. "
        "Hardest composition for Pony to get right; works best when the "
        "training data included POV shots. Add 'amateur' to anchor away "
        "from glossy magazine output.",
    ),
    (
        "Selfie · phone mirror",
        "{trigger}, selfie, holding smartphone, mirror selfie, bathroom "
        "mirror or bedroom mirror, casual home interior, natural pose, "
        "looking at phone camera, modern outfit or lingerie, "
        "instagram aesthetic, iphone photo, slight motion blur, {body}",
        "Mimics the OnlyFans / Instagram source distribution most LoRA "
        "training data is drawn from. Closest to the LoRA's training set, "
        "so produces the most consistent likeness. Include 'iphone photo' "
        "to anchor the camera characteristics.",
    ),
    (
        "Outdoor · golden hour SFW",
        "{trigger}, beautiful woman, outdoor portrait, golden hour, "
        "warm low sun, lens flare, summer dress, casual pose, candid "
        "moment, blurred natural background, depth of field, "
        "professional outdoor photography, {body}",
        "SFW outdoor framing. Useful as a likeness generalisation test — "
        "if the face stays consistent away from bedroom lighting, your "
        "LoRA generalises well. Otherwise, the LoRA may have memorised "
        "indoor lighting along with the face.",
    ),
    (
        "Lingerie · magazine glamour",
        "{trigger}, beautiful woman, professional lingerie photoshoot, "
        "lacy black lingerie, garter belt, stockings, dramatic studio "
        "lighting, key light + rim light, dark background, confident "
        "pose, detailed fabric texture, magazine cover quality, "
        "high fashion photography, {body}",
        "Polished editorial lingerie. Drama + texture anchors give Pony "
        "specific things to render. Pair with Lustify / Pony Realism "
        "stack + a 'glamour' style LoRA at 0.4-0.6.",
    ),
    (
        "Bath · steam + water",
        "{trigger}, beautiful woman, bathing, bathtub, warm bath water, "
        "wet skin, water droplets, steam, soft window light, "
        "intimate atmosphere, looking at viewer, hand on edge of tub, "
        "shallow depth of field, professional photography, {body}",
        "Implicit / steam-obscured nude. Water + steam tags add visual "
        "interest and let the model produce explicit content with "
        "natural censoring built in. Good for early LoRA tests.",
    ),
    (
        "Shower · wet hair",
        "{trigger}, beautiful nude woman, in shower, wet hair, water "
        "running down skin, water droplets, glass shower door, soft "
        "diffused light, looking at viewer, intimate framing, "
        "professional photography, photorealistic, {body}",
        "Shower scene — produces glossy wet-skin output Pony handles "
        "well. Glass door anchor adds depth without ControlNet.",
    ),
    (
        "Cowgirl / on-top pose",
        "{trigger}, nude, cowgirl position, straddling, hands on chest, "
        "looking down at viewer, intimate POV, soft warm bedroom light, "
        "tousled sheets, detailed anatomy, anatomically correct, "
        "raw photo, amateur photograph aesthetic, {body}",
        "Explicit on-top pose. POV framing — works best on Pony explicit "
        "NSFW stack. Combine with a pose LoRA from civitai if the model "
        "drifts.",
    ),
    (
        "Doggystyle / from behind",
        "{trigger}, nude, from behind, on hands and knees, ass focus, "
        "looking back at viewer, intimate angle, soft natural light, "
        "detailed anatomy, anatomically correct, raw photo, amateur "
        "photograph aesthetic, {body}",
        "Explicit from-behind framing. 'looking back at viewer' is the "
        "Pony tag that gets the head-turned eye-contact composition.",
    ),
    (
        "Cosplay · costume + character",
        "{trigger}, cosplay, dramatic pose, full costume, studio "
        "lighting, dynamic composition, detailed costume fabric, "
        "professional cosplay photography, {body}",
        "Skeleton for cosplay. Add the specific costume / character "
        "vocabulary in the body or stack with a character / outfit LoRA "
        "from civitai (e.g. 'sailor uniform LoRA').",
    ),
    (
        "Group · with another person",
        "{trigger} on the left, beautiful woman, intimate scene, "
        "interacting with another person, soft warm light, detailed "
        "anatomy, two people, raw photo, amateur photograph aesthetic, "
        "{body}",
        "Two-person composition. Hard for any base to get right — the "
        "trained likeness LoRA will only fire on the explicitly-tagged "
        "subject ('on the left'). Expect to retry; lock seed when one "
        "works and iterate from there.",
    ),
]


#: Negative-prompt presets. Pick one — or extend it in the entry below it.
#:
#: A good NSFW negative covers FOUR things at once:
#:   1. Quality / artefact suppression  (low quality, jpeg artefacts, etc.)
#:   2. Anatomy correction              (extra limbs, mutated hands, …)
#:   3. Censoring removal               (mosaic, bar censor, blurred, …)
#:   4. Style steering                  (anime if you want photo, vice versa)
#: The presets below combine these in different proportions for different
#: workflows.
NEGATIVE_PRESETS: list[tuple[str, str, str]] = [
    (
        "(none)", "",
        "No negative prompt. Pony at the right CFG can work without one, "
        "but you'll usually get better results with at least the standard "
        "quality preset.",
    ),
    (
        "Standard quality (safe baseline)",
        "score_4, score_5, score_6, low quality, worst quality, lowres, "
        "blurry, out of focus, jpeg artifacts, compression artifacts, "
        "watermark, signature, text, logo, username, error, "
        "bad anatomy, bad proportions, deformed, mutated, extra limbs, "
        "extra fingers, fused fingers, malformed hands, missing fingers, "
        "extra arms, extra legs, disfigured, ugly",
        "Generic quality + anatomy correction. Includes Pony's "
        "score_4/5/6 negative anchors (they suppress the low-quality "
        "training distribution). Safe baseline regardless of base.",
    ),
    (
        "NSFW · uncensor (recommended for explicit)",
        "score_4, score_5, score_6, censored, uncensored, mosaic, mosaic "
        "censoring, bar censor, black bar, pixelated, pixelization, "
        "blurred genitals, novelty censor, convenient censoring, "
        "covering breasts, covering crotch, hand over breasts, "
        "clothing covering body, low quality, worst quality, blurry, "
        "watermark, signature, text, jpeg artifacts, "
        "bad anatomy, deformed, mutated, extra limbs, extra fingers, "
        "fused fingers, malformed hands, missing fingers",
        "Comprehensive NSFW negative — actively pushes against the "
        "censoring artefacts (mosaic, black bars, 'novelty censor', "
        "'convenient censoring' tags Pony was trained on) AND the body-"
        "covering poses the model defaults to when uncertain. Pair with "
        "Pony explicit NSFW stack for the best uncensored output.",
    ),
    (
        "NSFW · photoreal push",
        "score_4, score_5, score_6, anime, manga, illustration, drawing, "
        "painting, cartoon, sketch, lineart, 3d render, cgi, doll, "
        "plastic skin, smooth skin, airbrushed, oversaturated, "
        "low quality, worst quality, blurry, watermark, signature, "
        "bad anatomy, deformed, extra limbs, extra fingers, "
        "fused fingers, malformed hands, censored, mosaic, bar censor",
        "Realism push — suppresses every stylised aesthetic so output "
        "stays photographic. Use with RealVisXL / JuggernautXL / Lustify "
        "or Pony in photo mode. The 'plastic skin' / 'smooth skin' "
        "negatives are key for breaking out of the Pony default skin "
        "look.",
    ),
    (
        "Anime / illustrated push",
        "photo, photograph, photorealistic, photo realistic, realistic, "
        "3d, 3d render, real life, real person, dslr, raw photo, "
        "low quality, worst quality, blurry, jpeg artifacts, "
        "watermark, signature, bad anatomy, deformed, extra limbs, "
        "extra fingers, fused fingers, malformed hands, "
        "censored, mosaic, bar censor",
        "Inverse of the photoreal push — suppresses every photographic "
        "anchor so the model produces drawn / illustrated output. Use "
        "with Pony anime mode or Illustrious / NoobAI.",
    ),
    (
        "Body-detail correction (pair with anything)",
        "score_4, score_5, score_6, bad anatomy, bad proportions, "
        "deformed body, mutated, extra limbs, extra arms, extra legs, "
        "extra hands, extra feet, extra fingers, fused fingers, "
        "malformed hands, missing fingers, missing limbs, asymmetrical "
        "eyes, cross-eyed, deformed face, mutated face, ugly face, "
        "fat rolls, unnatural body shape, weird torso, broken neck, "
        "long neck, short legs, short arms, deformed breasts, "
        "asymmetrical breasts, low quality, blurry, watermark",
        "Aggressive anatomy fix. Use when basic negatives aren't enough "
        "and you keep seeing extra limbs / weird hands. Combine with an "
        "anatomy-correction LoRA from civitai for best results.",
    ),
    (
        "Soft sensual (artistic, not explicit)",
        "score_4, score_5, score_6, hardcore, explicit, graphic, "
        "penetration, cum, ejaculation, semen, low quality, worst "
        "quality, blurry, watermark, signature, bad anatomy, deformed, "
        "extra limbs, extra fingers, fused fingers, malformed hands",
        "For sensual / artistic nude work that should NOT be explicit. "
        "Suppresses the explicit-content vocabulary while keeping the "
        "quality + anatomy negatives. Use with rating_safe / "
        "rating_questionable on the prompt side.",
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
    # Pin the tab content horizontally; the row weight for the scrollable
    # region is set further down once we know which row holds the scroll
    # canvas (after the sticky action bar above it).
    f.columnconfigure(0, weight=1)

    gui.prompt_var = tk.StringVar(value="")
    gui.negative_var = tk.StringVar(value="")
    gui.n_var = tk.StringVar(value="4")
    gui.steps_var = tk.StringVar(value="28")
    gui.guidance_var = tk.StringVar(value="6.5")
    gui.seed_var = tk.StringVar(value="")
    # Trained LoRA defaults to OFF — most users in NSFW workflows hit
    # Generate first against the bare base to validate the prompt + sampler
    # combo before committing to a real training run. Tick the checkbox in
    # the LoRA stack section once you have a trained LoRA in lora/.
    gui.use_trained_lora_var = tk.BooleanVar(value=False)
    gui.sampler_var = tk.StringVar(value="dpmpp_2m_karras")
    # Portrait 896×1152 is the SDXL bucket closest to "full body fits in
    # frame, head not cropped." 832×1216 is taller still, often cuts feet.
    gui.aspect_var = tk.StringVar(value="Portrait 896×1152")
    # Default quality stack is the recommended Pony NSFW opener — matches
    # the default base most users picked (Pony Diffusion V6 XL).
    gui.quality_stack_var = tk.StringVar(value="Pony · photoreal NSFW (recommended)")
    gui.output_name_var = tk.StringVar(value="")

    state = _GenerateState(gui)
    gui.generate_state = state

    # Header row.
    ttk.Label(
        f, text="Generate · prompt → images",
        style="Header.TLabel",
    ).grid(row=0, column=0, sticky="w", pady=(0, PAD // 2))

    # Sticky action bar — Generate button + progress feedback live ABOVE
    # the scrollable region so they're always visible. Without this, a
    # user scrolled to the top of the form (typing the prompt) would lose
    # sight of the click target + the spinner the moment they hit
    # Generate, and have to scroll back down to see anything happening.
    action_bar = ttk.Frame(f)
    action_bar.grid(row=1, column=0, sticky="we", pady=(0, PAD))
    action_bar.columnconfigure(2, weight=1)
    state.generate_btn = ttk.Button(
        action_bar, text="Generate", style="Primary.TButton",
        command=state._on_generate,
    )
    state.generate_btn.grid(row=0, column=0, sticky="w")
    ttk.Button(
        action_bar, text="Open outputs", style="Ghost.TButton",
        command=state._open_outputs,
    ).grid(row=0, column=1, sticky="w", padx=(PAD, 0))
    state.progress = ttk.Progressbar(
        action_bar, mode="determinate",
        style="Trainer.Horizontal.TProgressbar",
    )
    state.progress.grid(row=0, column=2, sticky="we", padx=(PAD, 0))
    state.progress_status_var = tk.StringVar(value="idle")
    ttk.Label(
        f, textvariable=state.progress_status_var, style="Status.TLabel",
    ).grid(row=2, column=0, sticky="w", pady=(0, PAD // 2))

    # The form + recommendations sidebar lives in a ScrollableFrame because
    # the Generate tab carries enough vertical content (prompt block + LoRA
    # stack + recommendations sidebar) to overflow on a 900-tall window.
    scroll = ScrollableFrame(f)
    scroll.grid(row=3, column=0, sticky="nswe")
    f.rowconfigure(3, weight=1)  # scroll region claims leftover height

    main = ttk.Frame(scroll.body)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=2)
    main.columnconfigure(1, weight=1)

    form = ttk.Frame(main)
    form.grid(row=0, column=0, sticky="nsew", padx=(0, PAD))
    form.columnconfigure(0, weight=1)

    rec_outer = CollapsibleFrame(main, text="Recommended setup ▾", start_open=True)
    rec_outer.grid(row=0, column=1, sticky="nsew")
    state.build_form(form)
    state.build_recommendations(rec_outer.body)


class _GenerateState:
    """Holds the LoRA-stack rows + handlers for the Generate tab."""

    def __init__(self, gui: "TrainerGUI") -> None:
        self.gui = gui
        self.lora_rows: dict[str, dict] = {}
        self.lora_table: tk.Widget | None = None
        # The actual prompt body the user types (separated from any preset
        # template wrapping or quality-stack prefix).
        # Pre-populated NSFW body — a standalone full-body nude blonde
        # prompt that works with the Pony · photoreal NSFW quality stack
        # even without a trained LoRA. Users edit this freely; the
        # Prompt template dropdown can reset it back to a skeleton with
        # the project's trigger word substituted.
        self.body_var = tk.StringVar(
            value=(
                "1girl, beautiful blonde woman, long wavy blonde hair, "
                "blue eyes, natural makeup, fully nude, full body visible, "
                "standing, contrapposto pose, arms relaxed at sides, "
                "looking at viewer, face visible, breasts visible, "
                "pubic area visible, thighs visible, legs visible, "
                "anatomically correct, natural body proportions, "
                "detailed anatomy, soft natural lighting, neutral "
                "background, raw photo, professional nude photography, "
                "85mm lens, shallow depth of field"
            )
        )
        # Run tracking so on_progress_line knows what to render.
        self.run_active: bool = False
        self.images_total: int = 0
        self.images_done: int = 0

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
        self.neg_preset_var = tk.StringVar(value=NEGATIVE_PRESETS[2][0])
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
        self.gui.negative_var.set(NEGATIVE_PRESETS[2][1])

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
            side="left", padx=(PAD // 2, PAD * 2),
        )

        ttk.Label(nums2, text="Folder name (optional):").pack(side="left")
        info_icon(
            nums2,
            "Optional friendly name for this run's output folder. Becomes "
            "outputs/<name>_<timestamp>/ — useful for grouping experiments "
            "('boudoir_test_v3', 'sampler_compare', etc.). Leave blank for "
            "the default outputs/<timestamp>/ format.",
        ).pack(side="left")
        ttk.Entry(nums2, textvariable=self.gui.output_name_var, width=22).pack(
            side="left", padx=(PAD // 2, 0),
        )

        # LoRA stack
        lora_box = ttk.LabelFrame(root, text="LoRA stack", padding=PAD)
        lora_box.grid(row=5, column=0, sticky="we", pady=(0, PAD))
        lora_box.columnconfigure(0, weight=1)
        self._build_lora_block(lora_box)

        # Action row
        # NOTE: Generate button + progress bar live in the sticky action
        # bar above the scrollable region (see ``build()``). Don't add
        # them here — duplicating them would confuse users + break the
        # progress-tracking state that points at the action-bar widgets.

        # Live preview of assembled prompt as the user types/picks.
        self.prompt_text.bind("<KeyRelease>", lambda _e: self._refresh_assembled())
        self.gui.quality_stack_var.trace_add("write", lambda *_: self._refresh_assembled())
        self._refresh_assembled()

    # ---- right column: NSFW guidance + civitai pointers ----

    def build_recommendations(self, root: ttk.Frame) -> None:
        # Header is supplied by the surrounding CollapsibleFrame so the
        # whole sidebar can be collapsed when the user wants more room
        # for the form. Just stack the three reference boxes here.
        PAD = gui_theme.PAD
        root.columnconfigure(0, weight=1)

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
            messagebox.showerror(
                "Prompt is empty",
                "Type a description in the Prompt body field, or click "
                "'Apply template to prompt' under Prompt template to start "
                "from one of the pre-written skeletons.",
            )
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
        out_name = self.gui.output_name_var.get().strip()
        if out_name:
            args += ["--output-name", out_name]
        if not self.gui.use_trained_lora_var.get():
            args.append("--no-trained-lora")
        extras = self.selected_extras()
        for path, weight in extras:
            args += ["--extra-lora", f"{path}:{weight}"]

        if not self.gui.use_trained_lora_var.get() and not extras:
            ok = messagebox.askokcancel(
                "Render with base model only?",
                "Both 'Use this project's trained LoRA' and the extra-LoRA "
                "list are off, so the output will be a plain base-checkpoint "
                "render — your trained subject won't appear unless the prompt "
                "names them by description.\n\n"
                "Useful for sanity-checking the base or comparing 'with vs "
                "without LoRA'. Continue with base only?",
            )
            if not ok:
                return

        # Reset progress UI + arrange for the spawn-finished hook to clean up.
        try:
            n_total = max(1, int(self.gui.n_var.get() or "4"))
        except ValueError:
            n_total = 4
        self._begin_run(n_total)
        self.gui.on_next_exit = self._end_run
        self.gui.spawn(args)

    def _open_outputs(self) -> None:
        if not self.gui.current_project:
            return
        gui_helpers.open_folder(self.gui.current_project.outputs_dir)

    # ---- live progress hooks driven by gui_app._drain_log ----

    def _begin_run(self, total: int) -> None:
        """Switch the tab into 'generating' visual state."""
        self.run_active = True
        self.images_total = total
        self.images_done = 0
        # While the pipeline loads (model load + LoRA injection +
        # scheduler swap) we don't have per-image work yet, so spin
        # indeterminate. We flip to determinate once the first image
        # lands.
        self.progress.configure(mode="indeterminate")
        try:
            self.progress.start(80)
        except Exception:
            pass
        self.progress_status_var.set(
            f"Loading pipeline (target {total} image{'s' if total != 1 else ''})…"
        )
        try:
            self.generate_btn.state(["disabled"])
            self.generate_btn.configure(text="Generating…")
        except Exception:
            pass

    def _end_run(self) -> None:
        """Called when the trainer subprocess exits — restore idle state."""
        self.run_active = False
        try:
            self.progress.stop()
        except Exception:
            pass
        self.progress.configure(mode="determinate", maximum=max(1, self.images_total))
        self.progress["value"] = self.images_total if self.images_done >= self.images_total else self.images_done
        self.progress_status_var.set(
            f"done · {self.images_done}/{self.images_total} image"
            f"{'s' if self.images_total != 1 else ''} saved"
            if self.images_done > 0
            else "exited (no images saved — see Telemetry pane for errors)"
        )
        try:
            self.generate_btn.state(["!disabled"])
            self.generate_btn.configure(text="Generate")
        except Exception:
            pass

    def on_progress_line(self, line: str) -> None:
        """Called from gui_app._drain_log for every log line. Cheap; ignores
        anything we don't recognise."""
        if not self.run_active:
            return
        s = line.strip()
        if not s:
            return
        # generate.py emits these markers — match conservatively.
        if s.startswith("Saved "):
            self.images_done += 1
            # Flip from indeterminate to determinate the first time we see
            # an image land. The bar then advances for every subsequent
            # image saved.
            try:
                self.progress.stop()
            except Exception:
                pass
            self.progress.configure(mode="determinate", maximum=max(1, self.images_total))
            self.progress["value"] = self.images_done
            self.progress_status_var.set(
                f"image {self.images_done}/{self.images_total} saved"
            )
            return
        if s.startswith("Loaded trained LoRA"):
            self.progress_status_var.set("loaded trained LoRA")
            return
        if s.startswith("Loaded extra LoRA"):
            self.progress_status_var.set(s.lower())
            return
        if s.startswith("Sampler set to"):
            self.progress_status_var.set(s.lower())
            return
        if s.startswith("Base-model render"):
            self.progress_status_var.set("base-model render — no LoRAs")
            return
