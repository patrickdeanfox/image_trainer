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
from ..prompt_presets import QUALITY_STACKS

if TYPE_CHECKING:
    from ..gui_app import TrainerGUI


# ---------- Preset content ----------
#
# These are the actual recipes the tab proposes. Kept as data tables at the
# top of the file so they're easy to tune without touching layout code.

#: Quality-tag stacks now live in :mod:`image_trainer.prompt_presets` so
#: the CLI's --compare-stacks loop can iterate the same list. Imported at
#: top of file. The remaining preset tables (NEGATIVE_PRESETS, ASPECT_RATIOS,
#: BUILDER_*) stay GUI-side — they're consumed only by this tab.
# (QUALITY_STACKS moved to image_trainer.prompt_presets — imported above.)


#: Legacy one-click template picker — retired in favour of the live
#: Subject / Scene / Action builder. Kept as an empty stub so the symbol
#: exists for any historical reference; wording for the 18 original
#: templates lives in git history. Nothing in the live code reads this.
_LEGACY_PROMPT_TEMPLATES: list[tuple[str, str, str]] = []


#: Negative-prompt presets. Pick one — or extend it in the entry below it.
#:
#: A good NSFW negative covers FOUR things at once:
#:   1. Quality / artefact suppression  (low quality, jpeg artefacts, etc.)
#:   2. Anatomy correction              (extra limbs, mutated hands, …)
#:   3. Censoring removal               (mosaic, bar censor, blurred, …)
#:   4. Style steering                  (anime if you want photo, vice versa)
#: The presets below combine these in different proportions for different
#: workflows.

#: The "photo-only" baseline — tokens that suppress every non-photographic
#: aesthetic. Per user directive, every preset (except literal "(none)")
#: must contain these tokens so the model defaults to photographic output.
#: Compel handles duplicates gracefully, so prepending this to presets
#: that already contain some of these tokens is cheap and correct.
_PHOTO_ONLY_BASELINE = (
    "anime, manga, illustration, drawing, painting, cartoon, sketch, "
    "lineart, 3d render, cgi, render, animation, animations, artwork, "
    "digital art, concept art, anime screenshot, hentai art, "
    "stylized, cel shading, doll"
)


def _with_photo_baseline(body: str) -> str:
    """Prefix the photo-only baseline to a preset body. Skips the empty
    string so the literal "(none)" preset stays empty. Otherwise returns
    "<baseline>, <body>".
    """
    if not body:
        return body
    return f"{_PHOTO_ONLY_BASELINE}, {body}"


NEGATIVE_PRESETS: list[tuple[str, str, str]] = [
    (
        "(none)", "",
        "No negative prompt. The literal 'no negative' option — every "
        "OTHER preset starts with the anti-illustration / anti-anime / "
        "anti-CGI baseline so output stays photographic. Pick this only "
        "when you want to see what Pony does with truly nothing in the "
        "negative slot (rarely useful but occasionally diagnostic).",
    ),
    (
        "Pony · minimal (score-only, recommended)",
        _with_photo_baseline("score_4, score_5, score_6"),
        "The smallest effective Pony negative — score buckets + the "
        "photo-only baseline (anti-illustration / anti-anime / "
        "anti-CGI). Default for photoreal NSFW on Pony.",
    ),
    (
        "Standard quality (safe baseline)",
        _with_photo_baseline(
            "score_4, score_5, score_6, low quality, worst quality, lowres, "
            "blurry, out of focus, jpeg artifacts, compression artifacts, "
            "watermark, signature, text, logo, username, error, "
            "bad anatomy, bad proportions, deformed, mutated, extra limbs, "
            "extra fingers, fused fingers, malformed hands, missing fingers, "
            "extra arms, extra legs, disfigured, ugly"
        ),
        "Generic quality + anatomy correction, plus the photo-only "
        "baseline. Pony's score_4/5/6 negative anchors suppress the "
        "low-quality training distribution. Safe baseline regardless of "
        "base. Use when minimal isn't enough for anatomy issues.",
    ),
    (
        "NSFW · uncensor (recommended for explicit)",
        _with_photo_baseline(
            "score_4, score_5, score_6, censored, uncensored, mosaic, mosaic "
            "censoring, bar censor, black bar, pixelated, pixelization, "
            "blurred genitals, novelty censor, convenient censoring, "
            "covering breasts, covering crotch, hand over breasts, "
            "clothing covering body, low quality, worst quality, blurry, "
            "watermark, signature, text, jpeg artifacts, "
            "bad anatomy, deformed, mutated, extra limbs, extra fingers, "
            "fused fingers, malformed hands, missing fingers"
        ),
        "Comprehensive NSFW negative — pushes against censoring "
        "artefacts (mosaic, black bars, 'novelty censor', 'convenient "
        "censoring' tags Pony was trained on) AND the body-covering "
        "poses the model defaults to when uncertain. Plus the photo-"
        "only baseline. Pair with Pony explicit NSFW stack.",
    ),
    (
        "NSFW · photoreal push",
        _with_photo_baseline(
            "score_4, score_5, score_6, plastic skin, smooth skin, "
            "airbrushed, oiled skin, glossy skin, matte painting, "
            "pin-up illustration, oversaturated, low quality, worst quality, "
            "blurry, out of focus, watermark, signature, text, jpeg artifacts, "
            "bad anatomy, deformed, mutated, extra limbs, extra fingers, "
            "fused fingers, malformed hands, missing fingers, censored, "
            "mosaic, bar censor"
        ),
        "Heavy realism push — photo-only baseline plus anti-glamour "
        "tokens (plastic skin / smooth skin / oiled / glossy). Use with "
        "RealVisXL / JuggernautXL / Lustify or Pony in photo mode. "
        "'plastic skin' / 'oiled skin' are the key anti-Pony-default "
        "levers — without them you get the airbrushed 3D-doll look "
        "regardless of LoRAs.",
    ),
    (
        "NSFW · heavy anti-anime (aggressive)",
        _with_photo_baseline(
            "score_4, score_5, score_6, source_anime, source_cartoon, "
            "anime style, cartoonish, "
            "unreal engine, blender, rendered, "
            "figurine, mannequin, plastic skin, plastic, smooth skin, "
            "airbrushed, flawless skin, perfect skin, porcelain skin, "
            "oversaturated, vivid colors, highly saturated, "
            "big eyes, large eyes, anime eyes, shiny eyes, "
            "low quality, worst quality, lowres, blurry, watermark, "
            "signature, text, bad anatomy, deformed, extra limbs, extra "
            "fingers, fused fingers, malformed hands, censored, mosaic"
        ),
        "Maximum anti-anime / anti-stylised push. Use when the standard "
        "photoreal push still gives you illustrated output — this one "
        "also kills the stylised 3D-render look and Pony's characteristic "
        "large-eye anime-face proportions. Heavier token budget but worth "
        "it on vanilla Pony.",
    ),
    (
        "NSFW · real-skin imperfection",
        _with_photo_baseline(
            "score_4, score_5, score_6, airbrushed, photoshopped, retouched, "
            "perfect skin, porcelain skin, plastic skin, smooth skin, "
            "flawless skin, glossy skin, wet-look skin, doll skin, "
            "oversaturated, vivid, hdr, "
            "low quality, blurry, watermark, signature, bad anatomy, "
            "deformed, extra limbs, extra fingers, fused fingers"
        ),
        "Targets the 'plastic doll' skin look specifically — Pony's "
        "biggest tell when it's trying to look photoreal but isn't. "
        "Photo-only baseline plus airbrushed/photoshopped/retouched/"
        "glossy negatives so output gets actual pore texture + natural "
        "skin variation instead of Instagram-filter perfection.",
    ),
    (
        "Body-detail correction (pair with anything)",
        _with_photo_baseline(
            "score_4, score_5, score_6, bad anatomy, bad proportions, "
            "deformed body, mutated, extra limbs, extra arms, extra legs, "
            "extra hands, extra feet, extra fingers, fused fingers, "
            "malformed hands, missing fingers, missing limbs, asymmetrical "
            "eyes, cross-eyed, deformed face, mutated face, ugly face, "
            "fat rolls, unnatural body shape, weird torso, broken neck, "
            "long neck, short legs, short arms, deformed breasts, "
            "asymmetrical breasts, low quality, blurry, watermark"
        ),
        "Aggressive anatomy fix plus the photo-only baseline. Use when "
        "basic negatives aren't enough and you keep seeing extra limbs "
        "/ weird hands. Combine with an anatomy-correction LoRA from "
        "civitai for best results.",
    ),
    (
        "Soft sensual (artistic, not explicit)",
        _with_photo_baseline(
            "score_4, score_5, score_6, hardcore, explicit, graphic, "
            "penetration, cum, ejaculation, semen, low quality, worst "
            "quality, blurry, watermark, signature, bad anatomy, deformed, "
            "extra limbs, extra fingers, fused fingers, malformed hands"
        ),
        "For sensual / artistic nude work that should NOT be explicit. "
        "Suppresses explicit vocabulary while keeping quality + anatomy "
        "negatives, plus the photo-only baseline. Use with rating_safe "
        "/ rating_questionable on the prompt side.",
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


# ---------- Prompt builder dropdown data ----------
#
# Each entry is (display_label, prompt_fragment).
# - "(none)" leaves that axis unspecified — its fragment is the empty string.
# - Friendly labels are user-facing; prompt fragments are what gets injected.
# - Order within a list = order in the dropdown. Most-common first.
#
# When the user clicks "Build prompt from picks", the builder concatenates
# the non-empty fragments in a sensible order: subject identity first
# (1girl + appearance), then outfit, then activity/pose, then scene + camera.

BUILDER_SUBJECT: dict[str, list[tuple[str, str]]] = {
    # "Subject count" and "Gender" drive the primary subject noun (the
    # equivalent of the old hardcoded `1girl` anchor). They're placed
    # first so they render on the first UI row — these are the most
    # fundamental choices.
    #
    # Interaction: if Subject count is anything other than `(any)`, its
    # fragment takes precedence (it sets the plural form like "2girls").
    # If Subject count is `(any)`, Gender's fragment is used (default
    # `(any)` → "1girl" for backward compatibility with existing prompts).
    # See `_on_build_from_picks` for the resolution logic.
    "Subject count": [
        ("(any)", ""),
        ("solo (1)", "solo"),
        ("couple (M+F)", "1boy, 1girl, couple"),
        ("lesbian couple (2F)", "2girls, couple, yuri"),
        ("gay couple (2M)", "2boys, couple, yaoi"),
        ("threesome (2F+1M)", "2girls, 1boy, threesome"),
        ("threesome (2M+1F)", "2boys, 1girl, threesome"),
        ("threesome (3F)", "3girls, threesome"),
        ("group (4+)", "multiple people, group shot"),
    ],
    "Gender": [
        # Default is "1girl" so a user who sets neither Count nor Gender
        # still gets the same prompt they used to. Drop to empty only if
        # the user explicitly picks "(any) + Subject count override".
        ("(any)", "1girl"),
        ("woman (cis)", "1girl"),
        ("man (cis)", "1boy"),
        ("trans woman", "1girl, trans woman, transgender"),
        ("trans man", "1boy, trans man, transgender"),
        ("futanari (anime-style)", "1girl, futanari"),
        ("non-binary", "androgynous person, non-binary"),
    ],
    "Hair colour": [
        ("(any)", ""),
        ("blonde", "long wavy blonde hair"),
        ("platinum blonde", "platinum blonde hair, very light blonde"),
        ("ash blonde", "ash blonde hair, cool tone"),
        ("golden blonde", "golden blonde hair, warm honey tone"),
        ("honey blonde", "honey blonde hair, warm caramel tone"),
        ("dirty blonde", "dirty blonde hair"),
        ("strawberry blonde", "strawberry blonde hair"),
        ("bleached blonde", "bleached blonde hair, very light"),
        ("balayage", "balayage hair, blonde highlights on darker base"),
        ("ombre", "ombre hair, dark roots fading to light tips"),
        ("brunette", "long brunette hair"),
        ("dark brown", "long dark brown hair"),
        ("chestnut brown", "chestnut brown hair, warm tone"),
        ("black", "long black hair"),
        ("raven black", "long raven black hair, glossy"),
        ("auburn", "auburn hair"),
        ("red / ginger", "long red hair, ginger"),
        ("copper red", "copper red hair, vibrant"),
        ("white / silver", "white silver hair, platinum white"),
        ("pink dyed", "pink dyed hair"),
        ("blue dyed", "blue dyed hair"),
        ("purple dyed", "purple dyed hair"),
        ("rainbow dyed", "multi-coloured rainbow dyed hair"),
        ("short blonde", "short blonde hair, pixie cut"),
        ("short dark", "short dark hair, bob cut"),
        ("buzz cut", "buzz cut, very short hair"),
    ],
    "Eye colour": [
        ("(any)", ""),
        ("blue", "blue eyes"),
        ("green", "green eyes"),
        ("hazel", "hazel eyes"),
        ("brown", "brown eyes"),
        ("dark brown", "dark brown eyes"),
        ("grey", "grey eyes"),
        ("amber", "amber eyes"),
    ],
    "Skin tone": [
        ("(any)", ""),
        ("fair", "fair skin"),
        ("light", "light skin"),
        ("medium", "medium skin tone"),
        ("tanned", "tanned skin, sun-kissed"),
        ("olive", "olive skin"),
        ("dark", "dark skin"),
        ("ebony", "ebony skin, dark complexion"),
    ],
    "Body type": [
        ("(any)", ""),
        ("slim", "slim body, slender"),
        ("athletic", "athletic body, fit, toned"),
        ("petite", "petite body, small frame"),
        ("average", "average body, natural proportions"),
        ("curvy", "curvy body, hourglass figure"),
        ("voluptuous", "voluptuous body, full figure"),
        ("plus size", "plus size body, BBW"),
        ("muscular", "muscular body, defined muscles"),
    ],
    "Breast size": [
        ("(any)", ""),
        ("small", "small breasts, A cup"),
        ("medium", "medium breasts, B cup"),
        ("large", "large breasts, D cup"),
        ("huge", "huge breasts, very large"),
        ("perky natural", "perky natural breasts"),
        ("natural", "natural breasts"),
    ],
    "Makeup": [
        ("(any)", ""),
        ("none", "no makeup, bare face"),
        ("natural", "natural makeup, minimal"),
        ("light glam", "light makeup, mascara, light lip"),
        ("heavy glam", "heavy makeup, smokey eye, bold lip"),
        ("red lipstick", "red lipstick, glamour makeup"),
        ("instagram filter", "instagram makeup, contoured, polished"),
    ],
}

BUILDER_SCENE: dict[str, list[tuple[str, str]]] = {
    "Setting": [
        ("(any)", ""),
        ("bedroom", "in bedroom, unmade bed visible, cozy interior"),
        ("kitchen", "in kitchen, modern kitchen interior, counter visible"),
        ("bathroom", "in bathroom, bathroom mirror, tile background"),
        ("living room", "in living room, sofa, casual home interior"),
        ("home office", "in home office, casual workspace"),
        ("outdoor day", "outdoor scene, daylight, natural environment"),
        ("outdoor night", "outdoor at night, urban street, low light"),
        ("beach", "on beach, sand, ocean background, summer"),
        ("pool", "by pool, poolside, sunny day"),
        ("hotel room", "in hotel room, modern hotel decor, luxury"),
        ("car interior", "inside car, driver seat, modern interior"),
        ("nightclub", "nightclub, neon lighting, dance floor"),
        ("studio", "professional studio, neutral seamless backdrop"),
        ("forest / nature", "outdoor in forest, natural setting, trees"),
        ("rooftop", "on rooftop, city skyline view"),
    ],
    "Time of day": [
        ("(any)", ""),
        ("morning", "morning light, soft warm dawn light"),
        ("afternoon", "afternoon light, bright daylight"),
        ("golden hour", "golden hour, warm sunset light, lens flare"),
        ("evening", "evening, blue hour, fading light"),
        ("night", "night time, low light, ambient glow"),
    ],
    "Lighting": [
        ("(any)", ""),
        ("natural window", "soft natural window light, daylight diffused"),
        ("overhead lamp", "warm overhead lamp light, indoor"),
        ("smartphone flash", "smartphone flash, harsh on-camera flash photo"),
        ("candlelight", "warm candlelight, intimate, soft glow"),
        ("studio softbox", "professional studio lighting, softbox key + rim"),
        ("backlight", "backlit subject, silhouette lighting, rim light"),
        ("dramatic side", "dramatic side lighting, chiaroscuro, shadows"),
        ("neon / nightclub", "neon lighting, mixed colour spotlights"),
    ],
    "Camera angle": [
        ("(any)", ""),
        ("selfie arm-out", "selfie shot, smartphone camera, arms-length, "
                           "iphone photo aesthetic"),
        ("mirror selfie", "mirror selfie, full body in mirror, holding phone"),
        ("POV downward", "POV downward angle, subject looking up at viewer"),
        ("POV upward", "POV upward angle, subject looking down at viewer"),
        ("eye level", "eye level shot, straight-on framing"),
        ("low angle", "low angle shot, looking up at subject"),
        ("high angle", "high angle shot, looking down at subject"),
        ("medium shot", "medium shot, waist-up framing"),
        ("full body", "full body shot, head to toe visible"),
        ("close-up", "close-up shot, intimate framing, head and shoulders"),
        ("over the shoulder", "over-the-shoulder shot, looking back"),
    ],
}

BUILDER_ACTION: dict[str, list[tuple[str, str]]] = {
    "Activity": [
        ("(any)", ""),
        ("posing", "posing for camera, model pose"),
        ("selfie", "taking a selfie, smiling at camera"),
        ("standing", "standing casually"),
        ("sitting", "sitting, casual relaxed pose"),
        ("lying down", "lying down, relaxed on back"),
        ("walking", "walking, mid-stride, candid"),
        ("dancing", "dancing pose, dynamic motion"),
        ("getting dressed", "in the process of getting dressed"),
        ("undressing", "in the act of undressing, clothes coming off"),
        ("applying makeup", "applying makeup at vanity, mirror visible"),
        ("brushing hair", "brushing hair, casual morning routine"),
        ("on the phone", "talking on phone, holding phone to ear"),
        ("eating", "eating food, casual meal"),
        ("drinking", "drinking from a glass, casual moment"),
        ("workout / gym", "exercising, gym workout, fitness pose"),
        ("yoga pose", "yoga pose, stretching, athletic pose"),
        ("sleeping", "sleeping, eyes closed, in bed"),
        ("waking up", "just waking up, sleepy expression, in bed"),
        ("showering", "in shower, water running, wet skin"),
        ("bathing", "in bathtub, bubble bath, relaxed"),
        ("reading", "reading a book, casual moment"),
        ("smoking", "smoking a cigarette, casual"),
        ("masturbating", "masturbating, intimate solo, hand between legs"),
        ("oral / blowjob", "performing oral sex, blowjob, fellatio"),
        ("intercourse", "intercourse, intimate sex"),
        ("cowgirl", "cowgirl position, on top, straddling"),
        ("reverse cowgirl", "reverse cowgirl position, facing away"),
        ("doggystyle", "doggystyle position, from behind"),
        ("missionary", "missionary position, lying on back"),
        ("69", "69 position, mutual oral"),
    ],
    "Outfit / clothing": [
        ("(any)", ""),
        ("fully nude", "fully nude, naked, completely undressed"),
        ("topless", "topless, bare breasts, jeans / pants on"),
        ("bottomless", "bottomless, naked from waist down, top still on"),
        ("lingerie", "wearing lingerie, lacy bra and panties"),
        ("bikini", "wearing bikini, swimsuit"),
        ("micro bikini", "tiny micro bikini, minimal coverage"),
        ("casual outfit", "casual clothing, t-shirt and jeans"),
        ("workout clothes", "wearing workout clothes, sports bra, leggings"),
        ("sleepwear", "wearing sleepwear, pajamas, oversized t-shirt"),
        ("cocktail dress", "wearing tight cocktail dress, evening wear"),
        ("sundress", "wearing summer sundress"),
        ("sheer / see-through", "wearing sheer see-through clothing"),
        ("schoolgirl", "schoolgirl outfit, plaid skirt, white shirt"),
        ("nurse", "nurse outfit, medical uniform"),
        ("french maid", "french maid outfit, lace and apron"),
    ],
    "Pose": [
        ("(any)", ""),
        ("standing contrapposto", "standing contrapposto, weight on one leg"),
        ("hand on hip", "one hand on hip"),
        ("hands on hips", "both hands on hips, confident stance"),
        ("arms behind head", "arms raised behind head"),
        ("arms behind back", "arms behind back"),
        ("arms crossed", "arms crossed across chest"),
        ("hands on chest", "hands cupping breasts"),
        ("hand in hair", "one hand running through hair"),
        ("peace sign", "flashing peace sign with one hand"),
        ("blowing a kiss", "blowing a kiss to camera"),
        ("on knees", "on knees, kneeling pose"),
        ("kneeling spread", "kneeling with knees spread apart"),
        ("on all fours", "on all fours, hands and knees"),
        ("legs spread", "legs spread wide"),
        ("legs crossed", "legs crossed"),
        ("legs up", "legs raised in the air"),
        ("squatting", "squatting pose"),
        ("crouching", "crouching low, balanced on toes"),
        ("bent over", "bent over, ass facing camera"),
        ("leaning forward", "leaning forward toward camera"),
        ("leaning back", "leaning back, weight on hands"),
        ("arched back", "arched back, breasts forward"),
        ("looking back", "looking back at viewer over shoulder"),
        ("over the shoulder", "glancing back over the shoulder, head turned"),
        ("on side", "lying on side, hip raised"),
        ("on stomach", "lying on stomach, propped on elbows"),
        ("on back", "lying on back, relaxed"),
        ("hugging knees", "sitting hugging knees to chest"),
        ("sitting on chair", "sitting on a chair, legs crossed"),
        ("sitting on floor", "sitting on the floor, casual"),
        ("jumping", "mid-jump, hair and clothing in motion"),
        ("walking toward camera", "walking toward camera, candid step"),
        ("touching face", "fingertips touching cheek or jaw"),
        ("biting finger", "biting tip of finger, playful"),
    ],
    "Expression": [
        ("(any)", ""),
        ("smile", "warm smile, genuine"),
        ("smirk", "playful smirk, mischievous"),
        ("parted lips", "parted lips, sultry expression"),
        ("biting lip", "biting lower lip, seductive"),
        ("tongue out", "tongue out, playful"),
        ("serious", "serious expression, neutral"),
        ("looking away", "looking away from camera, candid"),
        ("eye contact", "intense eye contact with viewer"),
        ("laughing", "laughing, candid moment, real expression"),
        ("orgasm face", "orgasm face, eyes closed, pleasure expression"),
        ("ahegao", "ahegao expression, eyes rolled back, tongue out"),
    ],
}


#: NSFW-friendly base checkpoint families with one-line guidance.
#: Photoreal-first ordering — vanilla Pony V6 XL is deliberately NOT at
#: the top because it's ~50% anime-trained and drifts stylised even with
#: source_real. For photoreal work, start from a Pony fine-tune that was
#: trained specifically on photo data.
BASE_RECOMMENDATIONS: list[tuple[str, str]] = [
    (
        "Lustify XL  ★ best photoreal NSFW",
        "Pony fine-tune trained heavily on photographic NSFW data. "
        "Same score_X vocabulary as Pony but photorealistic output out "
        "of the box — you DON'T have to fight anime bias. Top pick for "
        "personal-likeness LoRA work. Civitai.",
    ),
    (
        "Pony Realism V2.x  ★ photoreal",
        "Another Pony fine-tune optimized for photoreal output. Slightly "
        "different skin tone + lighting bias than Lustify. Pair with Pony "
        "score_X stack. Worth trying both to see which you prefer. Civitai.",
    ),
    (
        "CyberRealistic Pony",
        "Another Pony photoreal fine-tune. Tends toward warmer lighting "
        "and glossier skin — less gritty than Lustify, less editorial "
        "than Pony Realism. Civitai.",
    ),
    (
        "RealVisXL V5  ·  non-Pony photoreal",
        "SDXL fine-tune, not Pony-derivative. No score_X tags. Cleaner "
        "faces but tighter on NSFW — needs the heavier uncensor negative "
        "and explicit body-part anchors in the prompt. Use the 'Photoreal "
        "pro (non-Pony)' quality stack.",
    ),
    (
        "EpicRealism XL",
        "Another non-Pony photoreal SDXL. Strong on lighting + skin. "
        "Same caveats as RealVisXL — tighter NSFW, use heavy uncensor.",
    ),
    (
        "Pony Diffusion V6 XL  ·  caveat",
        "The original. Industry standard but ~50% anime-trained — drifts "
        "stylised even with source_real. Use 'Pony · heavy photoreal' "
        "quality stack + heavy anti-anime negative to force photo output. "
        "Or just switch to Lustify / Pony Realism for photoreal work.",
    ),
    (
        "JuggernautXL",
        "Versatile SDXL fine-tune. Good photoreal baseline, fewer "
        "Pony-isms. Tighter on NSFW than the Pony family.",
    ),
    (
        "Illustrious-XL / NoobAI-XL",
        "Anime/illustration NSFW. Different prompt vocabulary "
        "(masterpiece / very aware / etc.). Only if you WANT stylised "
        "output.",
    ),
]


#: Civitai LoRA categories worth searching for (no direct links —
#: civitai LoRAs come and go).
LORA_RECOMMENDATIONS: list[tuple[str, str]] = [
    (
        "Realistic skin  ★ photoreal fix",
        "Search civitai for 'realistic skin XL', 'real skin SDXL', "
        "'film skin LoRA', 'skin pores LoRA'. Single biggest lever for "
        "breaking out of Pony's plastic-doll skin. Stack at 0.5-0.8.",
    ),
    (
        "Anti-anime / photoreal push  ★ photoreal fix",
        "Search 'photo style XL', 'realistic photo LoRA', 'film photography "
        "XL'. Actively pushes Pony away from its anime default. Stack at "
        "0.6-0.9 if you're fighting stylised output.",
    ),
    (
        "Detail enhancer",
        "Search 'add detail XL' / 'detail tweaker XL'. Adds skin "
        "micro-texture, fabric weave, hair strands. Use at 0.3-0.6 weight; "
        "higher = noisy output.",
    ),
    (
        "Anatomy correction",
        "Search 'perfect anatomy' / 'better anatomy XL'. Fixes hand/foot "
        "deformity at the cost of some style. Use at 0.4-0.7.",
    ),
    (
        "Lighting LoRAs",
        "Search 'cinematic lighting XL' / 'natural lighting XL' / "
        "'golden hour LoRA'. Stack at 0.4-0.7 to push specific lighting "
        "setups.",
    ),
    (
        "Pose pack",
        "Search by pose name ('standing nude XL', 'lying down XL', "
        "'contrapposto SDXL', etc.). Pose LoRAs are the best way to get "
        "reliable composition without ControlNet.",
    ),
    (
        "Film / analog aesthetic",
        "Search '35mm film XL', 'kodak portra LoRA', 'analog photography "
        "SDXL'. Stack at 0.4-0.7 for grain + colour-science authenticity. "
        "Great anti-anime signal.",
    ),
    (
        "Amateur / smartphone aesthetic",
        "Search 'iphone photo LoRA', 'amateur photography XL', 'flash "
        "photography SDXL'. Best match to OnlyFans-style training data. "
        "Stack at 0.4-0.6.",
    ),
]


# ---------- LoRA category detection ----------
#
# Each entry: (keywords_to_match_in_stem, category_label, default_weight, weight_hint)
# First matching rule wins. Keyword matching is case-insensitive substring.
# Recommended-weight table for the LoRA picker. These defaults / hints
# are what the UI suggests when you tick a LoRA. Values calibrated for
# Pony V6 XL as the base — total stacked extra-LoRA weight past ~1.0
# reliably degrades photoreal output (see Test/outputs/defaults_analysis.md
# for the evidence: a 6-LoRA stack at total 3.55 produced literal mush),
# so every category's default sits well below vanilla SDXL recommendations.
#
# Order matters: the FIRST matching rule wins. More specific keywords
# (PONY REALISM, EYES, FACE) come before the broader PHOTO STYLE catch
# so e.g. "zy_Realism_Enhancer_v2" matches PONY REALISM, not PHOTO STYLE.
_LORA_CATEGORIES: list[tuple[list[str], str, str, str]] = [
    # Pony-family realism LoRAs — calibrated against Pony-derivative
    # checkpoints (ponyrealism, zy_*). Lower weight than vanilla SDXL
    # realism LoRAs because they're steering WITHIN Pony's distribution
    # instead of against it. 0.3–0.5 is the proven sweet spot.
    (["zy_realism", "pony_realism", "ponyrealism", "zy_amateur"],
     "PONY REALISM", "0.35", "0.3 – 0.5"),
    # Eye detailers get stacked frequently at too-high weights; 0.3 is
    # plenty. Don't tick two eye LoRAs at once — they fight each other.
    (["eye", "iris", "pupil"],
     "EYES", "0.30", "0.2 – 0.4"),
    # Face/visage helpers — same story as eyes. Don't stack with a
    # separate eye LoRA; pick one.
    (["face", "visage", "facial"],
     "FACE", "0.30", "0.2 – 0.4"),
    (["skin", "pore", "texture", "dermis", "complexion", "zit", "zib", "zt"],
     "SKIN TEXTURE", "0.50", "0.4 – 0.6"),
    (["ultrareal", "lenovo", "klein", "amateur", "iphone", "candid", "snapshot"],
     "AMATEUR PHOTO", "0.40", "0.3 – 0.5"),
    (["photo", "real", "realis", "film", "analog", "kodak", "portra", "grain", "dslr"],
     "PHOTO STYLE", "0.40", "0.3 – 0.6"),
    (["detail", "enhance", "sharp", "clarity", "micro", "add_detail"],
     "DETAIL BOOST", "0.35", "0.3 – 0.5"),
    (["anatomy", "hand", "feet", "finger", "correct", "body"],
     "ANATOMY FIX", "0.45", "0.3 – 0.6"),
    (["pose", "standing", "lying", "sitting", "contrap"],
     "POSE PACK", "0.50", "0.4 – 0.7"),
    (["light", "shadow", "golden", "sunset", "lamp", "lighting"],
     "LIGHTING", "0.40", "0.3 – 0.5"),
]


# ---------- Builder ----------

def _parse_run_info_text(text: str) -> dict:
    """Parse a `run_info.txt` (as written by `_write_run_info` in
    pipeline/generate.py) into a dict of settings the GUI can apply.

    Returns a dict with any subset of keys:
      - prompt   : str (multiline body without quality-stack prefix)
      - negative : str
      - sampler  : str (e.g. "dpmpp_2m_karras")
      - steps    : int
      - guidance : float
      - width    : int
      - height   : int
      - extra_loras : list[(path_str, weight_float)]

    Missing fields are simply absent from the dict — caller decides
    whether to treat absence as 'leave unchanged'.

    Tolerant of minor format drift: the parser only looks for specific
    line prefixes / section headers and ignores everything else.
    """
    result: dict = {}
    lines = text.splitlines()

    # Pass 1: scan top-of-file key:value pairs and the LoRA list block.
    in_lora_block = False
    extra_loras: list[tuple[str, float]] = []
    for ln in lines:
        s = ln.rstrip()
        # Stop the LoRA block at the first blank line OR a new key line.
        if in_lora_block:
            if not s.strip() or ":" in s.split("@")[0].split(":", 1)[0] and not s.lstrip().startswith("-"):
                in_lora_block = False
            else:
                # Match "  - <abs path>  @ weight <float>"
                stripped = s.strip()
                if stripped.startswith("-"):
                    body = stripped[1:].strip()
                    if "@" in body:
                        path_str, _, tail = body.rpartition("@")
                        path_str = path_str.strip()
                        # Tail looks like "weight 0.45"
                        toks = tail.split()
                        for t in toks:
                            try:
                                extra_loras.append((path_str, float(t)))
                                break
                            except ValueError:
                                continue
                    continue
                # If we hit something non-LoRA in the block, fall through
                in_lora_block = False

        if not in_lora_block:
            if s.startswith("sampler"):
                _, _, v = s.partition(":")
                if v.strip():
                    result["sampler"] = v.strip()
            elif s.startswith("steps"):
                _, _, v = s.partition(":")
                try:
                    result["steps"] = int(v.strip())
                except ValueError:
                    pass
            elif s.startswith("guidance"):
                _, _, v = s.partition(":")
                try:
                    result["guidance"] = float(v.strip())
                except ValueError:
                    pass
            elif s.startswith("dimensions"):
                _, _, v = s.partition(":")
                # "1024 x 1024" or "896 x 1152"
                parts = [p.strip() for p in v.lower().split("x")]
                if len(parts) == 2:
                    try:
                        result["width"] = int(parts[0])
                        result["height"] = int(parts[1])
                    except ValueError:
                        pass
            elif s.startswith("extra LoRAs"):
                # The header line is either:
                #   "extra LoRAs       : (none)"   → empty list
                #   "extra LoRAs       :"          → list follows
                _, _, v = s.partition(":")
                if v.strip().startswith("(none"):
                    pass  # leave extra_loras empty
                else:
                    in_lora_block = True

    if extra_loras:
        result["extra_loras"] = extra_loras

    # Pass 2: prompt / negative blocks (between header underline and the
    # next blank line + new section header). Walk the lines looking for
    # "prompt" then "------" then collect until the next blank-line +
    # non-section pattern.
    def _grab_block(label: str) -> str | None:
        for i, line in enumerate(lines):
            if line.strip() == label:
                if i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
                    j = i + 2
                    out: list[str] = []
                    while j < len(lines):
                        nxt = lines[j]
                        # Section break: blank line followed by a known
                        # section header on the next non-blank line.
                        if not nxt.strip():
                            # Peek: if the following non-blank line is
                            # a section header ("negative" / "outputs" /
                            # "stacks tested" / "recipes"), stop.
                            k = j + 1
                            while k < len(lines) and not lines[k].strip():
                                k += 1
                            if k >= len(lines) or lines[k].strip() in {
                                "prompt", "negative", "outputs",
                                "stacks tested", "recipes", "run_info",
                            }:
                                break
                        out.append(nxt)
                        j += 1
                    # Trim trailing blanks.
                    while out and not out[-1].strip():
                        out.pop()
                    return "\n".join(out).strip()
        return None

    body = _grab_block("prompt")
    neg = _grab_block("negative")
    if body is not None:
        # Strip any leading quality-stack opener so the body slot in the
        # GUI gets the clean text the user typed. Heuristic: if the body
        # starts with `score_X` tokens, drop everything up to and
        # including the first ", " that follows the last `score_` /
        # `source_` / `rating_` / `professional photography` token.
        result["prompt"] = _strip_known_stack_prefixes(body)
    if neg is not None and neg.lower() != "(none)":
        result["negative"] = neg

    return result


def _strip_known_stack_prefixes(prompt: str) -> str:
    """Drop a Pony / SDXL quality-stack opener from the front of a prompt.

    The GUI re-prepends the chosen stack at generate time, so storing
    the body WITHOUT the opener lets the user freely switch stacks
    without doubling tokens. Matches the openers in QUALITY_STACKS
    (score_X, source_real, rating_*, photo, photorealistic, raw photo,
    real photograph, 35mm film, skin pores, subsurface scattering,
    professional photography, candid snapshot, amateur photograph,
    natural skin, no makeup filter, real woman).

    Conservative: only strips contiguous opener tokens. Anything past
    the first non-opener token stays as-is.
    """
    OPENERS = {
        "score_9", "score_8_up", "score_7_up", "score_6_up",
        "score_5_up", "score_4_up", "source_real", "rating_explicit",
        "rating_safe", "rating_questionable", "photo",
        "photorealistic", "raw photo", "real photograph",
        "35mm film", "skin pores", "subsurface scattering",
        "professional photography", "candid snapshot",
        "amateur photograph", "natural skin", "no makeup filter",
        "real woman",
    }
    parts = [p.strip() for p in prompt.split(",")]
    i = 0
    while i < len(parts) and parts[i].lower() in OPENERS:
        i += 1
    if i == 0:
        return prompt
    return ", ".join(parts[i:]).strip()


class _LoraCompareDialog:
    """Modal dialog for the 'Compare LoRA recipes' feature.

    The dialog asks three questions:
      1. **Recipe set** — radio between three preset modes:
         "each_solo"  : every checked LoRA active alone (plus a no-LoRAs
                        baseline and an all-together render).
         "weight_sweep": one chosen LoRA, several weights; other checked
                        LoRAs stay at their current weight every render.
         "powerset"   : every subset (2^N renders for N checked LoRAs).
      2. **Trained LoRA in every recipe** — checkbox. When ticked, the
         project's trained likeness LoRA is added to every recipe at
         weight 1.0 so the subject identity stays consistent while the
         only thing varying is the style stack.
      3. **Cross with quality stacks** — checkbox + multiselect.
         When ticked, the recipe list is cross-producted with the
         picked stack labels. Total renders = recipes × stacks.

    Returns ``(recipes, stack_labels, use_trained_in_every)`` on Compare,
    or ``None`` on Cancel. ``recipes`` is a list of (label, [(path, weight), ...]).
    """

    def __init__(
        self,
        parent: tk.Misc,
        checked_loras: list[tuple[Path, float]],
        use_trained: bool,
        current_stack: str,
    ) -> None:
        self.parent = parent
        self.checked_loras = checked_loras  # [(path, weight), ...]
        self.use_trained = use_trained
        self.current_stack = current_stack
        self.result: object = None  # populated on OK; None on cancel

    def show(self) -> object:
        from ..prompt_presets import QUALITY_STACKS

        top = tk.Toplevel(self.parent)
        top.title("Compare LoRA recipes")
        top.transient(self.parent)
        top.resizable(False, False)
        # Modal: block parent until dialog resolves.
        top.grab_set()

        PAD = gui_theme.PAD
        body = ttk.Frame(top, padding=PAD)
        body.pack(fill="both", expand=True)

        # ---- Mode picker ----
        mode_var = tk.StringVar(value="each_solo")
        ttk.Label(
            body, text="Recipe set:", style="Mono.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 2))
        ttk.Radiobutton(
            body, text="Each checked LoRA solo (+ all-together + no-LoRAs baseline)",
            variable=mode_var, value="each_solo",
        ).grid(row=1, column=0, columnspan=3, sticky="w")

        # Weight-sweep row needs sub-controls (LoRA picklist + weights).
        sweep_row = ttk.Frame(body)
        sweep_row.grid(row=2, column=0, columnspan=3, sticky="we")
        ttk.Radiobutton(
            sweep_row, text="Weight sweep on:",
            variable=mode_var, value="weight_sweep",
        ).pack(side="left")
        sweep_lora_var = tk.StringVar(
            value=(self.checked_loras[0][0].stem if self.checked_loras else "")
        )
        sweep_lora_combo = ttk.Combobox(
            sweep_row,
            textvariable=sweep_lora_var,
            values=[p.stem for p, _ in self.checked_loras],
            state="readonly", width=28,
        )
        sweep_lora_combo.pack(side="left", padx=(4, 8))
        ttk.Label(sweep_row, text="weights:", style="Status.TLabel").pack(side="left")
        sweep_weights_var = tk.StringVar(value="0.0, 0.25, 0.5, 0.75, 1.0")
        ttk.Entry(
            sweep_row, textvariable=sweep_weights_var, width=22,
        ).pack(side="left", padx=(4, 0))

        ttk.Radiobutton(
            body, text="All combinations (powerset of checked LoRAs)",
            variable=mode_var, value="powerset",
        ).grid(row=3, column=0, columnspan=3, sticky="w")

        # ---- Trained-LoRA checkbox ----
        trained_in_every_var = tk.BooleanVar(value=self.use_trained)
        ttk.Checkbutton(
            body, variable=trained_in_every_var,
            text="Include trained likeness LoRA in every recipe (recommended)",
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(PAD, 2))

        # ---- Cross-with-stacks ----
        cross_var = tk.BooleanVar(value=True)
        cross_cb = ttk.Checkbutton(
            body, variable=cross_var,
            text="Also vary the quality stack",
        )
        cross_cb.grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 2))

        stacks_row = ttk.Frame(body)
        stacks_row.grid(row=6, column=0, columnspan=3, sticky="we")
        ttk.Label(stacks_row, text="Stacks to test:", style="Status.TLabel").pack(
            anchor="w"
        )
        # Multi-select listbox of stack labels — exclude "(none)".
        stack_labels = [lbl for lbl, prefix, _ in QUALITY_STACKS if prefix]
        stacks_list = tk.Listbox(
            stacks_row,
            selectmode="multiple",
            height=min(7, len(stack_labels)),
            exportselection=False,
        )
        for lbl in stack_labels:
            stacks_list.insert("end", lbl)
        # Pre-select the user's currently-active stack so the dialog opens
        # in a sensible state if they just want to test the current stack.
        if self.current_stack in stack_labels:
            stacks_list.selection_set(stack_labels.index(self.current_stack))
        stacks_list.pack(fill="x", padx=(20, 0))

        # ---- Image-count + estimate (live) ----
        count_var = tk.StringVar(value="")
        ttk.Label(
            body, textvariable=count_var, style="Mono.TLabel",
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(PAD, 0))

        def _recompute() -> None:
            try:
                n_recipes = len(self._build_recipes(
                    mode_var.get(),
                    sweep_lora_var.get(),
                    sweep_weights_var.get(),
                ))
            except Exception:
                n_recipes = 0
            cross_on = cross_var.get()
            if cross_on:
                sel = [stacks_list.get(i) for i in stacks_list.curselection()]
                n_stacks = max(1, len(sel))
            else:
                n_stacks = 1
            total = n_recipes * n_stacks
            secs = total * 15  # rough per-image estimate at 1024 / 28 steps
            count_var.set(
                f"Total images: {total}  ({n_recipes} recipes × "
                f"{n_stacks} stack{'s' if n_stacks != 1 else ''})"
                f"   est. ~{secs}s"
            )

        # Re-compute the count every time anything relevant changes.
        for var in (mode_var, sweep_lora_var, sweep_weights_var, cross_var):
            var.trace_add("write", lambda *_: _recompute())
        stacks_list.bind("<<ListboxSelect>>", lambda _e: _recompute())
        _recompute()

        # ---- buttons ----
        btn_row = ttk.Frame(body)
        btn_row.grid(row=8, column=0, columnspan=3, sticky="e", pady=(PAD, 0))

        def _on_ok() -> None:
            try:
                recipes = self._build_recipes(
                    mode_var.get(),
                    sweep_lora_var.get(),
                    sweep_weights_var.get(),
                )
            except ValueError as e:
                messagebox.showerror("Bad recipe input", str(e), parent=top)
                return
            if not recipes:
                messagebox.showerror(
                    "No recipes",
                    "The selected mode produced zero recipes. Tick more "
                    "LoRAs or pick a different mode.",
                    parent=top,
                )
                return
            if cross_var.get():
                sel_idx = stacks_list.curselection()
                stacks_chosen = [stacks_list.get(i) for i in sel_idx]
                if not stacks_chosen:
                    messagebox.showerror(
                        "No stacks selected",
                        "Tick at least one quality stack in the list, "
                        "or untick 'Also vary the quality stack' to use "
                        "your current stack only.",
                        parent=top,
                    )
                    return
            else:
                stacks_chosen = [self.current_stack]
            # Hard ceiling — refuse runaway runs.
            total = len(recipes) * max(1, len(stacks_chosen))
            if total > 32:
                ok = messagebox.askokcancel(
                    "Large compare run",
                    f"This will render {total} images "
                    f"(~{total * 15} seconds). Continue?",
                    parent=top,
                )
                if not ok:
                    return
            self.result = (
                recipes, stacks_chosen, bool(trained_in_every_var.get()),
            )
            top.destroy()

        ttk.Button(
            btn_row, text="Cancel", style="Ghost.TButton",
            command=top.destroy,
        ).pack(side="right", padx=(PAD // 2, 0))
        ttk.Button(
            btn_row, text="Compare ▶", style="Primary.TButton",
            command=_on_ok,
        ).pack(side="right")

        # Wait until the dialog closes (modal blocking).
        self.parent.wait_window(top)
        return self.result

    def _build_recipes(
        self, mode: str, sweep_stem: str, sweep_weights_csv: str,
    ) -> list[tuple[str, list[tuple[Path, float]]]]:
        """Translate (mode + state) into the list of recipes.

        Each recipe is ``(label, [(path, weight), ...])``. The trained
        LoRA is NOT included here — it's handled at engine-time by
        the ``use_trained_in_every`` flag.
        """
        if mode == "each_solo":
            recipes: list[tuple[str, list[tuple[Path, float]]]] = []
            recipes.append(("no_extra_loras", []))
            for path, weight in self.checked_loras:
                recipes.append((f"{path.stem}_only_{weight}", [(path, weight)]))
            if len(self.checked_loras) > 1:
                recipes.append(("all_together", list(self.checked_loras)))
            return recipes

        if mode == "weight_sweep":
            sweep_path = next(
                (p for p, _ in self.checked_loras if p.stem == sweep_stem),
                None,
            )
            if sweep_path is None:
                raise ValueError(
                    "Pick a LoRA from the dropdown to sweep weights for."
                )
            try:
                weights = [
                    float(w.strip()) for w in sweep_weights_csv.split(",")
                    if w.strip()
                ]
            except ValueError:
                raise ValueError(
                    f"Couldn't parse weight list {sweep_weights_csv!r}. "
                    "Use comma-separated floats: 0.0, 0.25, 0.5, 0.75, 1.0"
                )
            if not weights:
                raise ValueError("Need at least one weight to sweep.")
            other_loras = [
                (p, w) for p, w in self.checked_loras if p != sweep_path
            ]
            return [
                (
                    f"{sweep_stem}_at_{w:.2f}",
                    [(sweep_path, w)] + other_loras,
                )
                for w in weights
            ]

        if mode == "powerset":
            from itertools import combinations
            recipes = []
            n = len(self.checked_loras)
            for k in range(0, n + 1):
                for combo in combinations(self.checked_loras, k):
                    if not combo:
                        recipes.append(("no_extra_loras", []))
                    else:
                        label = "+".join(p.stem for p, _ in combo)[:60]
                        recipes.append((label, list(combo)))
            return recipes

        raise ValueError(f"Unknown mode {mode!r}")


def build(gui: "TrainerGUI") -> None:
    PAD = gui_theme.PAD
    f = gui.tab_generate
    # Pin the tab content horizontally; the row weight for the scrollable
    # region is set further down once we know which row holds the scroll
    # canvas (after the sticky action bar above it).
    f.columnconfigure(0, weight=1)

    gui.prompt_var = tk.StringVar(value="")
    gui.negative_var = tk.StringVar(value="")
    # Default n=2 for fast iteration. Scale up to 4+ once a prompt is
    # dialled in and you want multiple seeds to pick from.
    gui.n_var = tk.StringVar(value="2")
    gui.steps_var = tk.StringVar(value="28")
    # 5.0 is the photoreal sweet spot for Pony V6 XL + a Pony-calibrated
    # realism LoRA (e.g. zy_Realism_Enhancer_v2). Above ~5.5, skin fries
    # into plastic / hyperstylised looks. Bump to 7-8 only if you want
    # more aggressive prompt-following at the cost of realism.
    gui.guidance_var = tk.StringVar(value="5.0")
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
    # Default quality stack: read from per-user settings if set, else fall
    # back to the recommended Pony NSFW opener. The pick auto-persists to
    # .user_settings.json whenever the combobox changes (via the global
    # persist trace), so the saved value survives launches automatically.
    _user = gui_helpers.load_user_settings(gui.projects_root.root)
    _default_stack = _user.get(
        "default_quality_stack", "Pony · photoreal NSFW (recommended)",
    )
    # If the persisted default refers to a stack that no longer exists
    # (e.g. we renamed one), fall back rather than show an empty combobox.
    if _default_stack not in [label for label, _, _ in QUALITY_STACKS]:
        _default_stack = "Pony · photoreal NSFW (recommended)"
    gui.quality_stack_var = tk.StringVar(value=_default_stack)
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
    # Progress bar lives in column 5 (after Generate / Compare stacks /
    # Compare LoRAs / Load run_info / Open outputs), so that's the column
    # that absorbs leftover horizontal space.
    action_bar.columnconfigure(5, weight=1)
    state.generate_btn = ttk.Button(
        action_bar, text="Generate", style="Primary.TButton",
        command=state._on_generate,
    )
    state.generate_btn.grid(row=0, column=0, sticky="w")
    state.compare_btn = ttk.Button(
        action_bar, text="Compare stacks",
        style="Ghost.TButton",
        command=state._on_compare_stacks,
    )
    state.compare_btn.grid(row=0, column=1, sticky="w", padx=(PAD // 2, 0))
    state.compare_loras_btn = ttk.Button(
        action_bar, text="Compare LoRAs",
        style="Ghost.TButton",
        command=state._on_compare_loras,
    )
    state.compare_loras_btn.grid(row=0, column=2, sticky="w", padx=(PAD // 2, 0))
    ttk.Button(
        action_bar, text="Load run_info…", style="Ghost.TButton",
        command=state._on_load_run_info,
    ).grid(row=0, column=3, sticky="w", padx=(PAD // 2, 0))
    ttk.Button(
        action_bar, text="Open outputs", style="Ghost.TButton",
        command=state._open_outputs,
    ).grid(row=0, column=4, sticky="w", padx=(PAD // 2, 0))
    state.progress = ttk.Progressbar(
        action_bar, mode="determinate",
        style="Trainer.Horizontal.TProgressbar",
    )
    state.progress.grid(row=0, column=5, sticky="we", padx=(PAD, 0))
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
    # Restore the user's persisted builder picks (if any). Done AFTER
    # build_form so the StringVars actually exist when we set them.
    state._apply_persisted_builder_defaults()
    # Restore every OTHER Generate-tab field (prompt, negative, inference
    # knobs, LoRA stack). Must run AFTER build_form + refresh_lora_list so
    # the target StringVars / row dicts exist, and BEFORE installing the
    # write-traces so the restore itself doesn't trigger a save.
    state._apply_persisted_generate_defaults()
    # Auto-rebuild the prompt body whenever a builder pick changes. Installed
    # AFTER both restore passes so replaying saved picks doesn't stomp on
    # the restored prompt text. Both restore methods use self._restoring as
    # a guard that the trace respects.
    state._install_builder_autoupdate_traces()
    state._install_persistence_traces()


class _GenerateState:
    """Holds the LoRA-stack rows + handlers for the Generate tab."""

    def __init__(self, gui: "TrainerGUI") -> None:
        self.gui = gui
        self.lora_rows: dict[str, dict] = {}
        self.lora_table: tk.Widget | None = None
        # Prompt-builder selections — one StringVar per dropdown, defaulting
        # to "(any)" so nothing is injected unless the user picks something.
        # Three groups (Subject / Scene / Action) keyed by display label.
        self.builder_vars: dict[str, dict[str, tk.StringVar]] = {
            "Subject": {
                k: tk.StringVar(value=opts[0][0])
                for k, opts in BUILDER_SUBJECT.items()
            },
            "Scene": {
                k: tk.StringVar(value=opts[0][0])
                for k, opts in BUILDER_SCENE.items()
            },
            "Action": {
                k: tk.StringVar(value=opts[0][0])
                for k, opts in BUILDER_ACTION.items()
            },
        }
        # Tattoos is a single boolean — kept separate from the dropdowns.
        self.builder_tattoos_var = tk.BooleanVar(value=False)
        # Cold-start default for the prompt body. Only used on the very
        # first launch — once the user has interacted with the Generate
        # tab, the saved ``generate_defaults.prompt`` in .user_settings.json
        # overrides this on every subsequent launch.
        #
        # Kept intentionally lean: just the photoreal anchors and a lens
        # specifier, plus a single `1girl`. No hair/eye/body descriptors,
        # because the Prompt Builder picks add those live — duplicating
        # them here forces users to delete them every time they change a
        # pick. No filler adjectives (`cute`, `sexy`, `gorgeous`) — they
        # waste token budget without improving output on Pony.
        # Do NOT include `iphone photo` / `smartphone photo` — Pony takes
        # those literally and renders a phone held in front of the subject.
        self.body_var = tk.StringVar(
            value=(
                "1girl, raw photo, real photograph, professional "
                "photography, candid snapshot, amateur photograph, "
                "natural skin, 85mm lens, shallow depth of field"
            )
        )
        # Run tracking so on_progress_line knows what to render.
        self.run_active: bool = False
        self.images_total: int = 0
        self.images_done: int = 0

    # ---- left column: form ----

    def build_form(self, root: ttk.Frame) -> None:
        PAD = gui_theme.PAD

        # Layout strategy (after the consolidation pass):
        # The Prompt builder is now the single hub for everything that
        # shapes a prompt — quality-tag prefix, the dropdown picks, AND
        # the sampler / dimensions / seed / folder controls. The body
        # editor sits just below it because that's where the user lands
        # after Build. Negative + LoRA stack stay separate because they
        # have their own logic surfaces (preset picker / file table).
        #
        # The Quick prompt template picker is gone — every template it
        # used to surface is reproducible (and editable) via the builder
        # dropdowns + body editor, so keeping a parallel one-click path
        # was just adding choice paralysis with no payoff.

        # ROW 0: Prompt builder (expanded — the hub)
        pb_outer = CollapsibleFrame(root, text="Prompt builder", start_open=True)
        pb_outer.grid(row=0, column=0, sticky="we", pady=(0, PAD))
        pb_box = ttk.Frame(pb_outer.body, padding=PAD)
        pb_box.pack(fill="both", expand=True)
        self._build_prompt_builder(pb_box)

        # ROW 1: Prompt body (always visible — this is the main editable output)
        prompt_box = ttk.LabelFrame(root, text="Prompt body", padding=PAD)
        prompt_box.grid(row=1, column=0, sticky="we", pady=(0, PAD))
        prompt_box.columnconfigure(0, weight=1)
        # Explanatory intro paragraph removed — moved behind the LabelFrame
        # title's info_icon below, where the user can open it on demand.
        # Frees ~30px of always-visible vertical space.
        _body_hint_row = ttk.Frame(prompt_box)
        _body_hint_row.grid(row=0, column=0, sticky="we")
        info_icon(
            _body_hint_row,
            "What you'd type yourself. The quality stack picked above is "
            "prepended automatically; the trigger word is injected when a "
            "trained LoRA is enabled. Edit freely after building — but "
            "note: any live auto-rebuild (driven by builder picks) will "
            "overwrite hand-typed additions.",
        ).pack(side="left")
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

        # ROW 2: Negative prompt (collapsible, collapsed)
        neg_outer = CollapsibleFrame(
            root, text="Negative prompt", start_open=False,
        )
        neg_outer.grid(row=2, column=0, sticky="we", pady=(0, PAD))
        neg_box = ttk.Frame(neg_outer.body, padding=PAD)
        neg_box.pack(fill="both", expand=True)
        neg_box.columnconfigure(1, weight=1)
        ttk.Label(neg_box, text="Preset:").grid(row=0, column=0, sticky="w")
        info_icon(
            neg_box,
            "Pre-built negative prompts. Default is 'Pony · minimal' which "
            "is just the low-quality score tags — community testing shows "
            "Pony often produces cleaner output with a light negative "
            "than an over-filtered one. Reach for heavier presets only if "
            "you're seeing a specific failure mode (anime drift, plastic "
            "skin, etc.). You can layer your own additions into the entry "
            "below.",
        ).grid(row=0, column=2, sticky="w")
        # Look up the Pony · minimal preset by label (not index) so the
        # default isn't coupled to list position — if the table is ever
        # reordered, this still finds the right preset.
        _PONY_MIN_LABEL = "Pony · minimal (score-only, recommended)"
        _pony_min_idx = next(
            (i for i, (lbl, _, _) in enumerate(NEGATIVE_PRESETS)
             if lbl == _PONY_MIN_LABEL),
            1,  # fallback index if the label ever changes
        )
        self.neg_preset_var = tk.StringVar(value=NEGATIVE_PRESETS[_pony_min_idx][0])
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
        # Smart-augment toggle — when on, _on_generate computes a context-
        # aware addendum (anti-blonde if blonde subject, anti-anime if
        # photoreal anchors present, etc.) and appends it to whatever's
        # in the entry. The base preset stays editable; the augment is
        # additive and recomputed per render.
        smart_row = ttk.Frame(neg_box)
        smart_row.grid(row=2, column=0, columnspan=3, sticky="we", pady=(PAD // 2, 0))
        # Default ON — most users get a meaningful quality bump from the
        # auto-augment, especially when picking a hair colour or an outdoor
        # scene where the base presets don't have specific anti-tags.
        self.smart_neg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            smart_row,
            text="Auto-augment based on prompt",
            variable=self.smart_neg_var,
        ).pack(side="left")
        info_icon(
            smart_row,
            "Inspect the prompt body and append context-specific anti-tags: "
            "blonde → suppress brunette/black hair; raw photo → suppress "
            "anime/3d render; outdoor → suppress indoor/bedroom; etc. The "
            "base preset above stays as-is — the augment is concatenated "
            "per render and shown in the live preview before sending.",
        ).pack(side="left")
        # Seed initial negative — default to Pony · minimal (score-only).
        # Per community consensus, Pony produces cleaner output with a
        # light negative; over-filtering (anti-anime + anti-plastic + ...)
        # tends to soften the image more than it helps.
        self.gui.negative_var.set(NEGATIVE_PRESETS[_pony_min_idx][1])

        # ROW 3: LoRA stack (always expanded — high-frequency interaction).
        lora_box = ttk.LabelFrame(root, text="LoRA stack", padding=PAD)
        lora_box.grid(row=3, column=0, sticky="we", pady=(0, PAD))
        lora_box.columnconfigure(0, weight=1)
        self._build_lora_block(lora_box)

        # Action row
        # NOTE: Generate button + progress bar live in the sticky action
        # bar above the scrollable region (see ``build()``). Don't add
        # them here — duplicating them would confuse users + break the
        # progress-tracking state that points at the action-bar widgets.

        # Live preview of assembled prompt as the user types/picks.
        # Also re-runs when the smart-aug toggle flips or the negative entry
        # changes, so the "→ smart neg:" line on the preview is always honest.
        self.prompt_text.bind("<KeyRelease>", lambda _e: self._refresh_assembled())
        self.gui.quality_stack_var.trace_add("write", lambda *_: self._refresh_assembled())
        self.smart_neg_var.trace_add("write", lambda *_: self._refresh_assembled())
        self.gui.negative_var.trace_add("write", lambda *_: self._refresh_assembled())
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

        # Photoreal tips — previously a 6-bullet Label that ate ~150 px of
        # vertical real estate every time the user opened the Generate tab.
        # Now collapsed to a single-line header + info_icon that opens a
        # popup with the same content on click. Wording unchanged.
        tips_row = ttk.Frame(root)
        tips_row.grid(row=3, column=0, sticky="we")
        ttk.Label(
            tips_row, text="Photoreal survival guide",
            style="Mono.TLabel",
        ).pack(side="left")
        info_icon(
            tips_row,
            "Getting anime output despite source_real? The BASE MODEL is "
            "the biggest lever. Switch from vanilla Pony V6 XL to Lustify "
            "XL or Pony Realism — both Pony fine-tunes but trained on "
            "photo data, so photoreal is the default not the fight.\n\n"
            "If you're stuck on vanilla Pony: use 'Pony · heavy photoreal "
            "(anti-anime)' quality stack AND 'NSFW · photoreal push' "
            "negative. Combined they produce photoreal output from "
            "vanilla Pony.\n\n"
            "CFG 5-6 is the photoreal sweet spot on Pony. High CFG = "
            "fried / plastic; low CFG = ignores the prompt. 5.5 works "
            "for most photoreal workflows.\n\n"
            "Add ONE photoreal LoRA (zy_Realism_Enhancer_v2 at ~0.35 is "
            "Pony-calibrated). Don't stack several — total extra-LoRA "
            "weight past ~1.0 degrades output reliably.\n\n"
            "Stack trained likeness LoRA at 1.0 + style LoRA at 0.3-0.5. "
            "Trained carries likeness; style carries photoreal feel.\n\n"
            "Lock the seed when iterating prompt wording — change one "
            "word at a time, see exactly what it did.",
        ).pack(side="left", padx=(6, 0))

    # ---- Prompt builder ----

    def _build_prompt_builder(self, root: ttk.Frame) -> None:
        """Lay out the structured prompt-builder hub.

        Sections, top to bottom:

        * Quality-tag prefix (auto-prepended to every render) + "Set as
          default" so the user's picked stack persists across launches.
        * Subject / Scene / Action dropdown groups (2-col grid each).
        * Sampler + dimensions + N + steps + guidance + seed + folder
          name — moved here from the old standalone collapsible because
          they materially shape the output and should sit next to the
          knobs that produced the prompt.
        * Build / Reset / Save defaults / Clear defaults buttons.

        After "Build prompt from picks" the user edits the body freely;
        the dropdowns don't auto-sync afterwards. "Save current picks as
        defaults" persists every dropdown + tattoos boolean to
        ``.user_settings.json`` so re-opening the app starts with the
        same configuration.
        """
        PAD = gui_theme.PAD
        root.columnconfigure(0, weight=1)

        # Intro paragraph (4 lines) collapsed into the Prompt-builder
        # CollapsibleFrame header's info icon — one hover reveals the
        # same guidance on demand. Saves ~60 px of always-visible text.
        _intro_row = ttk.Frame(root)
        _intro_row.grid(row=0, column=0, sticky="w", pady=(0, PAD // 2))
        info_icon(
            _intro_row,
            "Pick a quality stack + descriptors per axis. Picks update "
            "the prompt body LIVE — 'Build prompt from picks ▶' is a "
            "manual rebuild, used to reset after hand-editing. Anything "
            "left on '(any)' is omitted from the prompt. Sampler / "
            "dimensions / seed live below the picks since they shape "
            "the render alongside the prompt.",
        ).pack(side="left")
        ttk.Label(
            _intro_row, text="Picks auto-update the prompt",
            style="Status.TLabel",
        ).pack(side="left", padx=(4, 0))

        # --- quality-tag prefix (formerly its own collapsible) ---
        qs_box = ttk.LabelFrame(root, text="Quality-tag prefix", padding=PAD)
        qs_box.grid(row=1, column=0, sticky="we", pady=(0, PAD // 2))
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
        # "Set as default" button removed — auto-persist saves every field
        # (quality stack included) to .user_settings.json on change, so the
        # explicit save action is redundant. See the builder-defaults
        # button below which has the same redundancy.
        self.qs_hint_var = tk.StringVar(
            value=self._stack_hint(self.gui.quality_stack_var.get())
        )
        ttk.Label(
            qs_box, textvariable=self.qs_hint_var, style="Status.TLabel",
            wraplength=520, justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        qs_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_quality_stack_change(),
        )

        # --- three picker sub-sections ---
        subj = ttk.LabelFrame(root, text="Subject", padding=PAD)
        subj.grid(row=2, column=0, sticky="we", pady=(0, PAD // 2))
        self._build_builder_grid(subj, BUILDER_SUBJECT, "Subject",
                                 include_tattoos=True)

        scn = ttk.LabelFrame(root, text="Scene", padding=PAD)
        scn.grid(row=3, column=0, sticky="we", pady=(0, PAD // 2))
        self._build_builder_grid(scn, BUILDER_SCENE, "Scene")

        act = ttk.LabelFrame(root, text="Action", padding=PAD)
        act.grid(row=4, column=0, sticky="we", pady=(0, PAD // 2))
        self._build_builder_grid(act, BUILDER_ACTION, "Action")

        # --- sampler + dimensions (formerly its own collapsible) ---
        nums_box = ttk.LabelFrame(root, text="Sampler & dimensions", padding=PAD)
        nums_box.grid(row=5, column=0, sticky="we", pady=(0, PAD // 2))
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

        nums3 = ttk.Frame(nums_box)
        nums3.grid(row=2, column=0, sticky="we", pady=(PAD // 2, 0))
        ttk.Label(nums3, text="Folder name (optional):").pack(side="left")
        info_icon(
            nums3,
            "Optional friendly name for this run's output folder. Becomes "
            "outputs/<name>_<timestamp>/ — useful for grouping experiments "
            "('boudoir_test_v3', 'sampler_compare', etc.). Leave blank for "
            "the default outputs/<timestamp>/ format.",
        ).pack(side="left")
        ttk.Entry(nums3, textvariable=self.gui.output_name_var, width=22).pack(
            side="left", padx=(PAD // 2, 0),
        )

        # --- build + reset row ---
        # "Save picks as my defaults" / "Clear my defaults" removed: the
        # auto-persist layer saves every dropdown on change, so the explicit
        # save button was redundant. "Clear" is now done via Reset (which
        # sets every pick back to "(any)") followed by any edit that
        # re-triggers the auto-persist. Kept Build + Reset since those are
        # one-shot commands, not state saves.
        btn_row = ttk.Frame(root)
        btn_row.grid(row=6, column=0, sticky="we", pady=(PAD // 2, 0))
        ttk.Button(
            btn_row, text="Build prompt from picks ▶", style="Primary.TButton",
            command=self._on_build_from_picks,
        ).pack(side="left")
        ttk.Button(
            btn_row, text="Reset all picks", style="Ghost.TButton",
            command=self._on_reset_picks,
        ).pack(side="left", padx=(PAD // 2, 0))
        info_icon(
            btn_row,
            "Picks update the prompt LIVE — any dropdown or the tattoos "
            "checkbox rebuilds the prompt body instantly. 'Build prompt "
            "from picks ▶' is a manual rebuild (useful to reset after "
            "hand-editing the prompt text). Every pick + the prompt body "
            "auto-saves to .user_settings.json on change, so your "
            "selections survive app restarts automatically — no explicit "
            "'save' needed.",
        ).pack(side="left", padx=(PAD // 2, 0))

    def _build_builder_grid(
        self,
        parent: ttk.LabelFrame,
        options: dict[str, list[tuple[str, str]]],
        group_key: str,
        include_tattoos: bool = False,
    ) -> None:
        """Render a dropdown group as a 2-column grid of label+combobox pairs."""
        PAD = gui_theme.PAD
        # Each row holds two (label, combobox) pairs — one on the left half,
        # one on the right half — so the group is compact.
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)
        keys = list(options.keys())
        for i, key in enumerate(keys):
            r, c = divmod(i, 2)
            col_label = c * 2
            col_combo = c * 2 + 1
            ttk.Label(parent, text=f"{key}:").grid(
                row=r, column=col_label, sticky="w", padx=(0, PAD // 2), pady=2,
            )
            combo = ttk.Combobox(
                parent,
                textvariable=self.builder_vars[group_key][key],
                values=[label for label, _ in options[key]],
                state="readonly",
                width=22,
            )
            combo.grid(
                row=r, column=col_combo, sticky="we",
                padx=(0, PAD), pady=2,
            )

        # Tattoos checkbox — only rendered for the Subject group.
        if include_tattoos:
            next_row = (len(keys) + 1) // 2
            ttk.Checkbutton(
                parent, text="Tattoos visible",
                variable=self.builder_tattoos_var,
            ).grid(
                row=next_row, column=0, columnspan=2, sticky="w", pady=(PAD // 2, 0),
            )

    # ---- prompt-builder handlers ----

    def _on_build_from_picks(self) -> None:
        """Assemble the selected dropdown fragments into the prompt body.

        Order of concatenation: identity anchor → subject appearance →
        outfit → action → pose → expression → scene → lighting → camera
        angle. This mirrors the order Pony was trained on booru tag lists,
        which produces more reliable output than a random permutation.

        Subject-noun resolution: the first token comes from Subject
        count (if the user picked anything other than "(any)") or from
        Gender (otherwise). Gender's "(any)" fragment is "1girl", so a
        user who touches neither picker still gets the legacy behaviour.
        Subject count takes precedence because a plural like "2girls" is
        a stronger prompt signal than the singular from Gender.
        """
        parts: list[str] = []

        # Inject the project's trigger word if a trained LoRA will be used.
        # This keeps the builder consistent with the prompt-template flow.
        if self.gui.current_project and self.gui.use_trained_lora_var.get():
            trigger = (self.gui.current_project.trigger_word or "").strip()
            if trigger:
                parts.append(trigger)

        def pick_fragment(group: str, key: str, table: dict) -> str:
            label = self.builder_vars[group][key].get()
            for lbl, frag in table[key]:
                if lbl == label:
                    return frag.strip()
            return ""

        # Subject-noun anchor (Subject count wins when set, else Gender).
        count_frag = pick_fragment("Subject", "Subject count", BUILDER_SUBJECT)
        gender_frag = pick_fragment("Subject", "Gender", BUILDER_SUBJECT)
        if count_frag:
            parts.append(count_frag)
        elif gender_frag:
            parts.append(gender_frag)
        else:
            # Both picks landed on "(any)" AND returned empty fragments —
            # only happens if someone edits the table to remove the
            # default "1girl" fragment. Safety net.
            parts.append("1girl")

        # Remaining subject descriptors (hair / eyes / skin / body / etc.)
        # Skip the two noun-anchor keys we already resolved above.
        for key in BUILDER_SUBJECT:
            if key in ("Subject count", "Gender"):
                continue
            f = pick_fragment("Subject", key, BUILDER_SUBJECT)
            if f:
                parts.append(f)
        if self.builder_tattoos_var.get():
            parts.append("visible tattoos on body")

        # Outfit comes early because it gates what body tags make sense.
        out_f = pick_fragment("Action", "Outfit / clothing", BUILDER_ACTION)
        if out_f:
            parts.append(out_f)

        # Activity + pose + expression
        for key in ("Activity", "Pose", "Expression"):
            f = pick_fragment("Action", key, BUILDER_ACTION)
            if f:
                parts.append(f)

        # Scene / setting / lighting / camera angle
        for key in BUILDER_SCENE:
            f = pick_fragment("Scene", key, BUILDER_SCENE)
            if f:
                parts.append(f)

        # NOTE: a hard-coded `raw photo, real photograph` tail used to be
        # appended here. It's been removed — every Pony quality stack now
        # provides its own photo anchors (source_real for the default,
        # `photo, photorealistic, raw photo, ...` for the heavy stack),
        # so duplicating here was causing token collisions and preventing
        # users from producing illustrated / stylised output from the
        # builder. If your chosen stack is "(none)" and you want photo
        # output, type `raw photo, real photograph` yourself in the body
        # or pick a Pony stack.

        built = ", ".join(parts)
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", built)
        self._refresh_assembled()

    def _on_reset_picks(self) -> None:
        """Reset every dropdown back to '(any)' and clear tattoos."""
        for group, axes in self.builder_vars.items():
            table = {
                "Subject": BUILDER_SUBJECT,
                "Scene": BUILDER_SCENE,
                "Action": BUILDER_ACTION,
            }[group]
            for key, var in axes.items():
                # First entry in each options list is "(any)".
                var.set(table[key][0][0])
        self.builder_tattoos_var.set(False)

    # ---- LoRA library ----

    def _build_lora_block(self, root: ttk.LabelFrame) -> None:
        PAD = gui_theme.PAD
        root.columnconfigure(0, weight=1)

        # ── LIKENESS ────────────────────────────────────────────────────────
        ttk.Label(root, text="LIKENESS", style="Mono.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 3),
        )

        likeness_card = ttk.Frame(root)
        likeness_card.grid(row=1, column=0, sticky="we", pady=(0, PAD))
        likeness_card.columnconfigure(0, weight=1)

        top = ttk.Frame(likeness_card)
        top.pack(fill="x")
        ttk.Checkbutton(
            top,
            text="Use this project's trained LoRA",
            variable=self.gui.use_trained_lora_var,
            command=self._update_active_count,
        ).pack(side="left")
        info_icon(
            top,
            "Tick to apply the LoRA you trained for THIS project. Untick for "
            "vanilla text-to-image with just the base checkpoint — useful to "
            "sanity-check the base or compare 'with vs without LoRA'. "
            "Either way, you can still stack enhancement LoRAs below.",
        ).pack(side="left", padx=(4, 0))

        self.likeness_hint_var = tk.StringVar(value="")
        ttk.Label(
            likeness_card,
            textvariable=self.likeness_hint_var,
            style="Status.TLabel",
        ).pack(anchor="w", padx=(22, 0))

        self.gui.use_trained_lora_var.trace_add(
            "write", lambda *_: self._update_active_count()
        )

        # ── ENHANCEMENTS ────────────────────────────────────────────────────
        enh_header = ttk.Frame(root)
        enh_header.grid(row=2, column=0, sticky="we", pady=(0, 4))
        enh_header.columnconfigure(1, weight=1)

        ttk.Label(enh_header, text="ENHANCEMENTS", style="Mono.TLabel").grid(
            row=0, column=0, sticky="w",
        )
        self.lora_active_count_var = tk.StringVar(value="")
        ttk.Label(
            enh_header, textvariable=self.lora_active_count_var,
            style="Status.TLabel",
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        btn_row = ttk.Frame(enh_header)
        btn_row.grid(row=0, column=2, sticky="e")
        info_icon(
            btn_row,
            "Drop .safetensors LoRAs from Civitai into the shared library folder, "
            "then click ↺ Refresh. Tick to enable; drag the slider to set weight. "
            "Good starting stack: trained likeness @ 1.0 + skin-texture LoRA @ 0.6 "
            "+ photo-style LoRA @ 0.65.",
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            btn_row, text="Import…", style="Ghost.TButton",
            command=self.import_safetensors,
        ).pack(side="left", padx=(0, 2))
        ttk.Button(
            btn_row, text="Open folder", style="Ghost.TButton",
            command=self.open_library,
        ).pack(side="left", padx=(0, 2))
        ttk.Button(
            btn_row, text="↺", style="Ghost.TButton",
            command=self.refresh_lora_list,
        ).pack(side="left")

        self.lora_table = ttk.Frame(root)
        self.lora_table.grid(row=3, column=0, sticky="we")
        self.refresh_lora_list()

    def refresh_lora_list(self) -> None:
        if self.lora_table is None:
            return
        for child in self.lora_table.winfo_children():
            child.destroy()

        # Refresh likeness hint now that we know the current project.
        self._refresh_likeness_hint()

        files = gui_helpers.list_shared_loras(self.gui.projects_root.root)
        if not files:
            empty = ttk.Frame(self.lora_table)
            empty.pack(fill="x", pady=4)
            ttk.Label(empty, text="No enhancement LoRAs yet.",
                      style="Mono.TLabel").pack(anchor="w")
            lib_path = gui_helpers.shared_loras_dir(self.gui.projects_root.root)
            ttk.Label(
                empty,
                text=(
                    f"Drop .safetensors files into:\n{lib_path}\n"
                    "then click ↺ — or use 'Import…' to copy files in directly.\n"
                    "Suggested first download: a skin-texture LoRA from Civitai "
                    "(search 'realistic skin XL'). Stack at 0.6 over your likeness LoRA."
                ),
                style="Status.TLabel", wraplength=520, justify="left",
            ).pack(anchor="w", pady=(2, 0))
            self._update_active_count()
            return

        new_rows: dict[str, dict] = {}
        PAD = gui_theme.PAD

        for path in files:
            prior = self.lora_rows.get(path.stem, {})
            selected = prior.get("selected_var") or tk.BooleanVar(value=False)
            category, default_weight, weight_hint = self._categorize_lora(path.stem)
            weight_var = prior.get("weight_var") or tk.StringVar(value=default_weight)

            # ── card ───────────────────────────────────────────────────────
            # Single-row layout. Columns: [0]checkbox [1]filename
            # [2]category badge [3]info-icon [4]filler [5]weight Spinbox.
            #
            # Slider replaced by Spinbox per user request — sliders made it
            # hard to set precise weights (e.g. exactly 0.35), and the
            # Spinbox gives both ↑↓ buttons for nudging and direct typing.
            # Range 0.00–2.00, increment 0.05 — fine enough for any
            # practical LoRA-mixing workflow.
            card = ttk.Frame(self.lora_table)
            card.pack(fill="x", pady=1)
            card.columnconfigure(4, weight=1)  # absorb slack between badge + spinbox

            ttk.Checkbutton(
                card, variable=selected,
                command=self._update_active_count,
            ).grid(row=0, column=0, sticky="w")

            display_name = path.name
            if len(display_name) > 38:
                display_name = display_name[:35] + "…"
            ttk.Label(
                card, text=display_name, style="Mono.TLabel",
            ).grid(row=0, column=1, sticky="w", padx=(4, 8))

            ttk.Label(
                card, text=f"[{category}]", style="Status.TLabel",
            ).grid(row=0, column=2, sticky="w", padx=(0, 2))

            info_icon(
                card,
                f"Category: {category}\n"
                f"Recommended weight range: {weight_hint}\n"
                f"Default on first tick: {default_weight}\n\n"
                f"Every LoRA's weight feeds into a total-weight ceiling — "
                f"keep total extra-LoRA weight under ~1.0 on Pony bases. "
                f"Past that, output reliably degrades (skin waxiness, soft "
                f"focus, over-baked LoRA artefacts).",
            ).grid(row=0, column=3, sticky="w", padx=(0, 6))

            # Normalise the saved string to a 2-decimal form so the
            # Spinbox starts in a clean state. Tolerate junk values from
            # historical saves by falling back to the category default.
            try:
                weight_var.set(f"{float(weight_var.get()):.2f}")
            except ValueError:
                weight_var.set(f"{float(default_weight):.2f}")

            ttk.Spinbox(
                card,
                from_=0.0, to=2.0, increment=0.05,
                textvariable=weight_var,
                width=6,
                format="%.2f",
                wrap=False,
                justify="right",
            ).grid(row=0, column=5, sticky="e", padx=(0, 2))

            new_rows[path.stem] = {
                "path": path,
                "selected_var": selected,
                "weight_var": weight_var,
            }

        self.lora_rows = new_rows
        self._update_active_count()
        # New row StringVars need persistence traces too. `_install_…` is
        # idempotent so we can call it unconditionally on every refresh.
        # Guarded so the very first refresh (before traces are set up at
        # bootstrap) is a no-op — bootstrap calls _install later.
        if hasattr(self, "_traced_vars"):
            self._install_persistence_traces()
            # The visible LoRA stack may have changed (file added/removed);
            # capture the new state.
            self._schedule_persist()

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

    # ---- LoRA helpers ----

    @staticmethod
    def _categorize_lora(stem: str) -> tuple[str, str, str]:
        """Return (category_label, default_weight, weight_hint) for a LoRA.

        Matches keywords against the filename stem (case-insensitive). First
        matching rule wins; falls back to a generic STYLE entry.
        """
        s = stem.lower()
        for keywords, label, default_w, hint in _LORA_CATEGORIES:
            if any(kw in s for kw in keywords):
                return label, default_w, hint
        # Fallback for LoRAs whose filename doesn't match any category.
        # 0.45 is conservative for Pony — user can raise it in the slider
        # once they know what the LoRA does. The old 0.60 default tended
        # to put total stack weight over the ~1.0 ceiling when combined
        # with a primary realism LoRA.
        return "STYLE", "0.45", "0.3 – 0.6"

    def _refresh_likeness_hint(self) -> None:
        """Update the sub-line under the trained-LoRA toggle.

        Shows the project's trigger word and whether a trained LoRA is
        already present in ``lora/``. Called from ``refresh_lora_list``
        so it stays current whenever the user switches projects or
        finishes a training run.
        """
        if not hasattr(self, "likeness_hint_var"):
            return
        proj = self.gui.current_project
        if not proj:
            self.likeness_hint_var.set("load a project first")
            return
        trigger = (proj.trigger_word or "").strip() or "—"
        lora_ready = (
            proj.lora_dir.exists()
            and any(f for f in proj.lora_dir.iterdir() if f.is_file())
        )
        status = "trained ✓" if lora_ready else "not trained yet — run Train first"
        self.likeness_hint_var.set(f"trigger word: {trigger}  ·  {status}")

    def _update_active_count(self) -> None:
        """Refresh the 'N of M active' badge in the enhancements header.

        Called whenever a checkbox is toggled or the LoRA list is rebuilt.
        Also reflected in the assembled-prompt preview via _refresh_assembled.
        """
        if not hasattr(self, "lora_active_count_var"):
            return
        total = len(self.lora_rows)
        active = sum(1 for r in self.lora_rows.values() if r["selected_var"].get())
        if total == 0:
            self.lora_active_count_var.set("(none in library)")
        elif active == 0:
            self.lora_active_count_var.set(f"(0 of {total} active)")
        else:
            self.lora_active_count_var.set(f"({active} of {total} active)")
        # Propagate to the assembled-prompt preview so it reflects LoRA state.
        self._refresh_assembled()

    # ---- preset handlers ----

    def _on_quality_stack_change(self) -> None:
        label = self.gui.quality_stack_var.get()
        self.qs_hint_var.set(self._stack_hint(label))
        self._refresh_assembled()

    @staticmethod
    def _stack_hint(label: str) -> str:
        """Return the hover-hint text for a given quality-stack label."""
        for name, _prefix, hint in QUALITY_STACKS:
            if name == label:
                return hint
        return ""

    def _on_save_default_stack(self) -> None:
        """Persist the currently-selected quality stack as this user's default.

        Writes to ``<projects_root>/.user_settings.json`` so the choice
        survives restarts and applies to every project.
        """
        label = self.gui.quality_stack_var.get()
        gui_helpers.update_user_setting(
            self.gui.projects_root.root, "default_quality_stack", label,
        )
        self.gui.status_var.set(f"Default quality stack saved: {label}")
        self.gui.log_queue.put(
            f"[default quality stack saved: {label}]\n"
        )

    def _on_neg_preset_change(self) -> None:
        label = self.neg_preset_var.get()
        for name, body, _hint in NEGATIVE_PRESETS:
            if name == label:
                self.gui.negative_var.set(body)
                break
        self._refresh_assembled()

    # ---- builder defaults (persistence) ----

    def _on_save_builder_defaults(self) -> None:
        """Persist every builder dropdown + the tattoos boolean to
        ``.user_settings.json`` under ``"default_builder_picks"``.

        Stored shape: ``{"Subject": {<key>: <label>, ...}, "Scene": ...,
        "Action": ..., "tattoos": bool}``. We persist the picker LABELS
        (not the underlying tag fragments) so a future preset edit
        doesn't corrupt the saved snapshot.
        """
        snapshot: dict[str, object] = {}
        for group, axes in self.builder_vars.items():
            snapshot[group] = {key: var.get() for key, var in axes.items()}
        snapshot["tattoos"] = bool(self.builder_tattoos_var.get())
        gui_helpers.update_user_setting(
            self.gui.projects_root.root, "default_builder_picks", snapshot,
        )
        self.gui.status_var.set("Builder defaults saved")
        self.gui.log_queue.put("[builder defaults saved to .user_settings.json]\n")

    def _on_clear_builder_defaults(self) -> None:
        """Wipe the persisted builder snapshot AND reset every dropdown to
        ``(any)``. The next launch will start empty again."""
        # Setting the value to None via update_user_setting clears the key
        # in our settings helper.
        gui_helpers.update_user_setting(
            self.gui.projects_root.root, "default_builder_picks", None,
        )
        self._on_reset_picks()
        self.gui.status_var.set("Builder defaults cleared")
        self.gui.log_queue.put("[builder defaults cleared]\n")

    def _apply_persisted_builder_defaults(self) -> None:
        """If the user previously saved builder picks, restore them now.

        Called once from the post-init bootstrap. Lenient: any axis whose
        saved label no longer exists (because we renamed/removed a preset)
        is silently skipped — the dropdown stays on ``(any)``.

        Uses the same ``_restoring`` guard as the generate-defaults
        restore, so the builder→prompt auto-rebuild trace (installed later
        in the bootstrap) sees the flag and doesn't clobber the prompt
        text while we're replaying saved picks.
        """
        settings = gui_helpers.load_user_settings(self.gui.projects_root.root)
        snapshot = settings.get("default_builder_picks")
        if not isinstance(snapshot, dict):
            return
        tables = {
            "Subject": BUILDER_SUBJECT,
            "Scene": BUILDER_SCENE,
            "Action": BUILDER_ACTION,
        }
        self._restoring = True
        try:
            for group, axes in self.builder_vars.items():
                saved_group = snapshot.get(group, {})
                if not isinstance(saved_group, dict):
                    continue
                for key, var in axes.items():
                    saved_label = saved_group.get(key)
                    if not isinstance(saved_label, str):
                        continue
                    # Only set if that label is still a valid choice.
                    valid_labels = [lbl for lbl, _ in tables[group][key]]
                    if saved_label in valid_labels:
                        var.set(saved_label)
            self.builder_tattoos_var.set(bool(snapshot.get("tattoos", False)))
        finally:
            self._restoring = False

    # ---- global auto-persistence for the Generate tab ----
    #
    # The builder defaults above have their own dedicated save/clear buttons
    # because users want an explicit "take a snapshot of my picks" action for
    # them. Everything ELSE on the Generate tab (prompt, negative, inference
    # knobs, LoRA stack, etc.) auto-saves — there's no Save button, values
    # just persist as you type. Global: one file at <projects_root>/
    # .user_settings.json under the key "generate_defaults", shared across
    # every project.
    #
    # Flow:
    #   1. On tab init: load the snapshot and apply it before installing
    #      traces, so the restore itself doesn't trigger a redundant write.
    #   2. During normal use: every `trace_add("write", ...)` fires into
    #      ``_schedule_persist`` which debounces rapid bursts (keystrokes)
    #      to one write every 400 ms.
    #   3. When `refresh_lora_list` rebuilds the LoRA row StringVars, we
    #      re-install traces on the new vars (idempotent — old tracked
    #      var IDs are remembered so we don't double-register).

    def _snapshot_generate_defaults(self) -> dict:
        """Capture every Generate-tab value in a JSON-safe dict.

        Extras are saved by LoRA **stem** (not absolute path) so moving the
        shared_loras folder, or renaming a file, doesn't corrupt restore —
        the worst case is a missing stem which is silently skipped.
        """
        extras = []
        for stem, row in self.lora_rows.items():
            if row["selected_var"].get():
                extras.append({
                    "stem": stem,
                    "weight": row["weight_var"].get(),
                })
        return {
            "prompt":           self.gui.prompt_var.get(),
            "negative":         self.gui.negative_var.get(),
            "n":                self.gui.n_var.get(),
            "steps":            self.gui.steps_var.get(),
            "guidance":         self.gui.guidance_var.get(),
            "seed":             self.gui.seed_var.get(),
            "sampler":          self.gui.sampler_var.get(),
            "aspect":           self.gui.aspect_var.get(),
            "output_name":      self.gui.output_name_var.get(),
            "use_trained_lora": bool(self.gui.use_trained_lora_var.get()),
            "quality_stack":    self.gui.quality_stack_var.get(),
            "extras":           extras,
        }

    def _persist_generate_defaults(self) -> None:
        """Write the snapshot to .user_settings.json.

        Skips during an active restore pass (so applying the persisted
        state doesn't trigger a recursive save). Swallows tk + OS errors
        because this runs in the background and must never crash the UI.
        """
        if getattr(self, "_restoring", False):
            return
        try:
            snap = self._snapshot_generate_defaults()
        except tk.TclError:
            # Window being torn down — tk vars are becoming inaccessible.
            return
        gui_helpers.update_user_setting(
            self.gui.projects_root.root, "generate_defaults", snap,
        )

    def _schedule_persist(self, *_tk_args) -> None:
        """Debounced entrypoint for trace_add callbacks.

        Tk fires one trace per character while the user types in an Entry.
        Coalescing to a single disk write every 400 ms is plenty and keeps
        the settings file quiet during fast edits.
        """
        try:
            self.gui.root.after_cancel(self._persist_after_id)
        except (AttributeError, ValueError, tk.TclError):
            pass
        try:
            self._persist_after_id = self.gui.root.after(
                400, self._persist_generate_defaults
            )
        except (AttributeError, tk.TclError):
            # Before mainloop or after teardown — best-effort only.
            pass

    def _apply_persisted_generate_defaults(self) -> None:
        """Restore every Generate-tab field from .user_settings.json.

        Lenient: any missing or malformed field falls back to the built-in
        default (whatever the widget was initialised with). LoRAs whose
        file has been deleted since the snapshot are silently dropped so
        the user doesn't hit a generate-time FileNotFoundError.
        """
        settings = gui_helpers.load_user_settings(self.gui.projects_root.root)
        snap = settings.get("generate_defaults")
        if not isinstance(snap, dict):
            return
        self._restoring = True
        try:
            def _set_str(var: "tk.Variable", key: str) -> None:
                v = snap.get(key)
                if isinstance(v, str):
                    var.set(v)

            _set_str(self.gui.prompt_var,        "prompt")
            _set_str(self.gui.negative_var,      "negative")
            _set_str(self.gui.n_var,             "n")
            _set_str(self.gui.steps_var,         "steps")
            _set_str(self.gui.guidance_var,      "guidance")
            _set_str(self.gui.seed_var,          "seed")
            _set_str(self.gui.sampler_var,       "sampler")
            _set_str(self.gui.aspect_var,        "aspect")
            _set_str(self.gui.output_name_var,   "output_name")
            _set_str(self.gui.quality_stack_var, "quality_stack")

            utl = snap.get("use_trained_lora")
            if isinstance(utl, bool):
                self.gui.use_trained_lora_var.set(utl)

            # Clear every current row first — stems dropped since last save
            # should go unchecked, not stay stale-checked.
            saved_extras = snap.get("extras") or []
            if isinstance(saved_extras, list):
                for row in self.lora_rows.values():
                    row["selected_var"].set(False)
                for entry in saved_extras:
                    if not isinstance(entry, dict):
                        continue
                    stem = entry.get("stem")
                    weight = entry.get("weight")
                    row = self.lora_rows.get(stem)
                    if row is None:
                        continue  # file deleted since last save — skip.
                    row["selected_var"].set(True)
                    if isinstance(weight, str):
                        # Normalise to 2-decimal form so Spinbox displays
                        # cleanly (e.g. saved "0.35" stays "0.35", but a
                        # raw "0.350000" gets tidied to "0.35").
                        try:
                            row["weight_var"].set(f"{float(weight):.2f}")
                        except ValueError:
                            row["weight_var"].set(weight)
                self._update_active_count()
        finally:
            self._restoring = False

    def _install_persistence_traces(self) -> None:
        """Attach write-traces to every persistable variable.

        Idempotent: remembers the id() of every var we've already traced
        so a re-install after ``refresh_lora_list`` doesn't stack multiple
        callbacks on the same var. Called on init AND whenever the LoRA
        row list is rebuilt, since row StringVars are recreated in place.
        """
        watched: list = [
            self.gui.prompt_var,   self.gui.negative_var,
            self.gui.n_var,        self.gui.steps_var,
            self.gui.guidance_var, self.gui.seed_var,
            self.gui.sampler_var,  self.gui.aspect_var,
            self.gui.output_name_var,
            self.gui.use_trained_lora_var,
            self.gui.quality_stack_var,
        ]
        for row in self.lora_rows.values():
            watched.append(row["selected_var"])
            watched.append(row["weight_var"])

        if not hasattr(self, "_traced_vars"):
            self._traced_vars = set()
        for var in watched:
            vid = id(var)
            if vid in self._traced_vars:
                continue
            try:
                var.trace_add("write", self._schedule_persist)
            except tk.TclError:
                continue
            self._traced_vars.add(vid)

    def _install_builder_autoupdate_traces(self) -> None:
        """Live-rebuild the prompt body whenever a builder pick changes.

        Every Subject / Scene / Action dropdown + the Tattoos checkbox
        fires the same rebuild path the ``Build prompt from picks`` button
        uses. The button still works (and is useful as a "force rebuild"
        when the user has typed manual edits and wants to reset), but it's
        no longer required for the builder to do its job.

        Caveat: a rebuild completely overwrites the current prompt-body
        Text widget. Any manual edits the user has typed into the body
        are lost the moment they touch another pick. This is the explicit
        behaviour the user asked for; flagging it here so future edits
        know the trade-off.

        Skipped during a ``_restoring`` pass (restore of either builder
        picks or generate defaults), so replaying saved state at launch
        doesn't also stomp on the restored prompt text.
        """
        def _on_any_pick_change(*_tk_args: object) -> None:
            if getattr(self, "_restoring", False):
                return
            self._on_build_from_picks()

        watched: list = []
        for axes in self.builder_vars.values():
            for var in axes.values():
                watched.append(var)
        watched.append(self.builder_tattoos_var)

        if not hasattr(self, "_builder_traced_vars"):
            self._builder_traced_vars = set()
        for var in watched:
            vid = id(var)
            if vid in self._builder_traced_vars:
                continue
            try:
                var.trace_add("write", _on_any_pick_change)
            except tk.TclError:
                continue
            self._builder_traced_vars.add(vid)

    # ---- smart negatives ----

    def _smart_negative_for(self, body: str) -> str:
        """Compute a context-aware negative-prompt augment based on the
        positive prompt body.

        Returns a comma-separated string of additional anti-tags to append
        to the user's chosen negative preset. Returns "" when nothing
        triggers — never injects useless tokens.

        Triggers (all case-insensitive substring checks against ``body``):

        * Hair-colour exclusion — picking a hair colour suppresses the
          opposite colours so the model commits.
        * Photoreal anchors (``raw photo``, ``photorealistic``,
          ``35mm``, ``dslr``, ``film``) → suppress anime / 3d / cgi.
        * Outdoor anchors (``outdoor``, ``beach``, ``forest``, ``street``)
          → suppress indoor scene tags so the model doesn't drift back
          to a bedroom.
        * Indoor anchors (``bedroom``, ``bathroom``, ``kitchen``) →
          suppress outdoor / landscape tags.
        * No tattoos pick → suppress tattoos so a clean-skin LoRA isn't
          overridden by a base-model bias.
        * Nude / topless → suppress underwear / bra / clothes (so the
          model commits to the nudity you asked for).
        * SFW outfit (cocktail dress, sundress, casual outfit) →
          suppress nude tags when the user explicitly wants clothing.
        """
        b = body.lower()
        adds: list[str] = []

        # ---- hair colour exclusion ----
        # Map: trigger substring → tags that compete with it.
        hair_rules = [
            (("blonde", "blond "), "brunette, dark hair, black hair, brown hair, raven hair, redhead"),
            (("brunette", "brown hair", "chestnut", "auburn"),
             "blonde, blond, platinum hair, white hair, ginger, redhead"),
            (("redhead", "red hair", "ginger", "copper"),
             "blonde, brunette, black hair, white hair"),
            (("black hair", "raven", "jet black"),
             "blonde, brunette, brown hair, white hair, grey hair"),
            (("white hair", "silver hair", "platinum"),
             "brunette, black hair, brown hair, raven hair"),
            (("pink hair", "pastel pink"),
             "blonde, brunette, black hair, brown hair"),
            (("blue hair",),
             "blonde, brunette, black hair, brown hair"),
            (("purple hair", "lavender hair"),
             "blonde, brunette, black hair, brown hair"),
        ]
        seen_hair = False
        for triggers, tags in hair_rules:
            if any(t in b for t in triggers):
                adds.append(tags)
                seen_hair = True
                break  # one hair rule wins — don't stack contradictions

        # ---- photoreal vs anime ----
        photoreal_anchors = (
            "raw photo", "photorealistic", "photoreal", "35mm",
            "dslr", "film grain", "kodak", "leica", "real photograph",
            "professional photograph", "skin pores", "subsurface scattering",
        )
        if any(a in b for a in photoreal_anchors):
            adds.append(
                "anime, manga, illustration, drawing, cartoon, "
                "3d render, cgi, render, sketch, painting, "
                "stylized, cel shading, anime screenshot, hentai art"
            )

        # ---- scene direction ----
        outdoor_anchors = (
            "outdoor", "outside", "beach", "forest", "street",
            "park", "rooftop", "garden", "poolside", "balcony",
        )
        indoor_anchors = (
            "bedroom", "bathroom", "kitchen", "living room",
            "studio", "bed,", "in bed", "shower", "bathtub",
        )
        if any(a in b for a in outdoor_anchors):
            adds.append("indoor, bedroom, studio, interior, closed room")
        elif any(a in b for a in indoor_anchors):
            adds.append("outdoor, landscape, exterior, sky, clouds, trees")

        # ---- tattoos opt-out ----
        # If the user did NOT explicitly include tattoos, push against any
        # base-model tendency to add them on NSFW / OnlyFans-style prompts.
        if "tattoo" not in b:
            adds.append("tattoo, tattoos, body ink")

        # ---- nudity commit ----
        nude_anchors = ("fully nude", "naked", "topless", "bottomless", "bare breasts")
        if any(a in b for a in nude_anchors):
            adds.append(
                "underwear, bra, panties, swimsuit, clothed, "
                "fully clothed, censor bar, mosaic censor, pixelated"
            )

        # ---- clothed commit ----
        clothed_anchors = (
            "cocktail dress", "sundress", "casual outfit", "business suit",
            "schoolgirl", "nurse outfit", "french maid", "workout clothes",
            "sleepwear", "lingerie",
        )
        if any(a in b for a in clothed_anchors):
            adds.append("nude, naked, fully nude, exposed breasts, topless")

        # Dedupe across rule outputs while preserving order — the last
        # overlapping tag wins.
        seen: set[str] = set()
        result: list[str] = []
        for chunk in adds:
            for tag in (t.strip() for t in chunk.split(",")):
                if not tag or tag in seen:
                    continue
                seen.add(tag)
                result.append(tag)
        return ", ".join(result)

    def _final_negative(self) -> str:
        """Compose the negative prompt actually sent to the pipeline.

        = base preset/entry + (if smart-aug enabled) the context augment.
        Used by both _on_generate and _on_compare_stacks.
        """
        base = self.gui.negative_var.get().strip()
        if not getattr(self, "smart_neg_var", None) or not self.smart_neg_var.get():
            return base
        body = self.prompt_text.get("1.0", "end").strip()
        aug = self._smart_negative_for(body)
        if not aug:
            return base
        if not base:
            return aug
        return f"{base}, {aug}"

    def _refresh_assembled(self) -> None:
        if not hasattr(self, "prompt_text") or not hasattr(self, "assembled_var"):
            return
        body = self.prompt_text.get("1.0", "end").strip()
        prefix = ""
        for name, p, _hint in QUALITY_STACKS:
            if name == self.gui.quality_stack_var.get():
                prefix = p
                break
        full = f"{prefix}{body}".strip()
        if not full:
            self.assembled_var.set("(empty prompt)")
            return
        display = full if len(full) <= 240 else full[:237] + "…"

        # Smart-augment line — context-aware negative additions.
        aug_line = ""
        if getattr(self, "smart_neg_var", None) and self.smart_neg_var.get():
            aug = self._smart_negative_for(body)
            if aug:
                aug_short = aug if len(aug) <= 200 else aug[:197] + "…"
                aug_line = f"\n→ smart neg: {aug_short}"

        # LoRA stack summary — shows exactly what will be loaded.
        lora_parts: list[str] = []
        if self.gui.use_trained_lora_var.get():
            lora_parts.append("likeness @ 1.0")
        for r in self.lora_rows.values():
            if r["selected_var"].get():
                stem = r["path"].stem
                label = stem[:22] + "…" if len(stem) > 22 else stem
                weight = r["weight_var"].get()
                lora_parts.append(f"{label} @ {weight}")
        lora_line = (
            f"\n→ LoRAs: {', '.join(lora_parts)}" if lora_parts else "\n→ LoRAs: none"
        )

        self.assembled_var.set(f"→ sending: {display}{aug_line}{lora_line}")

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
                "Type a description in the Prompt body field, or pick "
                "options in the Prompt builder above and click "
                "'Build prompt from picks ▶' to assemble one.",
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
            # Fallback aligned with the init default and the compare-stacks
            # path (both 5.5). The previous 6.5 fallback caused silent
            # over-CFG if the user cleared the field mid-session — high
            # CFG reliably fries skin on Pony + realism-LoRA combos.
            "--guidance", self.gui.guidance_var.get() or "5.5",
            "--width", str(width),
            "--height", str(height),
            "--sampler", self.gui.sampler_var.get() or "default",
        ]
        neg = self._final_negative()
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

    # ---- run_info.txt loader ----

    def _on_load_run_info(self) -> None:
        """File-picker → parse a previous run's run_info.txt → apply
        every settable field to the current Generate tab.

        Matches whatever a previous render saved next to its PNGs:
        sampler, steps, guidance (CFG), dimensions, prompt body,
        negative prompt, and the extra-LoRA list with weights. The
        LoRA list is matched by filename stem against the current
        shared_loras/ folder; missing files are silently skipped.

        Useful for: 'I liked output X — give me the same settings to
        iterate from' workflow, or restoring a config after the GUI
        has been changed.
        """
        path = filedialog.askopenfilename(
            title="Pick a run_info.txt to load settings from",
            filetypes=[("run_info", "run_info*.txt"), ("Text files", "*.txt"),
                       ("All files", "*.*")],
            initialdir=str(
                self.gui.current_project.outputs_dir
                if self.gui.current_project else Path.home()
            ),
        )
        if not path:
            return
        try:
            self._apply_run_info_file(Path(path))
        except Exception as e:
            messagebox.showerror(
                "Couldn't load run_info",
                f"Failed to parse {Path(path).name}:\n\n{e}",
            )

    def _apply_run_info_file(self, p: Path) -> None:
        """Parse a run_info.txt and apply its settings.

        Format reminder (from `_write_run_info` in pipeline/generate.py):
            sampler           : <name>
            steps             : <int>
            guidance (CFG)    : <float>
            dimensions        : <W> x <H>
            extra LoRAs       :
              - <abs path>  @ weight <float>
              - ...

            prompt
            ------
            <multiline prompt>

            negative
            --------
            <multiline negative>
        """
        text = p.read_text(encoding="utf-8")
        info = _parse_run_info_text(text)

        # ---- Temporarily suppress auto-rebuild + auto-save while we
        # bulk-apply, then refresh once at the end.
        self._restoring = True
        try:
            if info.get("prompt") is not None:
                self.prompt_text.delete("1.0", "end")
                self.prompt_text.insert("1.0", info["prompt"])
            if info.get("negative") is not None:
                self.gui.negative_var.set(info["negative"])
            if info.get("sampler"):
                self.gui.sampler_var.set(info["sampler"])
            if info.get("steps"):
                self.gui.steps_var.set(str(info["steps"]))
            if info.get("guidance"):
                self.gui.guidance_var.set(str(info["guidance"]))
            # Match width × height to the closest aspect-ratio preset.
            if info.get("width") and info.get("height"):
                w, h = info["width"], info["height"]
                for label, lw, lh in ASPECT_RATIOS:
                    if lw == w and lh == h:
                        self.gui.aspect_var.set(label)
                        break

            # LoRA stack: match by filename stem against currently-known
            # rows. Untick everything first so missing files = unticked.
            for row in self.lora_rows.values():
                row["selected_var"].set(False)
            applied: list[str] = []
            missing: list[str] = []
            for path_str, weight in info.get("extra_loras") or []:
                stem = Path(path_str).stem
                row = self.lora_rows.get(stem)
                if row is None:
                    missing.append(stem)
                    continue
                row["selected_var"].set(True)
                try:
                    row["weight_var"].set(f"{float(weight):.2f}")
                except (TypeError, ValueError):
                    row["weight_var"].set(str(weight))
                applied.append(stem)

            self._update_active_count()
        finally:
            self._restoring = False

        # Force a single refresh + save AFTER the bulk apply completes.
        self._refresh_assembled()
        self._schedule_persist()

        msg_lines = [f"Loaded run_info: {p.name}"]
        if applied:
            msg_lines.append(f"  Applied LoRAs: {', '.join(applied)}")
        if missing:
            msg_lines.append(f"  Skipped (not in shared_loras/): {', '.join(missing)}")
        self.gui.log_queue.put("\n".join(msg_lines) + "\n")
        self.gui.status_var.set("run_info applied")

    def _on_compare_stacks(self) -> None:
        """Render ONE image per defined quality stack with the same body
        + seed so the user can A/B/C the stacks side-by-side.

        Sends the prompt BODY only (no quality prefix) — the CLI iterates
        through every stack and prepends each prefix. Useful for picking
        which stack to keep as your default.
        """
        project = self.gui.require_project()
        if not project:
            return
        self.gui.save_settings_silent()

        # The body alone — DO NOT add the currently-selected quality stack
        # prefix; the CLI will iterate every stack itself.
        body = self.prompt_text.get("1.0", "end").strip()
        if not body:
            messagebox.showerror(
                "Prompt is empty",
                "Type a prompt body before running compare-stacks. The "
                "comparison only varies the quality stack — your body has "
                "to describe the actual subject + scene.",
            )
            return

        # Stack count drives image count for the progress bar.
        from ..prompt_presets import stacks_for_compare
        n_stacks = len(stacks_for_compare())

        ok = messagebox.askokcancel(
            "Compare across all quality stacks?",
            f"Will render {n_stacks} images — one per defined quality stack — "
            f"using the same prompt body + seed. Output goes to "
            f"outputs/stack_compare_<timestamp>/ with each file labelled by "
            f"the stack that produced it. Total time: roughly "
            f"{n_stacks * 15} seconds at your throughput.\n\n"
            f"Tip: lock a seed first (Sampler & dimensions section) so the "
            f"only variable between outputs is the stack itself.",
        )
        if not ok:
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
            "--prompt", body,                          # BODY ONLY in compare mode
            "--steps", self.gui.steps_var.get() or "28",
            "--guidance", self.gui.guidance_var.get() or "5.5",
            "--width", str(width),
            "--height", str(height),
            "--sampler", self.gui.sampler_var.get() or "default",
            "--compare-stacks",
        ]
        # Use the currently-selected negative preset, plus smart-aug if on.
        neg = self._final_negative()
        if neg:
            args += ["--negative", neg]
        # Honour the seed if set so reruns are bit-identical.
        seed = self.gui.seed_var.get().strip()
        if seed:
            args += ["--seed", seed]
        if not self.gui.use_trained_lora_var.get():
            args.append("--no-trained-lora")
        for path, weight in self.selected_extras():
            args += ["--extra-lora", f"{path}:{weight}"]

        self._begin_run(n_stacks)
        self.gui.on_next_exit = self._end_run
        self.gui.spawn(args)

    # ---- Compare LoRA recipes ----

    def _on_compare_loras(self) -> None:
        """Open the LoRA-compare dialog and run the chosen comparison.

        Renders one image per (recipe × stack) combo with a shared seed.
        Recipes are derived from the dialog's mode pick:
        - "each_solo"  : each currently-checked LoRA active alone, plus
                         a no-LoRAs baseline and an all-together render.
        - "weight_sweep": one LoRA, multiple weights, others held fixed.
        - "powerset"   : every subset of currently-checked LoRAs.

        Optionally cross-products with a user-selected stack subset so
        you can see how each recipe interacts with each stack prefix.
        """
        project = self.gui.require_project()
        if not project:
            return
        self.gui.save_settings_silent()

        body = self.prompt_text.get("1.0", "end").strip()
        if not body:
            messagebox.showerror(
                "Prompt is empty",
                "Type a prompt body before running compare-LoRAs. The "
                "comparison varies the LoRA stack and (optionally) the "
                "quality stack — your body has to describe the subject.",
            )
            return

        checked = self.selected_extras()
        if not checked and not self.gui.use_trained_lora_var.get():
            messagebox.showerror(
                "Nothing to compare",
                "Tick at least one LoRA (or enable the trained likeness "
                "LoRA) before opening Compare LoRAs. The comparison "
                "varies the active set.",
            )
            return

        dlg = _LoraCompareDialog(
            self.gui.root,
            checked_loras=checked,
            use_trained=bool(self.gui.use_trained_lora_var.get()),
            current_stack=self.gui.quality_stack_var.get(),
        )
        result = dlg.show()
        if result is None:
            return  # user cancelled

        recipes, stack_labels, use_trained_in_every = result

        # Serialise recipes + stacks to a tiny JSON file the CLI can
        # read. argv is too cramped for nested data; JSON keeps the
        # contract clear and lets us add fields later without breaking
        # the CLI signature.
        import json
        import tempfile
        payload = {
            "recipes": [
                {
                    "label": label,
                    "loras": [
                        {"path": str(p), "weight": float(w)}
                        for p, w in r_loras
                    ],
                }
                for label, r_loras in recipes
            ],
            "stacks": stack_labels,
            "use_trained_in_every": bool(use_trained_in_every),
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        )
        json.dump(payload, tmp)
        tmp.flush()
        tmp.close()

        # Resolve aspect ratio (same logic as _on_compare_stacks).
        width, height = 1024, 1024
        for label, w, h in ASPECT_RATIOS:
            if label == self.gui.aspect_var.get():
                width, height = w, h
                break

        args = [
            "generate", str(project.root),
            "--prompt", body,                          # BODY ONLY
            "--steps", self.gui.steps_var.get() or "28",
            "--guidance", self.gui.guidance_var.get() or "5.5",
            "--width", str(width),
            "--height", str(height),
            "--sampler", self.gui.sampler_var.get() or "default",
            "--compare-loras-json", tmp.name,
        ]
        neg = self._final_negative()
        if neg:
            args += ["--negative", neg]
        seed = self.gui.seed_var.get().strip()
        if seed:
            args += ["--seed", seed]
        # `use_trained_in_every` from the dialog overrides the toggle for
        # the duration of the comparison so the user can include or
        # exclude the trained LoRA without flipping the main checkbox.
        if not use_trained_in_every:
            args.append("--no-trained-lora")

        # Total image count for the progress bar.
        n_total = max(1, len(recipes) * max(1, len(stack_labels)))
        self._begin_run(n_total)
        self.gui.on_next_exit = self._end_run
        self.gui.spawn(args)

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
