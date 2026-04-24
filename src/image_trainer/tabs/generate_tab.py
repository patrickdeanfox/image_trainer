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
#: top of file. The remaining preset tables (PROMPT_TEMPLATES,
#: NEGATIVE_PRESETS, BUILDER_*) stay GUI-side — they're consumed only by
#: this tab.
# (QUALITY_STACKS moved to image_trainer.prompt_presets — imported above.)


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
        "Photoreal · heavy anti-anime anchor",
        "{trigger}, 1girl, beautiful blonde woman, real woman, not anime, "
        "not illustration, photograph, raw photo, shot on leica, "
        "35mm film grain, kodak portra 400, natural skin texture, "
        "skin pores visible, subsurface scattering, realistic skin tones, "
        "slight skin blemishes, fine facial hair, fully nude, full body "
        "visible, standing, contrapposto, looking at viewer, natural body "
        "proportions, anatomically correct, soft window light, neutral "
        "background, shallow depth of field, sharp focus on eyes, {body}",
        "For when vanilla Pony keeps giving you anime / illustrated "
        "output despite source_real. This template stuffs heavier photo "
        "anchors — film stock, skin pores, subsurface scattering, slight "
        "blemishes — that force Pony into its photograph distribution. "
        "'not anime, not illustration, real woman' are explicit anti-style "
        "tags. Pair with 'Pony · heavy photoreal (anti-anime)' quality "
        "stack + 'NSFW · photoreal push' negative. Long — install compel "
        "to use with extra body-text.",
    ),
    (
        "Amateur selfie · iPhone aesthetic",
        "{trigger}, 1girl, beautiful blonde woman, amateur selfie, "
        "iphone photo, smartphone photo, selfie in mirror, bedroom "
        "background, natural lighting, no filter, real skin, minor "
        "imperfections, candid expression, no makeup filter, fully nude, "
        "full body or mirror reflection, holding phone, casual intimate "
        "moment, slight motion blur, instagram aesthetic, {body}",
        "Mimics the OnlyFans / Instagram smartphone aesthetic that most "
        "personal-likeness LoRAs are trained on. This is the CLOSEST "
        "distribution to your training data if the dataset came from "
        "phone shots — the LoRA fires strongest here. Suppresses Pony's "
        "glossy-magazine tendency in favour of real-photo feel.",
    ),
    (
        "Studio glamour nude · professional",
        "{trigger}, 1girl, beautiful blonde woman, professional glamour "
        "photography, studio setup, softbox key light, rim light, "
        "seamless grey backdrop, medium format camera, hasselblad, "
        "natural skin detail, high detail body, fully nude, full body "
        "standing pose, arched back, looking at viewer, professional "
        "model, magazine quality, shallow depth of field, raw photo, "
        "editorial nude, tasteful composition, {body}",
        "Polished glamour nude with professional-studio anchors (softbox, "
        "rim light, medium format, hasselblad). Use with Lustify / Pony "
        "Realism quality stack for maximum 'fashion magazine' feel. Best "
        "when you want output that looks shot in a real photo studio "
        "rather than a bedroom.",
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
        "NSFW · heavy anti-anime (aggressive)",
        "score_4, score_5, score_6, source_anime, source_cartoon, "
        "anime, manga, illustration, drawing, painting, cartoon, sketch, "
        "lineart, cel shading, anime style, cartoonish, stylized, "
        "3d render, cgi, unreal engine, blender, rendered, "
        "doll, figurine, mannequin, plastic skin, plastic, smooth skin, "
        "airbrushed, flawless skin, perfect skin, porcelain skin, "
        "oversaturated, vivid colors, highly saturated, "
        "big eyes, large eyes, anime eyes, shiny eyes, "
        "low quality, worst quality, lowres, blurry, watermark, "
        "signature, text, bad anatomy, deformed, extra limbs, extra "
        "fingers, fused fingers, malformed hands, censored, mosaic",
        "Maximum anti-anime / anti-stylised push. Use when the standard "
        "photoreal push still gives you illustrated output — this one "
        "also kills the stylised 3D-render look and Pony's characteristic "
        "large-eye anime-face proportions. Pair with 'Pony · heavy "
        "photoreal' stack and the 'Photoreal · heavy anti-anime anchor' "
        "template. Heavier token budget but worth it on vanilla Pony.",
    ),
    (
        "NSFW · real-skin imperfection",
        "score_4, score_5, score_6, airbrushed, photoshopped, retouched, "
        "perfect skin, porcelain skin, plastic skin, smooth skin, "
        "flawless skin, glossy skin, wet-look skin, doll skin, "
        "oversaturated, vivid, hdr, "
        "anime, illustration, drawing, cartoon, 3d render, cgi, "
        "low quality, blurry, watermark, signature, bad anatomy, "
        "deformed, extra limbs, extra fingers, fused fingers",
        "Targets the 'plastic doll' skin look specifically — Pony's "
        "biggest tell when it's trying to look photoreal but isn't. Adds "
        "'airbrushed / photoshopped / retouched / glossy' to the "
        "negative so output gets actual pore texture + natural skin "
        "variation instead of Instagram-filter perfection.",
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
    "Hair colour": [
        ("(any)", ""),
        ("blonde", "long wavy blonde hair"),
        ("platinum blonde", "platinum blonde hair, very light blonde"),
        ("dirty blonde", "dirty blonde hair"),
        ("strawberry blonde", "strawberry blonde hair"),
        ("brunette", "long brunette hair"),
        ("dark brown", "long dark brown hair"),
        ("black", "long black hair"),
        ("auburn", "auburn hair"),
        ("red / ginger", "long red hair, ginger"),
        ("pink dyed", "pink dyed hair"),
        ("blue dyed", "blue dyed hair"),
        ("short blonde", "short blonde hair, pixie cut"),
        ("short dark", "short dark hair, bob cut"),
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
        ("undressing", "in the act of undressing, clothes coming off"),
        ("masturbating", "masturbating, intimate solo, hand between legs"),
        ("oral / blowjob", "performing oral sex, blowjob, fellatio"),
        ("intercourse", "intercourse, intimate sex"),
        ("cowgirl", "cowgirl position, on top, straddling"),
        ("doggystyle", "doggystyle position, from behind"),
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
        ("arms behind head", "arms raised behind head"),
        ("arms behind back", "arms behind back"),
        ("hands on chest", "hands cupping breasts"),
        ("on knees", "on knees, kneeling pose"),
        ("on all fours", "on all fours, hands and knees"),
        ("legs spread", "legs spread wide"),
        ("legs crossed", "legs crossed"),
        ("squatting", "squatting pose"),
        ("bent over", "bent over, ass facing camera"),
        ("arched back", "arched back, breasts forward"),
        ("looking back", "looking back at viewer over shoulder"),
        ("on side", "lying on side, hip raised"),
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
    # 5.5 is the photoreal sweet spot for Pony-family bases — high CFG
    # fries skin texture into plastic / hyperstylised looks. Bump to
    # 7-8 if you want more prompt-following at the cost of realism.
    gui.guidance_var = tk.StringVar(value="5.5")
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
    # back to the recommended Pony NSFW opener. The user can change the
    # default at any time via the "Set as default" button next to the
    # quality-stack combobox — that writes their current pick back to
    # .user_settings.json and it persists across launches + projects.
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
    action_bar.columnconfigure(3, weight=1)
    state.generate_btn = ttk.Button(
        action_bar, text="Generate", style="Primary.TButton",
        command=state._on_generate,
    )
    state.generate_btn.grid(row=0, column=0, sticky="w")
    state.compare_btn = ttk.Button(
        action_bar, text="Compare across all stacks",
        style="Ghost.TButton",
        command=state._on_compare_stacks,
    )
    state.compare_btn.grid(row=0, column=1, sticky="w", padx=(PAD // 2, 0))
    ttk.Button(
        action_bar, text="Open outputs", style="Ghost.TButton",
        command=state._open_outputs,
    ).grid(row=0, column=2, sticky="w", padx=(PAD // 2, 0))
    state.progress = ttk.Progressbar(
        action_bar, mode="determinate",
        style="Trainer.Horizontal.TProgressbar",
    )
    state.progress.grid(row=0, column=3, sticky="we", padx=(PAD, 0))
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

        # Layout strategy: Prompt builder + Prompt body + LoRA stack are the
        # primary interactions and stay expanded. Quality stack, prompt
        # template, negative prompt, and sampler/dimensions are
        # set-and-forget for most runs and start collapsed — one click to
        # open when the user wants to tweak them.
        #
        # The CollapsibleFrame pattern: outer frame carries the toggle
        # header; inner padded Frame holds the actual widgets so they don't
        # touch the collapsible's edges.

        # ROW 0: Prompt builder (NEW, expanded — primary interaction)
        pb_outer = CollapsibleFrame(root, text="Prompt builder", start_open=True)
        pb_outer.grid(row=0, column=0, sticky="we", pady=(0, PAD))
        pb_box = ttk.Frame(pb_outer.body, padding=PAD)
        pb_box.pack(fill="both", expand=True)
        self._build_prompt_builder(pb_box)

        # ROW 1: Prompt body (always visible — this is the main editable output)
        prompt_box = ttk.LabelFrame(root, text="Prompt body", padding=PAD)
        prompt_box.grid(row=1, column=0, sticky="we", pady=(0, PAD))
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

        # ROW 2: Quality stack (collapsible, collapsed — set-and-forget)
        qs_outer = CollapsibleFrame(
            root, text="Quality-tag prefix (advanced)", start_open=False,
        )
        qs_outer.grid(row=2, column=0, sticky="we", pady=(0, PAD))
        qs_box = ttk.Frame(qs_outer.body, padding=PAD)
        qs_box.pack(fill="both", expand=True)
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
        ttk.Button(
            qs_box, text="Set as default", style="Ghost.TButton",
            command=self._on_save_default_stack,
        ).grid(row=0, column=3, sticky="e", padx=(PAD // 2, 0))
        self.qs_hint_var = tk.StringVar(value=self._stack_hint(self.gui.quality_stack_var.get()))
        ttk.Label(
            qs_box, textvariable=self.qs_hint_var, style="Status.TLabel",
            wraplength=520, justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        qs_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_quality_stack_change(),
        )

        # ROW 3: Quick template picker (collapsible, collapsed — alternative
        # to the builder for users who want a one-click preset)
        tpl_outer = CollapsibleFrame(
            root, text="Quick prompt template (alt to builder)", start_open=False,
        )
        tpl_outer.grid(row=3, column=0, sticky="we", pady=(0, PAD))
        tpl_box = ttk.Frame(tpl_outer.body, padding=PAD)
        tpl_box.pack(fill="both", expand=True)
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

        # ROW 4: Negative prompt (collapsible, collapsed)
        neg_outer = CollapsibleFrame(
            root, text="Negative prompt", start_open=False,
        )
        neg_outer.grid(row=4, column=0, sticky="we", pady=(0, PAD))
        neg_box = ttk.Frame(neg_outer.body, padding=PAD)
        neg_box.pack(fill="both", expand=True)
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

        # ROW 5: Sampler & dimensions (collapsible, collapsed)
        nums_outer = CollapsibleFrame(
            root, text="Sampler & dimensions", start_open=False,
        )
        nums_outer.grid(row=5, column=0, sticky="we", pady=(0, PAD))
        nums_box = ttk.Frame(nums_outer.body, padding=PAD)
        nums_box.pack(fill="both", expand=True)
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
        # ROW 6: LoRA stack (always expanded — high-frequency interaction).
        lora_box = ttk.LabelFrame(root, text="LoRA stack", padding=PAD)
        lora_box.grid(row=6, column=0, sticky="we", pady=(0, PAD))
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

        tips_box = ttk.LabelFrame(root, text="Photoreal survival guide", padding=PAD)
        tips_box.grid(row=3, column=0, sticky="we")
        tips_box.columnconfigure(0, weight=1)
        ttk.Label(
            tips_box,
            text=(
                "• Getting anime output despite source_real? The BASE MODEL is "
                "the biggest lever. Switch from vanilla Pony V6 XL to Lustify "
                "XL or Pony Realism — both Pony fine-tunes but trained on "
                "photo data, so photoreal output is the default not the fight.\n"
                "• If you're stuck on vanilla Pony, use the 'Pony · heavy "
                "photoreal (anti-anime)' quality stack AND the 'NSFW · heavy "
                "anti-anime (aggressive)' negative preset. Combined they "
                "actually get photoreal output from vanilla Pony.\n"
                "• Drop CFG from 7 to 5-6 for photoreal. High CFG = fried / "
                "hyperstylized / plastic-looking; low CFG = natural but may "
                "ignore the prompt. 5.5 is the sweet spot for Pony photoreal.\n"
                "• Add an anti-anime LoRA from civitai (search 'realistic "
                "skin XL' or 'film photography LoRA') at 0.6-0.8. Bigger "
                "lever than any prompt change.\n"
                "• Stack trained likeness LoRA at 1.0 + style LoRA at 0.4-0.7. "
                "Trained carries likeness; style carries photoreal feel.\n"
                "• Lock the seed when iterating prompt wording — change one "
                "word at a time, see exactly what it did."
            ),
            style="Status.TLabel", wraplength=320, justify="left",
        ).grid(row=0, column=0, sticky="w")

    # ---- Prompt builder ----

    def _build_prompt_builder(self, root: ttk.Frame) -> None:
        """Lay out the structured prompt-builder dropdowns.

        Three sub-sections (Subject / Scene / Action), each with its
        dropdowns in a 2-column grid. Below them, a "Build prompt from
        picks" button that assembles the selected fragments into the
        prompt body. The user then edits the body freely — changing
        dropdowns afterwards doesn't touch the body until they click
        Build again.
        """
        PAD = gui_theme.PAD
        root.columnconfigure(0, weight=1)

        # --- one-line intro ---
        ttk.Label(
            root,
            text=(
                "Pick descriptors per axis → click 'Build prompt from picks'. "
                "Anything left on '(any)' is omitted. Edit the body freely "
                "after building."
            ),
            style="Status.TLabel", wraplength=520, justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, PAD // 2))

        # --- three sub-sections ---
        subj = ttk.LabelFrame(root, text="Subject", padding=PAD)
        subj.grid(row=1, column=0, sticky="we", pady=(0, PAD // 2))
        self._build_builder_grid(subj, BUILDER_SUBJECT, "Subject",
                                 include_tattoos=True)

        scn = ttk.LabelFrame(root, text="Scene", padding=PAD)
        scn.grid(row=2, column=0, sticky="we", pady=(0, PAD // 2))
        self._build_builder_grid(scn, BUILDER_SCENE, "Scene")

        act = ttk.LabelFrame(root, text="Action", padding=PAD)
        act.grid(row=3, column=0, sticky="we", pady=(0, PAD // 2))
        self._build_builder_grid(act, BUILDER_ACTION, "Action")

        # --- build + reset row ---
        btn_row = ttk.Frame(root)
        btn_row.grid(row=4, column=0, sticky="we", pady=(PAD // 2, 0))
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
            "'Build prompt from picks' replaces the Prompt body with a new "
            "prompt assembled from your dropdowns. Your LoRA trigger word is "
            "prepended automatically. Edits to the body after Build are "
            "preserved — the dropdowns don't auto-sync.",
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
        """
        parts: list[str] = ["1girl"]

        # Inject the project's trigger word if a trained LoRA will be used.
        # This keeps the builder consistent with the prompt-template flow.
        if self.gui.current_project and self.gui.use_trained_lora_var.get():
            trigger = (self.gui.current_project.trigger_word or "").strip()
            if trigger:
                parts.insert(0, trigger)

        def pick_fragment(group: str, key: str, table: dict) -> str:
            label = self.builder_vars[group][key].get()
            for lbl, frag in table[key]:
                if lbl == label:
                    return frag.strip()
            return ""

        # Subject descriptors
        for key in BUILDER_SUBJECT:
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

        # Always anchor "real photo" so even with (any) picks we push toward
        # photoreal output. Users can remove this in the body if they want
        # stylised output.
        parts.append("raw photo, real photograph")

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
        # Use the currently-selected negative preset.
        neg = self.gui.negative_var.get().strip()
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
