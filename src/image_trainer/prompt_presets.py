"""Shared prompt-builder presets — data tables.

Kept as a standalone module (no Tk imports) so :mod:`pipeline.generate`
can iterate over the same QUALITY_STACKS list the GUI uses for the
``--compare-stacks`` mode without pulling in any GUI code. The Generate
tab also imports from here so there's a single source of truth.

If you add a new quality stack: edit ONE list, both the GUI dropdown and
the CLI compare-stacks loop pick it up automatically.
"""

from __future__ import annotations


#: Quality-tag stacks. Each entry: (display_label, prompt_fragment, hint).
#: The first entry "(none)" intentionally has an empty fragment — it's the
#: "don't prepend anything" choice. Compare-stacks mode skips it because
#: an empty stack isn't meaningful to compare.
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
        "photographic output. Works best on Pony fine-tunes trained on "
        "photo data (Pony Realism, Lustify XL); on vanilla Pony V6 XL "
        "output still drifts stylised — use 'Pony · heavy photoreal' "
        "instead, or switch base.",
    ),
    (
        "Pony · heavy photoreal (anti-anime)",
        "score_9, score_8_up, score_7_up, source_real, rating_explicit, "
        "photo, photorealistic, raw photo, real photograph, "
        "35mm film, skin pores, subsurface scattering, ",
        "For vanilla Pony V6 XL when the model keeps drifting toward "
        "anime / illustration. Adds heavier photo anchors — 35mm film, "
        "skin pores, subsurface scattering — that Pony was trained to "
        "associate with its photo-distribution. Pair with 'NSFW · "
        "photoreal push' negative for maximum effect. Bigger token cost "
        "(~20) — install compel if the rest of your prompt is long.",
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
        "Photoreal pro (non-Pony) · heavy",
        "raw photo, professional photograph, dslr, shot on leica, "
        "35mm film grain, kodak portra 400, natural skin texture, "
        "skin pores, subsurface scattering, sharp focus, detailed skin, "
        "photorealistic, hyperrealistic, ",
        "Heaviest photoreal anchor stack for non-Pony realism bases "
        "(RealVisXL V5, CyberRealistic XL, EpicRealism XL). Camera + film "
        "stock + skin-detail vocabulary anchors output firmly in the "
        "photograph distribution. About 20 tokens; pair with 'NSFW · "
        "photoreal push' negative. Do NOT use with Pony — the non-Pony "
        "vocabulary confuses Pony's score_X training.",
    ),
    (
        "Amateur / OnlyFans aesthetic",
        "iphone photo, smartphone photo, candid snapshot, amateur "
        "photograph, natural skin, no makeup filter, real woman, "
        "instagram aesthetic, slight motion blur, ",
        "Mimics the smartphone + amateur-photography distribution most "
        "OnlyFans / personal-likeness training datasets are drawn from. "
        "Best match to the LoRA's training distribution once you have a "
        "trained LoRA. 'no makeup filter' + 'real woman' actively suppress "
        "the airbrushed / doll-like output Pony defaults to.",
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


def stack_label_to_prefix(label: str) -> str:
    """Lookup helper. Returns "" if the label isn't found."""
    for lbl, prefix, _hint in QUALITY_STACKS:
        if lbl == label:
            return prefix
    return ""


def stacks_for_compare() -> list[tuple[str, str]]:
    """Return (label, prefix) pairs suitable for compare-stacks iteration.

    Skips the "(none)" entry because rendering with no prefix isn't a
    meaningful comparison — it's just the user's bare prompt without any
    quality anchoring.
    """
    return [(label, prefix) for label, prefix, _ in QUALITY_STACKS if prefix]
