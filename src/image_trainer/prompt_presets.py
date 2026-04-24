"""Shared prompt-builder presets — data tables.

Kept as a standalone module (no Tk imports) so :mod:`pipeline.generate`
can iterate over the same QUALITY_STACKS list the GUI uses for the
``--compare-stacks`` mode without pulling in any GUI code. The Generate
tab also imports from here so there's a single source of truth.

If you add a new quality stack: edit ONE list, both the GUI dropdown and
the CLI compare-stacks loop pick it up automatically.
"""

from __future__ import annotations


# The Pony "score up-tag" opener. Pony V6 was trained with a labelling
# convention where `score_9` is the highest-quality bucket, `score_8_up`
# = "score_8 OR higher", and so on down to score_4_up. The community
# consensus (Civitai author notes, reddit threads, the Pony model card
# itself) is that ALL SIX tags should appear in the prompt opener — not
# just the top 3. Using only 3 leaves the model underprompted vs how it
# was trained. Keep this as a constant so every Pony stack stays in sync.
_PONY_SCORE_6 = (
    "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up"
)


#: Quality-tag stacks. Each entry: (display_label, prompt_fragment, hint).
#: The first entry "(none)" intentionally has an empty fragment — it's the
#: "don't prepend anything" choice. Compare-stacks mode skips it because
#: an empty stack isn't meaningful to compare.
#:
#: Design principles (rev 2):
#: - Every Pony-family stack uses the full 6-tag score opener (see above).
#: - No stack contains `iphone photo` / `smartphone photo` / `motion blur`
#:   — Pony renders those literally, producing phone-frame compositions
#:   with Instagram UI overlays (confirmed in outputs 170003, 170835).
#: - The "heavy photoreal" stack's photo anchors live in the stack ONLY,
#:   not the body — so users can't accidentally duplicate score tags.
#: - "Amateur / OnlyFans aesthetic" now uses snapshot/candid anchors
#:   WITHOUT the phone vocabulary that caused the known failures.
QUALITY_STACKS: list[tuple[str, str, str]] = [
    (
        "(none)", "",
        "No quality prefix — your prompt stands alone. Pick this when "
        "you've hand-authored a complete prompt (including the Pony "
        "score opener if you're on Pony) and want zero auto-prepending.",
    ),
    (
        "Pony · photoreal NSFW (recommended)",
        f"{_PONY_SCORE_6}, source_real, rating_explicit, ",
        "The default Pony photoreal NSFW opener. All six score_up tags "
        "(community-standard), plus source_real + rating_explicit to "
        "bias toward photographic and unlock explicit output. Pair with "
        "a Pony-calibrated realism LoRA (e.g. zy_Realism_Enhancer_v2 at "
        "~0.35) for the cleanest result. ~20 tokens.",
    ),
    (
        "Pony · heavy photoreal (anti-anime)",
        f"{_PONY_SCORE_6}, source_real, rating_explicit, "
        "photo, photorealistic, raw photo, real photograph, "
        "35mm film, skin pores, subsurface scattering, ",
        "For vanilla Pony V6 XL when it keeps drifting toward anime / "
        "illustration. Adds photo anchors Pony was trained to associate "
        "with its photo-distribution. DO NOT also add score_ / source_ "
        "tags to the body — they'll duplicate. ~35 tokens; ensure "
        "compel is installed if the rest of your prompt is long.",
    ),
    (
        "Pony · photoreal SFW",
        f"{_PONY_SCORE_6}, source_real, rating_safe, ",
        "Same 6-tag Pony opener but with rating_safe instead of "
        "rating_explicit. Use for SFW portrait / lifestyle work.",
    ),
    (
        "Pony · explicit (heavy)",
        f"{_PONY_SCORE_6}, source_real, rating_explicit, "
        "rating_questionable, ",
        "Wider explicit range — adds rating_questionable alongside "
        "rating_explicit. Use when the base produces tasteful / implied "
        "output and you want it to commit further. Costs ~3 more tokens "
        "than the recommended stack.",
    ),
    (
        "Pony · score-only (minimal)",
        f"{_PONY_SCORE_6}, ",
        "Just the six Pony score tags, nothing else. Use when you want "
        "to hand-author every other anchor (photo vs illustration, "
        "rating, subject, etc.) in the body yourself. Smallest Pony "
        "prefix that still gets you Pony quality.",
    ),
    (
        "Illustrious / NoobAI XL",
        "masterpiece, best quality, very aware, newest, absurdres, ",
        "Tag stack for Illustrious-XL / NoobAI-XL family checkpoints. "
        "Different vocabulary from Pony — masterpiece + best quality + "
        "very aware (NoobAI's anatomy anchor) + newest (recency bias). "
        "Do NOT use on Pony bases; the vocabulary doesn't transfer.",
    ),
    (
        "RealVis / Juggernaut · photoreal",
        "raw photo, dslr, photorealistic, ",
        "Short realism-tilted opener for RealVisXL / JuggernautXL / "
        "vanilla SDXL fine-tunes that don't use score_X tags. Pair "
        "with the NSFW uncensor negative preset for explicit work.",
    ),
    (
        "Photoreal pro (non-Pony) · heavy",
        "raw photo, professional photograph, dslr, shot on leica, "
        "35mm film grain, kodak portra 400, natural skin texture, "
        "skin pores, subsurface scattering, sharp focus, detailed skin, "
        "photorealistic, hyperrealistic, ",
        "Heaviest photoreal anchor stack for non-Pony realism bases "
        "(RealVisXL V5, CyberRealistic XL, EpicRealism XL). About 20 "
        "tokens; pair with 'NSFW · photoreal push' negative. Do NOT "
        "use with Pony — the non-Pony vocabulary confuses Pony's score "
        "training.",
    ),
    (
        "Amateur / OnlyFans aesthetic",
        f"{_PONY_SCORE_6}, source_real, rating_explicit, "
        "candid snapshot, amateur photograph, natural skin, "
        "no makeup filter, real woman, ",
        "Mimics the amateur-photography distribution most OnlyFans / "
        "personal-likeness training datasets draw from. Does NOT include "
        "'iphone photo / smartphone photo / motion blur' — those tokens "
        "caused Pony to render phone-in-frame compositions with "
        "Instagram UI overlays. Keeps the candid anchors that actually "
        "help. Pair with a trained likeness LoRA.",
    ),
    (
        "Lustify / Pony Realism",
        f"{_PONY_SCORE_6}, source_real, rating_explicit, "
        "professional photography, ",
        "Tuned for Lustify XL and Pony Realism. Full 6-tag Pony opener "
        "+ professional-photography anchor for the magazine look these "
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
