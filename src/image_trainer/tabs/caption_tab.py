"""03 · Caption tab — pick captioner, tune thresholds, run."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING

from .. import gui_theme
from ..gui_widgets import info_icon

if TYPE_CHECKING:
    from ..gui_app import TrainerGUI


def build(gui: "TrainerGUI") -> None:
    PAD = gui_theme.PAD
    f = gui.tab_caption
    f.columnconfigure(0, weight=1)

    # Tk vars owned by the GUI so save_settings_silent can persist them.
    gui.captioner_var = tk.StringVar(value="both")
    gui.general_threshold_var = tk.StringVar(value="0.35")
    gui.character_threshold_var = tk.StringVar(value="0.85")
    gui.caption_suffix_var = tk.StringVar(value="")
    gui.caption_nsfw_var = tk.BooleanVar(value=False)

    ttk.Label(f, text="Caption · processed images", style="Header.TLabel").grid(
        row=0, column=0, sticky="w", pady=(0, PAD)
    )
    ttk.Label(
        f,
        text=(
            "Writes '<trigger>, <caption>' .txt files next to each processed PNG. "
            "Edit afterwards on the Review tab. CUDA required for BLIP."
        ),
        style="Muted.TLabel", justify="left", wraplength=820,
    ).grid(row=1, column=0, sticky="w", pady=(0, PAD))

    # ---- mode picker ----
    mode_box = ttk.LabelFrame(f, text="Captioner", padding=PAD)
    mode_box.grid(row=2, column=0, sticky="we", pady=(0, PAD))
    mode_box.columnconfigure(0, weight=1)

    mode_row = ttk.Frame(mode_box)
    mode_row.grid(row=0, column=0, sticky="w")
    for label, value, hint in (
        (
            "BLIP only", "blip",
            "Salesforce BLIP-large generates an English sentence per image. "
            "Tame, non-explicit. Fast. Good baseline.",
        ),
        (
            "WD14 only", "wd14",
            "SmilingWolf's Danbooru-style tagger. Outputs comma-separated booru "
            "tags including anatomy + clothing + pose. Best for NSFW/lewd "
            "datasets where BLIP is too prudish.",
        ),
        (
            "BLIP + WD14 (recommended)", "both",
            "Concatenates both: '<trigger>, <BLIP sentence>, <WD14 tags>'. "
            "Sentence supplies framing, tags supply anatomy. Best default for "
            "person LoRAs.",
        ),
    ):
        rb_row = ttk.Frame(mode_row)
        rb_row.pack(side="left", padx=(0, PAD * 2))
        ttk.Radiobutton(
            rb_row, text=label, value=value, variable=gui.captioner_var,
        ).pack(side="left")
        info_icon(rb_row, hint).pack(side="left")

    # ---- WD14 thresholds ----
    thr_box = ttk.LabelFrame(f, text="WD14 thresholds (only applies to WD14 / Both)", padding=PAD)
    thr_box.grid(row=3, column=0, sticky="we", pady=(0, PAD))

    g_row = ttk.Frame(thr_box)
    g_row.grid(row=0, column=0, sticky="w", pady=2)
    ttk.Label(g_row, text="General:").pack(side="left")
    info_icon(
        g_row,
        "Confidence cutoff for general booru tags (clothing, pose, body parts). "
        "Lower = more tags appear in captions. Default 0.35; the NSFW preset "
        "lowers this to 0.25 to surface anatomy detail.",
    ).pack(side="left")
    ttk.Spinbox(
        g_row, from_=0.05, to=0.9, increment=0.05,
        textvariable=gui.general_threshold_var, width=6,
    ).pack(side="left", padx=(PAD // 2, 0))

    c_row = ttk.Frame(thr_box)
    c_row.grid(row=1, column=0, sticky="w", pady=2)
    ttk.Label(c_row, text="Character:").pack(side="left")
    info_icon(
        c_row,
        "Confidence cutoff for character-name tags. These rarely make sense "
        "for personal-likeness LoRAs (you don't want 'hatsune miku' showing "
        "up in your caption), so the default 0.85 deliberately suppresses "
        "most of them.",
    ).pack(side="left")
    ttk.Spinbox(
        c_row, from_=0.05, to=0.99, increment=0.05,
        textvariable=gui.character_threshold_var, width=6,
    ).pack(side="left", padx=(PAD // 2, 0))

    # ---- NSFW preset + suffix ----
    nsfw_box = ttk.LabelFrame(f, text="Style & NSFW", padding=PAD)
    nsfw_box.grid(row=4, column=0, sticky="we", pady=(0, PAD))
    nsfw_box.columnconfigure(1, weight=1)

    nsfw_row = ttk.Frame(nsfw_box)
    nsfw_row.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
    ttk.Checkbutton(
        nsfw_row, text="NSFW / lewd dataset preset",
        variable=gui.caption_nsfw_var,
    ).pack(side="left")
    info_icon(
        nsfw_row,
        "Lowers the WD14 general threshold to 0.25 and seeds the suffix with "
        "'explicit, nsfw, detailed anatomy' if you haven't set your own. "
        "Steers the LoRA toward learning adult-content vocabulary so prompts "
        "summon the right style at generation time.",
    ).pack(side="left")

    suf_row = ttk.Frame(nsfw_box)
    suf_row.grid(row=1, column=0, columnspan=2, sticky="we", pady=(PAD, 2))
    suf_row.columnconfigure(2, weight=1)
    ttk.Label(suf_row, text="Extra suffix:").grid(row=0, column=0, sticky="w")
    info_icon(
        suf_row,
        "Comma-separated tokens appended to every caption. Use this to anchor "
        "a stylistic vibe ('photorealistic, soft lighting') or domain hints "
        "('amateur photography, smartphone'). Leave blank for default behaviour.",
    ).grid(row=0, column=1, sticky="w")
    ttk.Entry(suf_row, textvariable=gui.caption_suffix_var).grid(
        row=0, column=2, sticky="we", padx=(PAD, 0),
    )

    # ---- run ----
    ttk.Button(
        f, text="Run captioning", style="Primary.TButton",
        command=lambda: _on_caption(gui),
    ).grid(row=5, column=0, sticky="w", pady=PAD)


def _on_caption(gui: "TrainerGUI") -> None:
    project = gui.require_project()
    if not project:
        return
    # Persist the captioner settings before spawning so the subprocess's
    # Project.load() sees them. Keeping this read-side rather than passing
    # everything as flags also means the user's tweaks survive across runs.
    try:
        project.captioner = gui.captioner_var.get()
        project.caption_general_threshold = float(gui.general_threshold_var.get())
        project.caption_character_threshold = float(gui.character_threshold_var.get())
        project.caption_extra_suffix = gui.caption_suffix_var.get().strip()
        project.caption_nsfw_preset = bool(gui.caption_nsfw_var.get())
    except ValueError as e:
        messagebox.showerror("Invalid threshold", str(e))
        return
    project.save()

    args = ["caption", str(project.root)]
    # The subprocess will read the persisted Project values, so we don't need
    # to forward thresholds on the CLI here. The flags exist for direct CLI use.
    gui.spawn(args)
