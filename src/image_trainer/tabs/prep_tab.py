"""02 · INGEST tab — pick source folder, resize into processed/.

Adds a Dry-run button that computes crops without writing full-res PNGs,
writing 256px previews into ``project/preview/`` instead. The GUI opens
the preview folder when the dry-run finishes so the user can inspect the
crop strategy at a glance.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING

from .. import gui_helpers, gui_theme
from ..gui_widgets import FolderField

if TYPE_CHECKING:
    from ..gui_app import TrainerGUI


def build(gui: "TrainerGUI") -> None:
    """Lay out the Ingest tab widgets on ``gui.tab_prep``."""
    PAD = gui_theme.PAD
    f = gui.tab_prep
    f.columnconfigure(1, weight=1)
    gui.source_dir_var = tk.StringVar()

    ttk.Label(
        f, text="Ingest · resize to target square",
        style="Header.TLabel",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, PAD))

    ttk.Label(f, text="Source folder:").grid(row=1, column=0, sticky="w")
    FolderField(
        f, textvariable=gui.source_dir_var,
        browse_title="Choose source folder of raw images",
    ).grid(row=1, column=1, columnspan=2, sticky="we", padx=PAD)

    ttk.Checkbutton(
        f,
        text=(
            "Face-aware crop (rule of thirds) — detects the subject and "
            "places the face on a third-line intersection. Images with no "
            "detectable face are marked excluded for review."
        ),
        variable=gui.face_aware_var,
    ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(PAD, 0))

    btns = ttk.Frame(f)
    btns.grid(row=3, column=0, columnspan=3, sticky="w", pady=PAD)
    ttk.Button(
        btns, text="IMPORT & RESIZE", style="Primary.TButton",
        command=lambda: _on_prep(gui, dry_run=False),
    ).pack(side="left")
    ttk.Button(
        btns, text="DRY-RUN PREVIEW",
        command=lambda: _on_prep(gui, dry_run=True),
    ).pack(side="left", padx=(PAD, 0))
    ttk.Button(
        btns, text="OPEN PROCESSED",
        style="Ghost.TButton",
        command=lambda: _open_processed(gui),
    ).pack(side="left", padx=(PAD, 0))

    ttk.Label(
        f,
        text=(
            "Dry-run writes 256px thumbnails to project/preview/ without "
            "touching processed/, so you can eyeball crop choices before "
            "committing."
        ),
        style="Status.TLabel", justify="left", wraplength=700,
    ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, PAD))


def _on_prep(gui: "TrainerGUI", *, dry_run: bool) -> None:
    project = gui.require_project()
    if not project:
        return
    gui.save_settings_silent()
    args = ["prep", str(project.root)]
    src = gui.source_dir_var.get().strip()
    if src:
        args += ["--source", src]
    if dry_run:
        args += ["--dry-run"]
        gui.log_queue.put("[dry-run: no files will be written to processed/]\n")
        # Queue a folder open when the run finishes.
        gui.on_next_exit = lambda: _open_preview(gui)
    gui.spawn(args)


def _open_processed(gui: "TrainerGUI") -> None:
    if not gui.current_project:
        return
    gui_helpers.open_folder(gui.current_project.processed_dir)


def _open_preview(gui: "TrainerGUI") -> None:
    if not gui.current_project:
        return
    preview_dir = gui.current_project.root / "preview"
    gui_helpers.open_folder(preview_dir)
