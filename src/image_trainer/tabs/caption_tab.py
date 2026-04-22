"""03 · CAPTION tab — run the configured captioner over processed/."""

from __future__ import annotations

from tkinter import ttk
from typing import TYPE_CHECKING

from .. import gui_theme

if TYPE_CHECKING:
    from ..gui_app import TrainerGUI


def build(gui: "TrainerGUI") -> None:
    PAD = gui_theme.PAD
    f = gui.tab_caption
    f.columnconfigure(0, weight=1)

    ttk.Label(f, text="Caption · processed images", style="Header.TLabel").grid(
        row=0, column=0, sticky="w", pady=(0, PAD)
    )
    ttk.Label(
        f,
        text=(
            "Runs the configured captioner over processed/ and writes\n"
            "'<trigger>, <caption>' .txt files.  Requires a CUDA GPU."
        ),
        justify="left",
    ).grid(row=1, column=0, sticky="w")
    ttk.Button(
        f, text="RUN CAPTIONING", style="Primary.TButton",
        command=lambda: _on_caption(gui),
    ).grid(row=2, column=0, sticky="w", pady=PAD)


def _on_caption(gui: "TrainerGUI") -> None:
    project = gui.require_project()
    if not project:
        return
    gui.save_settings_silent()
    gui.spawn(["caption", str(project.root)])
