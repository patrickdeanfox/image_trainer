"""06 · GENERATE tab — load base + trained LoRA and sample images."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from .. import gui_helpers, gui_theme

if TYPE_CHECKING:
    from ..gui_app import TrainerGUI


def build(gui: "TrainerGUI") -> None:
    PAD = gui_theme.PAD
    f = gui.tab_generate
    f.columnconfigure(1, weight=1)

    gui.prompt_var = tk.StringVar(value="ohwx person, portrait, natural lighting")
    gui.negative_var = tk.StringVar(value="")
    gui.n_var = tk.StringVar(value="4")
    gui.steps_var = tk.StringVar(value="30")
    gui.guidance_var = tk.StringVar(value="7.0")
    gui.seed_var = tk.StringVar(value="")

    ttk.Label(f, text="Generate · trained LoRA", style="Header.TLabel").grid(
        row=0, column=0, columnspan=4, sticky="w", pady=(0, PAD)
    )

    ttk.Label(f, text="Prompt:").grid(row=1, column=0, sticky="w", pady=2)
    ttk.Entry(f, textvariable=gui.prompt_var).grid(
        row=1, column=1, columnspan=3, sticky="we", padx=PAD
    )

    ttk.Label(f, text="Negative:").grid(row=2, column=0, sticky="w", pady=2)
    ttk.Entry(f, textvariable=gui.negative_var).grid(
        row=2, column=1, columnspan=3, sticky="we", padx=PAD
    )

    ttk.Label(f, text="N images:").grid(row=3, column=0, sticky="w", pady=2)
    ttk.Entry(f, textvariable=gui.n_var, width=8).grid(
        row=3, column=1, sticky="w", padx=PAD
    )
    ttk.Label(f, text="Steps:").grid(row=3, column=2, sticky="w", pady=2)
    ttk.Entry(f, textvariable=gui.steps_var, width=8).grid(
        row=3, column=3, sticky="w", padx=PAD
    )

    ttk.Label(f, text="Guidance:").grid(row=4, column=0, sticky="w", pady=2)
    ttk.Entry(f, textvariable=gui.guidance_var, width=8).grid(
        row=4, column=1, sticky="w", padx=PAD
    )
    ttk.Label(f, text="Seed (blank = random):").grid(row=4, column=2, sticky="w", pady=2)
    ttk.Entry(f, textvariable=gui.seed_var, width=12).grid(
        row=4, column=3, sticky="w", padx=PAD
    )

    btns = ttk.Frame(f)
    btns.grid(row=5, column=0, columnspan=4, sticky="w", pady=PAD)
    ttk.Button(btns, text="GENERATE", style="Primary.TButton",
               command=lambda: _on_generate(gui)).pack(side="left")
    ttk.Button(btns, text="OPEN OUTPUTS", style="Ghost.TButton",
               command=lambda: _open_outputs(gui)).pack(side="left", padx=PAD)


def _on_generate(gui: "TrainerGUI") -> None:
    project = gui.require_project()
    if not project:
        return
    gui.save_settings_silent()
    args = [
        "generate",
        str(project.root),
        "--prompt", gui.prompt_var.get(),
        "--n", gui.n_var.get() or "4",
        "--steps", gui.steps_var.get() or "30",
        "--guidance", gui.guidance_var.get() or "7.0",
    ]
    neg = gui.negative_var.get().strip()
    if neg:
        args += ["--negative", neg]
    seed = gui.seed_var.get().strip()
    if seed:
        args += ["--seed", seed]
    gui.spawn(args)


def _open_outputs(gui: "TrainerGUI") -> None:
    if not gui.current_project:
        return
    gui_helpers.open_folder(gui.current_project.outputs_dir)
