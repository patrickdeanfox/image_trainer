"""01 · SETTINGS tab — subject, base model, OOM knobs, training length.

All widgets bind to `gui.*_var` Tk variables that :class:`TrainerGUI`
synchronises with the current :class:`Project` on load/save.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING

from .. import gui_helpers, gui_theme
from ..config import Project
from ..gui_widgets import FolderField

if TYPE_CHECKING:
    from ..gui_app import TrainerGUI


def build(gui: "TrainerGUI") -> None:
    """Lay out the Settings tab widgets on ``gui.tab_settings``."""
    t = gui_theme.THEME
    PAD = gui_theme.PAD
    f = gui.tab_settings
    f.columnconfigure(1, weight=1)

    # Tk vars — owned by the GUI so other modules can read them.
    gui.trigger_var = tk.StringVar()
    gui.base_model_var = tk.StringVar()
    gui.resolution_var = tk.StringVar(value="1024")
    gui.lora_rank_var = tk.StringVar(value="32")
    gui.grad_accum_var = tk.StringVar(value="1")
    gui.max_steps_var = tk.StringVar(value="1500")
    gui.checkpointing_steps_var = tk.StringVar(value="100")
    gui.validation_steps_var = tk.StringVar(value="200")
    gui.xformers_var = tk.BooleanVar(value=True)
    gui.te_lora_var = tk.BooleanVar(value=False)
    gui.face_aware_var = tk.BooleanVar(value=True)

    ttk.Label(f, text="Subject · base model", style="Header.TLabel").grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, PAD)
    )

    ttk.Label(f, text="Trigger word:").grid(row=1, column=0, sticky="w", pady=2)
    ttk.Entry(f, textvariable=gui.trigger_var, width=30).grid(
        row=1, column=1, sticky="w", padx=PAD
    )

    ttk.Label(f, text="Base SDXL checkpoint:").grid(row=2, column=0, sticky="w", pady=2)
    base_field = FolderField(
        f, textvariable=gui.base_model_var,
        browse_title="Choose base SDXL checkpoint",
        file_mode=True,
        filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")],
    )
    base_field.grid(row=2, column=1, columnspan=2, sticky="we", padx=PAD)

    # --- OOM / quality knobs ---
    oom = ttk.LabelFrame(f, text="OOM · quality knobs", padding=PAD)
    oom.grid(row=3, column=0, columnspan=3, sticky="we", pady=(PAD * 2, PAD))
    oom.columnconfigure(1, weight=0)
    oom.columnconfigure(3, weight=1)

    ttk.Label(oom, text="Resolution:").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Combobox(
        oom, textvariable=gui.resolution_var,
        values=["512", "768", "1024"], state="readonly", width=8,
    ).grid(row=0, column=1, sticky="w", padx=PAD)

    ttk.Label(oom, text="LoRA rank:").grid(row=0, column=2, sticky="w", pady=2)
    ttk.Combobox(
        oom, textvariable=gui.lora_rank_var,
        values=["8", "16", "32", "64"], state="readonly", width=6,
    ).grid(row=0, column=3, sticky="w", padx=PAD)

    ttk.Label(oom, text="Grad accumulation:").grid(row=1, column=0, sticky="w", pady=2)
    ttk.Combobox(
        oom, textvariable=gui.grad_accum_var,
        values=["1", "2", "4", "8"], state="readonly", width=6,
    ).grid(row=1, column=1, sticky="w", padx=PAD)

    ttk.Checkbutton(oom, text="xformers", variable=gui.xformers_var).grid(
        row=1, column=2, sticky="w", padx=PAD
    )
    ttk.Checkbutton(
        oom, text="Text-encoder LoRA (higher quality, slower)",
        variable=gui.te_lora_var,
    ).grid(row=1, column=3, sticky="w", padx=PAD)

    # --- schedule ---
    sched = ttk.LabelFrame(f, text="Training length", padding=PAD)
    sched.grid(row=4, column=0, columnspan=3, sticky="we", pady=(0, PAD))

    ttk.Label(sched, text="Max steps:").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Entry(sched, textvariable=gui.max_steps_var, width=10).grid(
        row=0, column=1, sticky="w", padx=PAD
    )
    ttk.Label(sched, text="Checkpoint every:").grid(row=0, column=2, sticky="w", pady=2)
    ttk.Entry(sched, textvariable=gui.checkpointing_steps_var, width=10).grid(
        row=0, column=3, sticky="w", padx=PAD
    )
    ttk.Label(sched, text="Validation every (0 = off):").grid(
        row=1, column=0, sticky="w", pady=2
    )
    ttk.Entry(sched, textvariable=gui.validation_steps_var, width=10).grid(
        row=1, column=1, sticky="w", padx=PAD
    )

    ttk.Button(
        f, text="SAVE SETTINGS", style="Primary.TButton",
        command=lambda: _on_save(gui),
    ).grid(row=5, column=1, sticky="w", padx=PAD, pady=PAD)


def _on_save(gui: "TrainerGUI") -> None:
    """Validate, compute diff, confirm via messagebox, then persist."""
    if not gui.current_project:
        messagebox.showerror("No project", "Create or open a project first.")
        return
    project = gui.current_project

    # Snapshot the current on-disk state so we can diff.
    try:
        old = Project.load(project.root)
    except Exception:
        old = project

    # Apply UI values into a fresh Project copy (so we don't mutate the
    # current object until the user confirms).
    proposed = Project.load(project.root) if old is not project else project
    try:
        proposed.trigger_word = gui.trigger_var.get().strip() or proposed.trigger_word
        base = gui.base_model_var.get().strip()
        proposed.base_model_path = Path(base) if base else None
        proposed.resolution = int(gui.resolution_var.get())
        proposed.lora_rank = int(gui.lora_rank_var.get())
        proposed.lora_alpha = proposed.lora_rank
        proposed.gradient_accumulation_steps = int(gui.grad_accum_var.get())
        proposed.max_train_steps = int(gui.max_steps_var.get())
        proposed.checkpointing_steps = int(gui.checkpointing_steps_var.get())
        proposed.validation_steps = int(gui.validation_steps_var.get())
        proposed.use_xformers = bool(gui.xformers_var.get())
        proposed.train_text_encoder = bool(gui.te_lora_var.get())
        proposed.face_aware_crop = bool(gui.face_aware_var.get())
    except ValueError as e:
        messagebox.showerror("Invalid value", str(e))
        return

    diff = gui_helpers.config_diff(old, proposed)
    if diff:
        body = gui_helpers.format_config_diff(diff)
        warn = _describe_invalidation(diff)
        full = body if not warn else f"{body}\n\n⚠  {warn}"
        ok = messagebox.askokcancel(
            "Confirm config changes",
            f"The following fields will change:\n\n{full}\n\nApply?",
        )
        if not ok:
            gui.status_var.set("save cancelled")
            return

    # Mutate the live project and persist.
    project.trigger_word = proposed.trigger_word
    project.base_model_path = proposed.base_model_path
    project.resolution = proposed.resolution
    project.lora_rank = proposed.lora_rank
    project.lora_alpha = proposed.lora_alpha
    project.gradient_accumulation_steps = proposed.gradient_accumulation_steps
    project.max_train_steps = proposed.max_train_steps
    project.checkpointing_steps = proposed.checkpointing_steps
    project.validation_steps = proposed.validation_steps
    project.use_xformers = proposed.use_xformers
    project.train_text_encoder = proposed.train_text_encoder
    project.face_aware_crop = proposed.face_aware_crop
    project.save()
    gui.status_var.set(f"settings saved to {project.config_path.name}")
    gui.log_queue.put(f"[settings saved to {project.config_path}]\n")
    gui.refresh_step_status()


def _describe_invalidation(diff) -> str:
    """Return a warning string when the diff touches fields that invalidate
    existing caches or checkpoints."""
    invalidators = {"resolution", "lora_rank", "lora_alpha", "base_model_path"}
    touched = {k for k, _, _ in diff}
    hits = sorted(touched & invalidators)
    if not hits:
        return ""
    return (
        f"Changing {', '.join(hits)} invalidates cache/ and checkpoints/. "
        "Next training run will rebuild them; --resume will be rejected."
    )
