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
from ..gui_widgets import FolderField, info_icon

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

    trigger_row = ttk.Frame(f)
    trigger_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=2)
    ttk.Label(trigger_row, text="Trigger word:").pack(side="left")
    info_icon(
        trigger_row,
        "A short phrase that becomes the 'name' of your subject in every "
        "training caption (e.g. 'ohwx person'). At generation time, putting "
        "the trigger in your prompt summons the LoRA's likeness. Pick "
        "something rare so it doesn't collide with normal English.",
    ).pack(side="left")
    ttk.Entry(trigger_row, textvariable=gui.trigger_var, width=30).pack(side="left", padx=(PAD, 0))

    base_row = ttk.Frame(f)
    base_row.grid(row=2, column=0, columnspan=3, sticky="we", pady=2)
    base_row.columnconfigure(2, weight=1)
    ttk.Label(base_row, text="Base SDXL checkpoint:").grid(row=0, column=0, sticky="w")
    info_icon(
        base_row,
        "The .safetensors file your LoRA trains on top of. Use a Pony "
        "checkpoint for stylised/NSFW work, base SDXL 1.0 for photoreal, "
        "or any community SDXL fine-tune. The same base must be re-used at "
        "generation time or quality collapses.",
    ).grid(row=0, column=1, sticky="w")
    base_field = FolderField(
        base_row, textvariable=gui.base_model_var,
        browse_title="Choose base SDXL checkpoint",
        file_mode=True,
        filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")],
    )
    base_field.grid(row=0, column=2, sticky="we", padx=PAD)

    # --- OOM / quality knobs ---
    oom = ttk.LabelFrame(f, text="OOM · quality knobs", padding=PAD)
    oom.grid(row=3, column=0, columnspan=3, sticky="we", pady=(PAD * 2, PAD))
    oom.columnconfigure(1, weight=0)
    oom.columnconfigure(3, weight=1)

    res_row = ttk.Frame(oom)
    res_row.grid(row=0, column=0, columnspan=4, sticky="w", pady=2)
    ttk.Label(res_row, text="Resolution:").pack(side="left")
    info_icon(
        res_row,
        "Square edge of training images in pixels. SDXL was trained at 1024. "
        "Drop to 768 if you OOM; 512 only as a last resort (heavy quality "
        "loss). Changing this invalidates the cache.",
    ).pack(side="left")
    ttk.Combobox(
        res_row, textvariable=gui.resolution_var,
        values=["512", "768", "1024"], state="readonly", width=8,
    ).pack(side="left", padx=PAD)
    ttk.Label(res_row, text="     LoRA rank:").pack(side="left")
    info_icon(
        res_row,
        "How much capacity the LoRA has. Higher = can capture more detail but "
        "needs more VRAM and is more prone to overfitting. 32 is the sweet "
        "spot for likeness LoRAs; drop to 16 if you OOM, raise to 64 only on "
        "big cards. Cannot be changed during --resume.",
    ).pack(side="left")
    ttk.Combobox(
        res_row, textvariable=gui.lora_rank_var,
        values=["8", "16", "32", "64"], state="readonly", width=6,
    ).pack(side="left", padx=PAD)

    grad_row = ttk.Frame(oom)
    grad_row.grid(row=1, column=0, columnspan=4, sticky="w", pady=2)
    ttk.Label(grad_row, text="Grad accumulation:").pack(side="left")
    info_icon(
        grad_row,
        "Effective batch size when train_batch_size is fixed at 1. "
        "Accumulates gradients across N forward passes before stepping the "
        "optimizer — same end result as a bigger batch but no extra VRAM. "
        "Tradeoff: each 'step' takes N× longer wall time. Leave at 1 unless "
        "training feels noisy.",
    ).pack(side="left")
    ttk.Combobox(
        grad_row, textvariable=gui.grad_accum_var,
        values=["1", "2", "4", "8"], state="readonly", width=6,
    ).pack(side="left", padx=PAD)

    ttk.Checkbutton(grad_row, text="xformers", variable=gui.xformers_var).pack(side="left", padx=PAD)
    info_icon(
        grad_row,
        "Memory-efficient attention via the xformers library. Saves ~1 GB "
        "of VRAM during training with no quality loss. If xformers isn't "
        "installed the loop falls back to PyTorch SDPA automatically.",
    ).pack(side="left")
    ttk.Checkbutton(
        grad_row, text="Text-encoder LoRA (higher quality, slower)",
        variable=gui.te_lora_var,
    ).pack(side="left", padx=PAD)
    info_icon(
        grad_row,
        "Also trains LoRA adapters on top of the two CLIP text encoders. "
        "Improves prompt-following and likeness for text-heavy descriptions. "
        "BUT: needs ~12 GB of VRAM because both encoders stay resident on "
        "GPU during training. The GUI will refuse to start a run with this "
        "on if your card has less.",
    ).pack(side="left")

    # --- schedule ---
    sched = ttk.LabelFrame(f, text="Training length", padding=PAD)
    sched.grid(row=4, column=0, columnspan=3, sticky="we", pady=(0, PAD))

    ms_row = ttk.Frame(sched)
    ms_row.grid(row=0, column=0, columnspan=4, sticky="w", pady=2)
    ttk.Label(ms_row, text="Max steps:").pack(side="left")
    info_icon(
        ms_row,
        "Total optimizer steps for this run. 1500 is a reasonable likeness-LoRA "
        "starting point; 2500-3000 if you have lots of varied images. More is "
        "not always better — past a point the LoRA memorises individual frames "
        "and stops generalising.",
    ).pack(side="left")
    ttk.Entry(ms_row, textvariable=gui.max_steps_var, width=10).pack(side="left", padx=PAD)
    ttk.Label(ms_row, text="     Checkpoint every:").pack(side="left")
    info_icon(
        ms_row,
        "Save a resumable training checkpoint every N steps. Higher = less "
        "disk used, lower = finer-grained recovery if you crash mid-run. "
        "100 is a good default; the saved snapshots live in checkpoints/.",
    ).pack(side="left")
    ttk.Entry(ms_row, textvariable=gui.checkpointing_steps_var, width=10).pack(side="left", padx=PAD)

    val_row = ttk.Frame(sched)
    val_row.grid(row=1, column=0, columnspan=4, sticky="w", pady=2)
    ttk.Label(val_row, text="Validation every (0 = off):").pack(side="left")
    info_icon(
        val_row,
        "Every N steps, the loop generates one preview image with your trigger "
        "word so you can eyeball training progress. Output goes to "
        "logs/validation/. Costs extra VRAM during the inference, so leave "
        "this at 0 if you're tight on memory; 200 is a nice cadence otherwise.",
    ).pack(side="left")
    ttk.Entry(val_row, textvariable=gui.validation_steps_var, width=10).pack(side="left", padx=PAD)

    ttk.Button(
        f, text="Save settings", style="Primary.TButton",
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
        messagebox.showerror(
            "Couldn't parse a numeric setting",
            f"One of the numeric fields (resolution, LoRA rank, grad "
            f"accumulation, max steps, checkpoint every, validation every) "
            f"isn't a valid integer.\n\nDetail: {e}",
        )
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
