"""Main Tk window: header, project bar, notebook, telemetry pane.

Owns the :class:`CLIRunner`, the shared log queue, the palette, and the
current :class:`Project`. All tab-specific layout and behaviour lives in
:mod:`image_trainer.tabs.*`; this file is just the scaffolding that binds
them together.

Header vs. project bar: the two used to be stacked horizontal bands. Now
they're folded into a single compact strip:

    ┌────────────────────────────────────────────────────────────────┐
    │ IMAGE TRAINER  [project ▾] [NEW] [RECENT ▾]  ● ● ● ● ● ●     │
    └────────────────────────────────────────────────────────────────┘

The six dots on the right mirror the wizard steps and change colour based
on the project's on-disk state (processed imgs exist → done, review has
excluded items → warn, etc.).
"""

from __future__ import annotations

import queue
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Callable, Optional

from . import gui_helpers, gui_theme
from .config import DEFAULT_PROJECTS_ROOT, Project, ProjectsRoot
from .gui_runner import CLIRunner
from .gui_widgets import CollapsibleFrame, StatusDot
from .tabs import (
    caption_tab,
    generate_tab,
    prep_tab,
    review_tab,
    settings_tab,
    storage_tab,
    train_tab,
    video_tab,
)


STEP_LABELS = [
    ("settings", "Settings"),
    ("prep", "Ingest"),
    ("caption", "Caption"),
    ("review", "Review"),
    ("train", "Train"),
    ("generate", "Generate"),
]
# NOTE: 07 · Storage is a real tab but not part of the linear pipeline,
# so it deliberately doesn't appear in STEP_LABELS / status dots.


class TrainerGUI:
    """Top-level window. Orchestrates tabs + subprocess runner + log pump."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("IMAGE TRAINER · OPERATIONS")
        self.root.geometry("1280x900")
        self.root.minsize(1100, 760)

        self.theme = gui_theme.apply_style(self.root)

        self.projects_root = ProjectsRoot(DEFAULT_PROJECTS_ROOT)
        self.projects_root.ensure()
        self.current_project: Optional[Project] = None
        self.recent_projects: list[str] = gui_helpers.load_recent(self.projects_root.root)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.runner = CLIRunner(self.log_queue)

        # Tab-state objects (populated by the tab builders).
        self.review_state = None
        self.train_state = None
        self.storage_state = None

        # Populated as tabs build their widgets.
        self.trigger_var: tk.StringVar
        self.base_model_var: tk.StringVar
        self.resolution_var: tk.StringVar
        self.lora_rank_var: tk.StringVar
        self.grad_accum_var: tk.StringVar
        self.max_steps_var: tk.StringVar
        self.checkpointing_steps_var: tk.StringVar
        self.validation_steps_var: tk.StringVar
        self.xformers_var: tk.BooleanVar
        self.te_lora_var: tk.BooleanVar
        self.face_aware_var: tk.BooleanVar
        self.source_dir_var: tk.StringVar
        self.prompt_var: tk.StringVar
        self.negative_var: tk.StringVar
        self.n_var: tk.StringVar
        self.steps_var: tk.StringVar
        self.guidance_var: tk.StringVar
        self.seed_var: tk.StringVar
        self.train_note_var: tk.StringVar
        self.train_status_var: tk.StringVar
        self.train_progress: ttk.Progressbar
        self.train_pct_var: tk.StringVar

        # Optional one-shot hook fired by _drain_log when the current subprocess exits.
        self.on_next_exit: Optional[Callable[[], None]] = None

        self._build_ui()
        self.root.after(100, self._drain_log)
        self.root.after(200, self._refresh_project_list)

    # ---- layout ----

    def _build_ui(self) -> None:
        PAD = gui_theme.PAD
        self._build_header()

        # Pack-order matters: bottom-anchored widgets claim their height
        # FIRST so the notebook (with fill="both" expand=True) only fills
        # the leftover middle space. Doing this in declaration order
        # (notebook first, then log_pane, then status) lets a tall tab
        # like Generate balloon the notebook and squeeze the telemetry
        # pane / status bar off the bottom of the window.
        self.status_var = tk.StringVar(value="ready")
        status = ttk.Label(self.root, textvariable=self.status_var,
                           style="Status.TLabel", anchor="w")
        status.pack(side="bottom", fill="x", padx=PAD, pady=(4, PAD))

        # Build the log pane next; it also packs side="bottom" so it sits
        # immediately above the status bar regardless of tab content size.
        self._build_log_pane()

        # Notebook fills whatever's left between the header and the
        # bottom-anchored log + status widgets.
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=PAD, pady=(PAD // 2, 0))

        self.tab_settings = ttk.Frame(self.nb, padding=PAD)
        self.tab_prep = ttk.Frame(self.nb, padding=PAD)
        self.tab_caption = ttk.Frame(self.nb, padding=PAD)
        self.tab_review = ttk.Frame(self.nb, padding=PAD)
        self.tab_train = ttk.Frame(self.nb, padding=PAD)
        self.tab_generate = ttk.Frame(self.nb, padding=PAD)
        self.tab_storage = ttk.Frame(self.nb, padding=PAD)
        self.tab_video = ttk.Frame(self.nb, padding=PAD)

        # Initial labels; refresh_step_status() adds live counts.
        self.nb.add(self.tab_settings, text="01 · Settings")
        self.nb.add(self.tab_prep, text="02 · Ingest")
        self.nb.add(self.tab_caption, text="03 · Caption")
        self.nb.add(self.tab_review, text="04 · Review")
        self.nb.add(self.tab_train, text="05 · Train")
        self.nb.add(self.tab_generate, text="06 · Generate")
        self.nb.add(self.tab_storage, text="07 · Storage")
        self.nb.add(self.tab_video, text="08 · Video")

        settings_tab.build(self)
        prep_tab.build(self)
        caption_tab.build(self)
        review_tab.build(self)
        train_tab.build(self)
        generate_tab.build(self)
        storage_tab.build(self)
        video_tab.build(self)

        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _build_header(self) -> None:
        """Compact single-strip header with title, project combo, recent menu, status dots."""
        t = self.theme
        PAD = gui_theme.PAD

        bar = ttk.Frame(self.root, style="Panel.TFrame")
        bar.pack(fill="x", padx=PAD, pady=(PAD, 0))

        # Left: title block.
        left = ttk.Frame(bar, style="Panel.TFrame")
        left.pack(side="left", padx=(PAD, PAD * 2), pady=PAD // 2)

        ttk.Label(left, text="Image Trainer", style="Title.TLabel").pack(
            side="left", padx=(0, PAD)
        )
        ttk.Label(left, text="· a darkroom for SDXL portraits", style="SubHeader.TLabel").pack(
            side="left"
        )

        # Middle: project controls.
        mid = ttk.Frame(bar, style="Panel.TFrame")
        mid.pack(side="left", fill="x", expand=True, padx=PAD, pady=PAD // 2)

        self.project_combo = ttk.Combobox(mid, state="readonly", width=32)
        self.project_combo.pack(side="left")
        self.project_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_project_selected()
        )

        ttk.Button(mid, text="New…", style="Ghost.TButton",
                   command=self._new_project).pack(side="left", padx=(PAD // 2, 0))
        ttk.Button(mid, text="Refresh", style="Ghost.TButton",
                   command=self._refresh_project_list).pack(side="left", padx=(PAD // 2, 0))

        # Recent projects menubutton
        self.recent_mb = ttk.Menubutton(mid, text="Recent ▾", style="Ghost.TButton")
        self.recent_menu = tk.Menu(self.recent_mb, tearoff=0)
        self.recent_mb.configure(menu=self.recent_menu)
        self.recent_mb.pack(side="left", padx=(PAD // 2, 0))
        self._rebuild_recent_menu()

        ttk.Button(mid, text="Root…", style="Ghost.TButton",
                   command=self._change_projects_root).pack(side="left", padx=(PAD // 2, 0))

        # Right: status dots — one per wizard step.
        right = ttk.Frame(bar, style="Panel.TFrame")
        right.pack(side="right", padx=PAD, pady=PAD // 2)
        ttk.Label(right, text="status", style="SubHeader.TLabel",
                  background=t.BG_PANEL).pack(side="left", padx=(0, PAD // 2))
        self.step_dots: dict[str, StatusDot] = {}
        for key, _label in STEP_LABELS:
            dot = StatusDot(right, state="pending")
            dot.pack(side="left", padx=2)
            self.step_dots[key] = dot

        # Thin amber rule under the header.
        rule = tk.Frame(self.root, background=t.ACCENT_AMBER, height=1,
                        highlightthickness=0, borderwidth=0)
        rule.pack(fill="x", padx=PAD)

    def _build_log_pane(self) -> None:
        t = self.theme
        PAD = gui_theme.PAD

        self.log_pane = CollapsibleFrame(self.root, text="Telemetry", start_open=True)
        # Bottom-anchored so the notebook can't push it off-screen when a
        # tab's content is taller than the window. See _build_ui's
        # pack-order comment for why this matters.
        self.log_pane.pack(side="bottom", fill="both", expand=False, padx=PAD, pady=(PAD, 0))

        self.log = scrolledtext.ScrolledText(
            self.log_pane.body, height=10, state="disabled",
            background=t.BG_PANEL, foreground=t.TEXT_SECONDARY,
            insertbackground=t.ACCENT_CYAN, font=t.FONT_MONO,
            relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=t.DIVIDER,
        )
        self.log.pack(fill="both", expand=True)

        # Colour tags for log levels; apply via _append_line().
        self.log.tag_configure("error", foreground=t.ACCENT_RED)
        self.log.tag_configure("warn", foreground=t.ACCENT_GOLD)
        self.log.tag_configure("info", foreground=t.ACCENT_CYAN)
        self.log.tag_configure("step", foreground=t.ACCENT_AMBER)
        self.log.tag_configure("meta", foreground=t.TEXT_MUTED)

    # ---- public helpers used by tab modules ----

    def require_project(self) -> Optional[Project]:
        if not self.current_project:
            messagebox.showerror(
                "No project loaded",
                "Pick a project from the dropdown at the top of the window, "
                "or click 'New…' to create one before running this step.",
            )
            return None
        return self.current_project

    def save_settings_silent(self) -> None:
        """Persist the UI's settings values into the current project without
        prompting. Used by every CLI-dispatching handler so subprocess calls
        see the latest on-disk config. No dialog, no diff."""
        if not self.current_project:
            return
        p = self.current_project
        try:
            p.trigger_word = self.trigger_var.get().strip() or p.trigger_word
            base = self.base_model_var.get().strip()
            p.base_model_path = Path(base) if base else None
            p.resolution = int(self.resolution_var.get())
            p.lora_rank = int(self.lora_rank_var.get())
            p.lora_alpha = p.lora_rank
            p.gradient_accumulation_steps = int(self.grad_accum_var.get())
            p.max_train_steps = int(self.max_steps_var.get())
            p.checkpointing_steps = int(self.checkpointing_steps_var.get())
            p.validation_steps = int(self.validation_steps_var.get())
            p.use_xformers = bool(self.xformers_var.get())
            p.train_text_encoder = bool(self.te_lora_var.get())
            p.face_aware_crop = bool(self.face_aware_var.get())
        except ValueError as e:
            messagebox.showerror("Invalid value", str(e))
            return
        p.save()

    def spawn(self, args: list[str]) -> None:
        try:
            self.runner.start(args)
            self.status_var.set(f"running: {args[0]}")
        except RuntimeError:
            messagebox.showwarning("Busy", "Another step is still running.")

    # ---- project bar handlers ----

    def _change_projects_root(self) -> None:
        path = filedialog.askdirectory(title="Choose projects root folder")
        if not path:
            return
        self.projects_root = ProjectsRoot(Path(path))
        self.projects_root.ensure()
        self.recent_projects = gui_helpers.load_recent(self.projects_root.root)
        self._rebuild_recent_menu()
        self._refresh_project_list()

    def _refresh_project_list(self) -> None:
        names = [p.name for p in self.projects_root.list_projects()]
        self.project_combo["values"] = names
        if self.current_project and self.current_project.root.name in names:
            self.project_combo.set(self.current_project.root.name)
        elif names:
            self.project_combo.set(names[0])
            self._on_project_selected()
        else:
            self.project_combo.set("")
        self.refresh_step_status()

    def _on_project_selected(self) -> None:
        name = self.project_combo.get()
        if not name:
            return
        path = self.projects_root.root / name
        try:
            self.current_project = Project.load(path)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        self._load_settings_into_ui(self.current_project)
        self.status_var.set(f"loaded {path}")
        self.log_queue.put(f"[Loaded project: {path}]\n")
        self.recent_projects = gui_helpers.touch_recent(
            self.projects_root.root, self.current_project.root
        )
        self._rebuild_recent_menu()
        self.refresh_step_status()

    def _new_project(self) -> None:
        name = gui_helpers.ask_string(self.root, "New project", "Project name (folder-friendly):")
        if not name:
            return
        try:
            project = self.projects_root.create(name)
        except Exception as e:
            messagebox.showerror("Create failed", str(e))
            return
        self.current_project = project
        self._refresh_project_list()
        self.project_combo.set(name)
        self._load_settings_into_ui(project)
        self.status_var.set(f"created {project.root}")
        self.log_queue.put(f"[Created project: {project.root}]\n")
        self.recent_projects = gui_helpers.touch_recent(
            self.projects_root.root, project.root
        )
        self._rebuild_recent_menu()
        self.refresh_step_status()

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.delete(0, "end")
        if not self.recent_projects:
            self.recent_menu.add_command(label="(none yet)", state="disabled")
            return
        for p in self.recent_projects:
            label = Path(p).name or p
            self.recent_menu.add_command(
                label=label, command=lambda pp=p: self._open_recent(pp)
            )
        self.recent_menu.add_separator()
        self.recent_menu.add_command(label="Clear list",
                                     command=self._clear_recent)

    def _open_recent(self, project_path: str) -> None:
        p = Path(project_path)
        if not p.exists():
            messagebox.showinfo("Missing", f"Project folder no longer exists:\n{p}")
            return
        try:
            project = Project.load(p)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        # If the project is under a different projects root, re-point the browser.
        if p.parent != self.projects_root.root:
            self.projects_root = ProjectsRoot(p.parent)
            self.projects_root.ensure()
        self.current_project = project
        self._load_settings_into_ui(project)
        self._refresh_project_list()
        self.project_combo.set(p.name)
        self.recent_projects = gui_helpers.touch_recent(
            self.projects_root.root, project.root
        )
        self._rebuild_recent_menu()
        self.refresh_step_status()
        self.log_queue.put(f"[Opened recent: {project.root}]\n")

    def _clear_recent(self) -> None:
        p = gui_helpers.recent_path(self.projects_root.root)
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass
        self.recent_projects = []
        self._rebuild_recent_menu()

    # ---- settings sync ----

    def _load_settings_into_ui(self, project: Project) -> None:
        self.trigger_var.set(project.trigger_word)
        self.base_model_var.set(str(project.base_model_path) if project.base_model_path else "")
        self.resolution_var.set(str(project.resolution))
        self.lora_rank_var.set(str(project.lora_rank))
        self.grad_accum_var.set(str(project.gradient_accumulation_steps))
        self.max_steps_var.set(str(project.max_train_steps))
        self.checkpointing_steps_var.set(str(project.checkpointing_steps))
        self.validation_steps_var.set(str(project.validation_steps))
        self.xformers_var.set(project.use_xformers)
        self.te_lora_var.set(project.train_text_encoder)
        self.face_aware_var.set(project.face_aware_crop)
        # Caption tab settings — persisted on Project so they survive across
        # GUI restarts. Guarded with hasattr because the caption tab might
        # not have built its widgets yet during an early-init load.
        if hasattr(self, "captioner_var"):
            self.captioner_var.set(project.captioner)
            self.general_threshold_var.set(str(project.caption_general_threshold))
            self.character_threshold_var.set(str(project.caption_character_threshold))
            self.caption_suffix_var.set(project.caption_extra_suffix)
            self.caption_nsfw_var.set(project.caption_nsfw_preset)
        if not self.negative_var.get().strip():
            self.negative_var.set(project.default_negative_prompt)
        # Force Review tab to rebuild for this project next time it's shown.
        if self.review_state is not None:
            self.review_state.review = None

    # ---- step status ----

    def refresh_step_status(self) -> None:
        """Recompute dot colours + notebook tab suffixes from project state."""
        project = self.current_project
        if project is None:
            for dot in self.step_dots.values():
                dot.set_state("pending")
            return

        # SETTINGS — always "done" when a project is loaded.
        self.step_dots["settings"].set_state("done")

        # PREP — based on processed/ count.
        processed = list(project.processed_dir.glob("*.png")) if project.processed_dir.exists() else []
        n_processed = len(processed)
        if n_processed == 0:
            self.step_dots["prep"].set_state("pending")
            self._set_tab_text(1, "02 · Ingest")
        else:
            self.step_dots["prep"].set_state("done")
            self._set_tab_text(1, f"02 · Ingest · {n_processed} imgs")

        # CAPTION — based on .txt file count vs PNGs.
        n_captions = sum(1 for p in processed if p.with_suffix(".txt").exists())
        if n_processed == 0:
            self.step_dots["caption"].set_state("pending")
            self._set_tab_text(2, "03 · Caption")
        elif n_captions == 0:
            self.step_dots["caption"].set_state("pending")
            self._set_tab_text(2, "03 · Caption")
        elif n_captions < n_processed:
            self.step_dots["caption"].set_state("warn")
            self._set_tab_text(2, f"03 · Caption · {n_captions}/{n_processed}")
        else:
            self.step_dots["caption"].set_state("done")
            self._set_tab_text(2, f"03 · Caption · {n_captions}/{n_processed}")

        # REVIEW — from review.json summary.
        review_path = project.root / "review.json"
        if review_path.exists():
            try:
                from .pipeline import review as rm
                s = rm.summary(project)
                total, inc = s["total"], s["included"]
                if total == 0:
                    self.step_dots["review"].set_state("pending")
                    self._set_tab_text(3, "04 · Review")
                else:
                    self.step_dots["review"].set_state("done" if inc > 0 else "warn")
                    self._set_tab_text(3, f"04 · Review · {inc}/{total}")
            except Exception:
                self.step_dots["review"].set_state("warn")
                self._set_tab_text(3, "04 · Review · ?")
        else:
            self.step_dots["review"].set_state("pending")
            self._set_tab_text(3, "04 · Review")

        # TRAIN — based on checkpoints/ or lora/ presence.
        ckpts = list(project.checkpoints_dir.glob("step_*")) if project.checkpoints_dir.exists() else []
        lora_unet = project.lora_dir / "unet" if project.lora_dir.exists() else None
        if lora_unet and lora_unet.exists():
            self.step_dots["train"].set_state("done")
            self._set_tab_text(4, "05 · Train · done")
        elif ckpts:
            self.step_dots["train"].set_state("warn")
            self._set_tab_text(4, f"05 · Train · ckpt({len(ckpts)})")
        else:
            self.step_dots["train"].set_state("pending")
            self._set_tab_text(4, "05 · Train")

        # GENERATE — outputs/ presence.
        outs = []
        if project.outputs_dir.exists():
            outs = [p for p in project.outputs_dir.iterdir() if p.is_dir()]
        if outs:
            self.step_dots["generate"].set_state("done")
            self._set_tab_text(5, f"06 · Generate · {len(outs)} runs")
        else:
            self.step_dots["generate"].set_state("pending")
            self._set_tab_text(5, "06 · Generate")

    def _set_tab_text(self, index: int, text: str) -> None:
        try:
            self.nb.tab(index, text=text)
        except tk.TclError:
            pass

    # ---- tab switching ----

    def _on_tab_changed(self, _event) -> None:
        selected = self.nb.select()
        if selected == str(self.tab_review):
            if self.review_state is not None and self.current_project is not None:
                self.review_state.on_tab_enter()
        else:
            if self.review_state is not None:
                self.review_state.on_tab_leave()
        if selected == str(self.tab_storage):
            if self.storage_state is not None and self.current_project is not None:
                self.storage_state.on_tab_enter()
        self.refresh_step_status()

    # ---- log pump ----

    def _drain_log(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._append_log_line(line)
                self._route_progress(line)
        except queue.Empty:
            pass
        # Watch for subprocess exit to fire any queued one-shot.
        if not self.runner.is_running():
            if self.status_var.get().startswith("running:"):
                self.status_var.set("ready")
                self.refresh_step_status()
            if self.on_next_exit is not None:
                try:
                    self.on_next_exit()
                finally:
                    self.on_next_exit = None
        # Let the Train tab drive its live tick even when no line arrived.
        if self.train_state is not None:
            self.train_state.tick()
        self.root.after(100, self._drain_log)

    def _append_log_line(self, line: str) -> None:
        tag = self._tag_for(line)
        self.log.configure(state="normal")
        self.log.insert("end", line, (tag,) if tag else ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def _tag_for(self, line: str) -> Optional[str]:
        low = line.lower()
        if line.startswith("$ ") or line.startswith("[") and line.rstrip().endswith("]"):
            return "meta"
        if "error" in low or "traceback" in low or line.startswith("[exit ") and line[7:8] != "0":
            return "error"
        if "warn" in low or "warning" in low:
            return "warn"
        if line.startswith("step ") or line.startswith("caching "):
            return "step"
        return None

    def _route_progress(self, line: str) -> None:
        if self.train_state is not None:
            self.train_state.on_progress_line(line)
        # Generate-tab progress runs off the same log queue. Cheap no-op
        # when no generate run is active.
        gs = getattr(self, "generate_state", None)
        if gs is not None:
            try:
                gs.on_progress_line(line)
            except Exception:
                pass


_TAB_INDEX = {
    "settings": 0, "prep": 1, "caption": 2, "review": 3, "train": 4, "generate": 5,
    "storage": 6, "video": 7,
}


def launch(
    initial_project_dir: Optional[Path] = None,
    initial_tab: Optional[str] = None,
) -> None:
    """Create the Tk root and run the main loop."""
    root = tk.Tk()
    gui = TrainerGUI(root)

    if initial_project_dir is not None:
        try:
            project = Project.load(initial_project_dir)
        except Exception as e:
            gui.log_queue.put(f"[Could not open {initial_project_dir}: {e}]\n")
        else:
            parent = project.root.parent
            if parent != gui.projects_root.root:
                gui.projects_root = ProjectsRoot(parent)
                gui.projects_root.ensure()
            gui.current_project = project
            gui._load_settings_into_ui(project)
            gui._refresh_project_list()
            gui.project_combo.set(project.root.name)
            gui.recent_projects = gui_helpers.touch_recent(
                gui.projects_root.root, project.root
            )
            gui._rebuild_recent_menu()
            gui.refresh_step_status()
            gui.log_queue.put(f"[Opened project via CLI: {project.root}]\n")

    if initial_tab and initial_tab in _TAB_INDEX:
        root.after(300, lambda: gui.nb.select(_TAB_INDEX[initial_tab]))

    root.mainloop()
