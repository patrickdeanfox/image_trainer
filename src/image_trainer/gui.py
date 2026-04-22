"""Tkinter wizard for image_trainer.

Top bar is a project browser. The rest of the window is a 6-step notebook:
Settings → Import & Resize → Caption → Review → Train → Generate.

All heavy work is dispatched to the `trainer` CLI in a subprocess; the Popen
handle is stored on the instance so the GUI can send SIGINT for a real
"Stop (graceful)" action, and stdout is tailed into the shared log pane so
nothing blocks the UI thread.

Styling: ttk "clam" theme + consistent padding + section separators. Kept
minimal on purpose (no external assets).
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

from .config import DEFAULT_PROJECTS_ROOT, Project, ProjectsRoot
from .pipeline import insights, review as review_mod


# ---------- subprocess runner ----------

class CLIRunner:
    """Manages the currently-running ``trainer`` subprocess for the GUI.

    A single :class:`CLIRunner` is owned by the :class:`TrainerGUI` instance
    and reused across actions (prep / caption / train / generate). Holding
    the :class:`subprocess.Popen` on ``self`` lets the Stop (graceful)
    button deliver a real SIGINT — something the earlier design couldn't do
    because ``Popen`` lived only inside a worker thread's local scope.

    A background thread tails the subprocess's combined stdout/stderr and
    forwards every line to :attr:`log_queue`; the Tk main loop drains the
    queue on a timer (:meth:`TrainerGUI._drain_log`) so the UI never blocks.
    """

    def __init__(self, log_queue: "queue.Queue[str]") -> None:
        self.log_queue = log_queue
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        """True while the pump thread is still alive (i.e. the subprocess
        hasn't exited and been joined)."""
        return self.thread is not None and self.thread.is_alive()

    def start(self, args: list[str]) -> None:
        """Spawn ``python -m image_trainer.cli <args>`` and start the pump thread.

        Raises :class:`RuntimeError` if another job is already running — the
        GUI surfaces this as a "Busy" message box rather than quietly
        queueing, because two training jobs on the same project at once
        would clobber each other's checkpoints.
        """
        if self.is_running():
            raise RuntimeError("another step is still running")
        cmd = [sys.executable, "-m", "image_trainer.cli", *args]
        self.log_queue.put(f"$ {' '.join(cmd)}\n")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _pump() -> None:
            assert self.proc is not None
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log_queue.put(line)
            self.proc.wait()
            self.log_queue.put(f"[exit {self.proc.returncode}]\n")

        self.thread = threading.Thread(target=_pump, daemon=True)
        self.thread.start()

    def stop_graceful(self) -> bool:
        """Signal the running subprocess to checkpoint and exit cleanly.

        The training loop traps SIGINT/SIGTERM and writes a final checkpoint
        before exiting, so this is how the GUI's "Stop (graceful)" button
        preserves partial overnight progress. On Windows we send
        ``CTRL_BREAK_EVENT`` since the CRT doesn't translate SIGINT the
        same way. Returns ``True`` if a signal was delivered.
        """
        if not self.is_running() or self.proc is None:
            return False
        try:
            if sys.platform.startswith("win"):
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                self.proc.send_signal(signal.SIGINT)
            return True
        except Exception as e:
            self.log_queue.put(f"[stop failed: {e}]\n")
            return False


# ---------- styling ----------

#: Uniform widget padding in pixels. Used everywhere so adjusting it in one
#: place rescales the whole layout.
PAD = 8


class _Theme:
    """Central palette and font token set for the GUI.

    LCARS-inspired (Star-Trek-style) obsidian/glass aesthetic: near-black
    stacked surface tiers, thin dividers, a single amber primary, cyan +
    violet secondaries, and a gold for caution notices. All colours are
    hex strings so they flow cleanly into both `ttk.Style` configure
    calls and the handful of `tk.*` widgets that don't honour ttk styles
    (Text, Listbox, ScrolledText).

    Keep this class the single source of truth — no ad-hoc hex strings
    scattered through the layout code.
    """

    # Surface tiers — each step slightly lighter than the one below,
    # simulating layered glass on an obsidian console.
    BG_ROOT = "#05070D"
    BG_PANEL = "#0B1020"
    BG_ELEVATED = "#121A33"
    BG_INPUT = "#0E1526"
    BG_HOVER = "#1A2340"
    BG_PRESSED = "#22305A"

    # Primary/secondary/tertiary accents. Amber is reserved for the
    # dominant call-to-action and for the active tab underline.
    ACCENT_AMBER = "#FFB74D"
    ACCENT_CYAN = "#6FD6FF"
    ACCENT_VIOLET = "#B79CFF"
    ACCENT_RED = "#FF6B6B"
    ACCENT_GOLD = "#F2C14E"

    # Structural strokes — kept low-contrast on purpose.
    DIVIDER = "#1F2A47"
    BORDER_SOFT = "#2A3656"

    # Text ramp.
    TEXT_PRIMARY = "#E8EEF8"
    TEXT_SECONDARY = "#B7C0D8"
    TEXT_MUTED = "#6B7591"
    TEXT_ON_ACCENT = "#0A0E1A"

    # Fonts. Helvetica Neue is the closest freely-available match to the
    # LCARS `Swiss 911` look. `TkFixedFont` keeps the log pane monospace
    # on any platform even when we can't find a preferred mono face.
    FONT_BODY = ("Helvetica Neue", 10)
    FONT_HEADER = ("Helvetica Neue", 11, "bold")
    FONT_LCARS = ("Helvetica Neue", 10, "bold")
    FONT_DISPLAY = ("Helvetica Neue", 16, "bold")
    FONT_MONO = ("JetBrains Mono", 10)


THEME = _Theme()


def _apply_style(root: tk.Tk) -> None:
    """Apply the glass/LCARS visual language to every ttk and raw-tk widget.

    Two-part strategy:

    1. ``ttk.Style`` configures every ``T*`` widget class the GUI uses
       (frames, labels, buttons, notebook, entries, combos, checks,
       labelframes, progress bars, scrollbars).
    2. ``root.option_add`` installs defaults for the handful of raw-tk
       widgets Tk can't reach via styles — ``tk.Text``, ``tk.Listbox``,
       ``scrolledtext.ScrolledText``. This avoids sprinkling per-widget
       ``background=`` kwargs through the layout code.
    """
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    root.configure(background=THEME.BG_ROOT)

    # ---- raw-tk defaults (Text, Listbox, Menu) ----
    root.option_add("*Font", THEME.FONT_BODY)
    root.option_add("*background", THEME.BG_ROOT)
    root.option_add("*foreground", THEME.TEXT_PRIMARY)

    for w in ("Text", "Listbox", "ScrolledText"):
        root.option_add(f"*{w}.background", THEME.BG_INPUT)
        root.option_add(f"*{w}.foreground", THEME.TEXT_PRIMARY)
        root.option_add(f"*{w}.borderWidth", 0)
        root.option_add(f"*{w}.relief", "flat")
        root.option_add(f"*{w}.highlightThickness", 1)
        root.option_add(f"*{w}.highlightBackground", THEME.DIVIDER)
        root.option_add(f"*{w}.highlightColor", THEME.ACCENT_AMBER)
        root.option_add(f"*{w}.selectBackground", THEME.ACCENT_AMBER)
        root.option_add(f"*{w}.selectForeground", THEME.TEXT_ON_ACCENT)
    root.option_add("*Text.insertBackground", THEME.ACCENT_CYAN)
    root.option_add("*ScrolledText.insertBackground", THEME.ACCENT_CYAN)
    root.option_add("*Listbox.activeStyle", "none")

    root.option_add("*Menu.background", THEME.BG_ELEVATED)
    root.option_add("*Menu.foreground", THEME.TEXT_PRIMARY)
    root.option_add("*Menu.activeBackground", THEME.ACCENT_AMBER)
    root.option_add("*Menu.activeForeground", THEME.TEXT_ON_ACCENT)
    root.option_add("*Menu.borderWidth", 0)

    # ---- frames ----
    style.configure("TFrame", background=THEME.BG_ROOT)
    style.configure("Panel.TFrame", background=THEME.BG_PANEL)
    style.configure("Elevated.TFrame", background=THEME.BG_ELEVATED)

    # ---- labels ----
    style.configure(
        "TLabel",
        background=THEME.BG_ROOT,
        foreground=THEME.TEXT_PRIMARY,
        font=THEME.FONT_BODY,
        padding=(2, 2),
    )
    style.configure(
        "Header.TLabel",
        background=THEME.BG_ROOT,
        foreground=THEME.ACCENT_AMBER,
        font=THEME.FONT_HEADER,
    )
    style.configure(
        "Display.TLabel",
        background=THEME.BG_ROOT,
        foreground=THEME.ACCENT_AMBER,
        font=THEME.FONT_DISPLAY,
    )
    style.configure(
        "SubHeader.TLabel",
        background=THEME.BG_ROOT,
        foreground=THEME.ACCENT_CYAN,
        font=THEME.FONT_LCARS,
    )
    style.configure(
        "Status.TLabel",
        background=THEME.BG_ROOT,
        foreground=THEME.TEXT_MUTED,
        font=THEME.FONT_BODY,
    )
    style.configure(
        "Warn.TLabel",
        background=THEME.BG_ROOT,
        foreground=THEME.ACCENT_GOLD,
        font=THEME.FONT_BODY,
    )
    style.configure(
        "Preview.TLabel",
        background=THEME.BG_INPUT,
        foreground=THEME.TEXT_MUTED,
        borderwidth=1,
        relief="flat",
    )

    # ---- buttons ----
    style.configure(
        "TButton",
        background=THEME.BG_ELEVATED,
        foreground=THEME.ACCENT_AMBER,
        font=THEME.FONT_LCARS,
        padding=(14, 6),
        borderwidth=0,
        relief="flat",
        focuscolor=THEME.BG_ELEVATED,
    )
    style.map(
        "TButton",
        background=[
            ("pressed", THEME.BG_PRESSED),
            ("active", THEME.BG_HOVER),
            ("disabled", THEME.BG_PANEL),
        ],
        foreground=[
            ("disabled", THEME.TEXT_MUTED),
            ("pressed", THEME.TEXT_PRIMARY),
            ("active", THEME.ACCENT_CYAN),
        ],
    )
    # Primary call-to-action: solid amber fill.
    style.configure(
        "Primary.TButton",
        background=THEME.ACCENT_AMBER,
        foreground=THEME.TEXT_ON_ACCENT,
        font=THEME.FONT_LCARS,
        padding=(18, 7),
        borderwidth=0,
        relief="flat",
        focuscolor=THEME.ACCENT_AMBER,
    )
    style.map(
        "Primary.TButton",
        background=[
            ("pressed", THEME.ACCENT_VIOLET),
            ("active", THEME.ACCENT_CYAN),
            ("disabled", THEME.BG_PANEL),
        ],
        foreground=[
            ("disabled", THEME.TEXT_MUTED),
            ("active", THEME.TEXT_ON_ACCENT),
            ("pressed", THEME.TEXT_ON_ACCENT),
        ],
    )
    # Caution action: gold fill; reserved for the graceful-stop button.
    style.configure(
        "Caution.TButton",
        background=THEME.BG_ELEVATED,
        foreground=THEME.ACCENT_RED,
        font=THEME.FONT_LCARS,
        padding=(14, 6),
        borderwidth=0,
        relief="flat",
    )
    style.map(
        "Caution.TButton",
        background=[("active", THEME.BG_HOVER), ("pressed", THEME.BG_PRESSED)],
        foreground=[("active", THEME.ACCENT_AMBER)],
    )

    # ---- entries / comboboxes ----
    style.configure(
        "TEntry",
        fieldbackground=THEME.BG_INPUT,
        foreground=THEME.TEXT_PRIMARY,
        insertcolor=THEME.ACCENT_CYAN,
        bordercolor=THEME.DIVIDER,
        lightcolor=THEME.DIVIDER,
        darkcolor=THEME.DIVIDER,
        borderwidth=1,
        relief="flat",
        padding=(8, 5),
    )
    style.map(
        "TEntry",
        bordercolor=[("focus", THEME.ACCENT_AMBER)],
        lightcolor=[("focus", THEME.ACCENT_AMBER)],
        darkcolor=[("focus", THEME.ACCENT_AMBER)],
    )

    style.configure(
        "TCombobox",
        fieldbackground=THEME.BG_INPUT,
        background=THEME.BG_ELEVATED,
        foreground=THEME.TEXT_PRIMARY,
        arrowcolor=THEME.ACCENT_AMBER,
        bordercolor=THEME.DIVIDER,
        lightcolor=THEME.DIVIDER,
        darkcolor=THEME.DIVIDER,
        borderwidth=1,
        relief="flat",
        padding=(6, 4),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", THEME.BG_INPUT)],
        foreground=[("readonly", THEME.TEXT_PRIMARY), ("disabled", THEME.TEXT_MUTED)],
        bordercolor=[("focus", THEME.ACCENT_AMBER)],
        arrowcolor=[("active", THEME.ACCENT_CYAN)],
    )
    # The drop-down popdown uses tk listbox styling via option_add above.

    # ---- checkbutton ----
    style.configure(
        "TCheckbutton",
        background=THEME.BG_ROOT,
        foreground=THEME.TEXT_PRIMARY,
        indicatorbackground=THEME.BG_INPUT,
        indicatorforeground=THEME.ACCENT_AMBER,
        focuscolor=THEME.BG_ROOT,
        padding=(4, 2),
    )
    style.map(
        "TCheckbutton",
        background=[("active", THEME.BG_ROOT)],
        foreground=[("active", THEME.ACCENT_AMBER), ("disabled", THEME.TEXT_MUTED)],
        indicatorcolor=[
            ("selected", THEME.ACCENT_AMBER),
            ("!selected", THEME.BG_INPUT),
        ],
    )

    # ---- labelframe ----
    style.configure(
        "TLabelframe",
        background=THEME.BG_ROOT,
        bordercolor=THEME.DIVIDER,
        lightcolor=THEME.DIVIDER,
        darkcolor=THEME.DIVIDER,
        borderwidth=1,
        relief="flat",
        padding=PAD,
    )
    style.configure(
        "TLabelframe.Label",
        background=THEME.BG_ROOT,
        foreground=THEME.ACCENT_CYAN,
        font=THEME.FONT_LCARS,
    )

    # ---- notebook — flat tabs, amber on active ----
    style.configure(
        "TNotebook",
        background=THEME.BG_ROOT,
        bordercolor=THEME.BG_ROOT,
        lightcolor=THEME.BG_ROOT,
        darkcolor=THEME.BG_ROOT,
        borderwidth=0,
        tabmargins=(0, 4, 0, 0),
    )
    style.configure(
        "TNotebook.Tab",
        background=THEME.BG_PANEL,
        foreground=THEME.TEXT_SECONDARY,
        font=THEME.FONT_LCARS,
        padding=(22, 9),
        borderwidth=0,
        focuscolor=THEME.BG_PANEL,
    )
    style.map(
        "TNotebook.Tab",
        background=[
            ("selected", THEME.BG_ELEVATED),
            ("active", THEME.BG_HOVER),
        ],
        foreground=[
            ("selected", THEME.ACCENT_AMBER),
            ("active", THEME.ACCENT_CYAN),
        ],
    )

    # ---- progressbar ----
    style.configure(
        "Trainer.Horizontal.TProgressbar",
        troughcolor=THEME.BG_INPUT,
        background=THEME.ACCENT_AMBER,
        bordercolor=THEME.BG_INPUT,
        lightcolor=THEME.ACCENT_AMBER,
        darkcolor=THEME.ACCENT_AMBER,
        thickness=14,
    )

    # ---- scrollbars ----
    for orient in ("Vertical", "Horizontal"):
        style.configure(
            f"{orient}.TScrollbar",
            background=THEME.BG_PANEL,
            troughcolor=THEME.BG_ROOT,
            bordercolor=THEME.BG_ROOT,
            arrowcolor=THEME.ACCENT_AMBER,
            lightcolor=THEME.BG_PANEL,
            darkcolor=THEME.BG_PANEL,
            borderwidth=0,
            relief="flat",
        )
        style.map(
            f"{orient}.TScrollbar",
            background=[("active", THEME.BG_HOVER)],
            arrowcolor=[("active", THEME.ACCENT_CYAN)],
        )

    # ---- separators ----
    style.configure("TSeparator", background=THEME.DIVIDER)


# ---------- GUI ----------

class TrainerGUI:
    """The main Tkinter window.

    Owns the :class:`CLIRunner`, the shared :mod:`queue.Queue` that the
    subprocess pump writes into, and all per-tab widgets. Every button
    handler on this class is either a layout builder (``_build_*_tab``) or
    a thin dispatcher that constructs a CLI argv list and calls
    :meth:`_spawn`. The GUI intentionally owns no model, training, or
    review persistence logic — that lives in the CLI and pipeline modules,
    so the two surfaces never drift.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("IMAGE TRAINER · OPERATIONS")
        self.root.geometry("1040x860")
        self.root.minsize(940, 720)

        _apply_style(self.root)

        self.projects_root = ProjectsRoot(DEFAULT_PROJECTS_ROOT)
        self.projects_root.ensure()
        self.current_project: Optional[Project] = None

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.runner = CLIRunner(self.log_queue)

        self._build_ui()
        self.root.after(100, self._drain_log)
        self.root.after(200, self._refresh_project_list)

    # ---- layout ----

    def _build_ui(self) -> None:
        self._build_header_bar()
        self._build_project_bar()

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=PAD, pady=(0, 0))

        self.tab_settings = ttk.Frame(self.nb, padding=PAD)
        self.tab_prep = ttk.Frame(self.nb, padding=PAD)
        self.tab_caption = ttk.Frame(self.nb, padding=PAD)
        self.tab_review = ttk.Frame(self.nb, padding=PAD)
        self.tab_train = ttk.Frame(self.nb, padding=PAD)
        self.tab_generate = ttk.Frame(self.nb, padding=PAD)

        self.nb.add(self.tab_settings, text="01 · SETTINGS")
        self.nb.add(self.tab_prep, text="02 · INGEST")
        self.nb.add(self.tab_caption, text="03 · CAPTION")
        self.nb.add(self.tab_review, text="04 · REVIEW")
        self.nb.add(self.tab_train, text="05 · TRAIN")
        self.nb.add(self.tab_generate, text="06 · GENERATE")

        self._build_settings_tab()
        self._build_prep_tab()
        self._build_caption_tab()
        self._build_review_tab()
        self._build_train_tab()
        self._build_generate_tab()

        # Refresh the Review tab whenever the user switches to it.
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        log_frame = ttk.LabelFrame(self.root, text="TELEMETRY", padding=PAD)
        log_frame.pack(fill="both", expand=True, padx=PAD, pady=(0, 0))
        self.log = scrolledtext.ScrolledText(
            log_frame, height=10, state="disabled",
            background=THEME.BG_PANEL, foreground=THEME.TEXT_SECONDARY,
            insertbackground=THEME.ACCENT_CYAN, font=("TkFixedFont", 10),
            relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=THEME.DIVIDER,
        )
        self.log.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="READY")
        status = ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel", anchor="w")
        status.pack(fill="x", padx=PAD, pady=(4, PAD))

    def _build_header_bar(self) -> None:
        """LCARS-style top strip: a slim amber rule, a system identifier, and
        two colour-coded marker blocks on the right. Decorative only — no
        widgets here are hooked to commands or vars."""
        bar = ttk.Frame(self.root, style="Panel.TFrame")
        bar.pack(fill="x", padx=PAD, pady=(PAD, 0))

        # Left: display-weight title.
        ttk.Label(
            bar,
            text="IMAGE TRAINER",
            style="Display.TLabel",
            background=THEME.BG_PANEL,
        ).pack(side="left", padx=(PAD, PAD * 2), pady=(PAD // 2, PAD // 2))

        ttk.Label(
            bar,
            text="LOCAL SDXL LORA · OPERATIONS CONSOLE",
            style="SubHeader.TLabel",
            background=THEME.BG_PANEL,
        ).pack(side="left", pady=(PAD // 2, PAD // 2))

        # Right: decorative LCARS marker pills.
        right = ttk.Frame(bar, style="Panel.TFrame")
        right.pack(side="right", padx=PAD, pady=PAD // 2)
        for color in (THEME.ACCENT_VIOLET, THEME.ACCENT_CYAN, THEME.ACCENT_AMBER):
            pill = tk.Frame(right, background=color, width=28, height=10,
                            highlightthickness=0, borderwidth=0)
            pill.pack(side="left", padx=(0, 4))
            pill.pack_propagate(False)

        # Thin amber rule under the header.
        rule = tk.Frame(self.root, background=THEME.ACCENT_AMBER, height=1,
                        highlightthickness=0, borderwidth=0)
        rule.pack(fill="x", padx=PAD)

    def _build_project_bar(self) -> None:
        bar = ttk.LabelFrame(self.root, text="PROJECTS", padding=PAD)
        bar.pack(fill="x", padx=PAD, pady=PAD)

        ttk.Label(bar, text="Projects folder:").grid(row=0, column=0, sticky="w")
        self.projects_root_var = tk.StringVar(value=str(self.projects_root.root))
        ttk.Entry(bar, textvariable=self.projects_root_var, width=55).grid(
            row=0, column=1, padx=PAD, sticky="we"
        )
        ttk.Button(bar, text="Browse...", command=self._change_projects_root).grid(row=0, column=2)

        ttk.Label(bar, text="Project:").grid(row=1, column=0, sticky="w", pady=(PAD, 0))
        self.project_combo = ttk.Combobox(bar, state="readonly", width=52)
        self.project_combo.grid(row=1, column=1, padx=PAD, sticky="we", pady=(PAD, 0))
        self.project_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_project_selected())

        actions = ttk.Frame(bar)
        actions.grid(row=1, column=2, pady=(PAD, 0))
        ttk.Button(actions, text="New...", command=self._new_project).pack(side="left")
        ttk.Button(actions, text="Refresh", command=self._refresh_project_list).pack(side="left", padx=(PAD // 2, 0))

        bar.columnconfigure(1, weight=1)

    def _build_settings_tab(self) -> None:
        f = self.tab_settings
        f.columnconfigure(1, weight=1)

        self.trigger_var = tk.StringVar()
        self.base_model_var = tk.StringVar()
        self.resolution_var = tk.StringVar(value="1024")
        self.lora_rank_var = tk.StringVar(value="32")
        self.grad_accum_var = tk.StringVar(value="1")
        self.max_steps_var = tk.StringVar(value="1500")
        self.checkpointing_steps_var = tk.StringVar(value="100")
        self.validation_steps_var = tk.StringVar(value="200")
        self.xformers_var = tk.BooleanVar(value=True)
        self.te_lora_var = tk.BooleanVar(value=False)
        self.face_aware_var = tk.BooleanVar(value=True)

        ttk.Label(f, text="SUBJECT · BASE MODEL", style="Header.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, PAD)
        )

        ttk.Label(f, text="Trigger word:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(f, textvariable=self.trigger_var, width=30).grid(row=1, column=1, sticky="w", padx=PAD)

        ttk.Label(f, text="Base SDXL checkpoint:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(f, textvariable=self.base_model_var).grid(row=2, column=1, sticky="we", padx=PAD)
        ttk.Button(f, text="Browse...", command=self._pick_base_model).grid(row=2, column=2, padx=PAD)

        # --- OOM / quality knobs ---
        oom = ttk.LabelFrame(f, text="OOM · QUALITY KNOBS", padding=PAD)
        oom.grid(row=3, column=0, columnspan=3, sticky="we", pady=(PAD * 2, PAD))
        oom.columnconfigure(1, weight=0)
        oom.columnconfigure(3, weight=1)

        ttk.Label(oom, text="Resolution:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(
            oom, textvariable=self.resolution_var,
            values=["512", "768", "1024"], state="readonly", width=8,
        ).grid(row=0, column=1, sticky="w", padx=PAD)

        ttk.Label(oom, text="LoRA rank:").grid(row=0, column=2, sticky="w", pady=2)
        ttk.Combobox(
            oom, textvariable=self.lora_rank_var,
            values=["8", "16", "32", "64"], state="readonly", width=6,
        ).grid(row=0, column=3, sticky="w", padx=PAD)

        ttk.Label(oom, text="Grad accumulation:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Combobox(
            oom, textvariable=self.grad_accum_var,
            values=["1", "2", "4", "8"], state="readonly", width=6,
        ).grid(row=1, column=1, sticky="w", padx=PAD)

        ttk.Checkbutton(oom, text="xformers", variable=self.xformers_var).grid(
            row=1, column=2, sticky="w", padx=PAD
        )
        ttk.Checkbutton(
            oom, text="Text-encoder LoRA (higher quality, slower)",
            variable=self.te_lora_var,
        ).grid(row=1, column=3, sticky="w", padx=PAD)

        # --- schedule ---
        sched = ttk.LabelFrame(f, text="TRAINING LENGTH", padding=PAD)
        sched.grid(row=4, column=0, columnspan=3, sticky="we", pady=(0, PAD))

        ttk.Label(sched, text="Max steps:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(sched, textvariable=self.max_steps_var, width=10).grid(row=0, column=1, sticky="w", padx=PAD)
        ttk.Label(sched, text="Checkpoint every:").grid(row=0, column=2, sticky="w", pady=2)
        ttk.Entry(sched, textvariable=self.checkpointing_steps_var, width=10).grid(row=0, column=3, sticky="w", padx=PAD)
        ttk.Label(sched, text="Validation every (0 = off):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(sched, textvariable=self.validation_steps_var, width=10).grid(row=1, column=1, sticky="w", padx=PAD)

        ttk.Button(f, text="Save settings", style="Primary.TButton", command=self._save_settings).grid(
            row=5, column=1, sticky="w", padx=PAD, pady=PAD
        )

    def _build_prep_tab(self) -> None:
        f = self.tab_prep
        f.columnconfigure(1, weight=1)
        self.source_dir_var = tk.StringVar()

        ttk.Label(f, text="INGEST · RESIZE TO 1024 × 1024", style="Header.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, PAD)
        )
        ttk.Label(f, text="Source folder:").grid(row=1, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.source_dir_var).grid(row=1, column=1, padx=PAD, sticky="we")
        ttk.Button(f, text="Browse...", command=self._pick_source_dir).grid(row=1, column=2)
        ttk.Checkbutton(
            f,
            text=(
                "Face-aware crop (rule of thirds) \u2014 detects the subject "
                "and places the face on a third-line intersection. Images "
                "with no detectable face are marked excluded for review."
            ),
            variable=self.face_aware_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=0, pady=(PAD, 0))
        ttk.Button(f, text="Import & resize", style="Primary.TButton", command=self._on_prep).grid(
            row=3, column=1, sticky="w", padx=PAD, pady=PAD
        )

    def _build_caption_tab(self) -> None:
        f = self.tab_caption
        ttk.Label(f, text="CAPTION · PROCESSED IMAGES", style="Header.TLabel").grid(
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
        ttk.Button(f, text="Run captioning", style="Primary.TButton", command=self._on_caption).grid(
            row=2, column=0, sticky="w", pady=PAD
        )

    def _build_review_tab(self) -> None:
        f = self.tab_review
        f.columnconfigure(0, weight=0)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(1, weight=1)

        # --- top bar: header + nav + counts ---
        top = ttk.Frame(f)
        top.grid(row=0, column=0, columnspan=2, sticky="we", pady=(0, PAD))
        ttk.Label(top, text="REVIEW · CAPTION EACH IMAGE", style="Header.TLabel").pack(side="left")
        self.review_counts_var = tk.StringVar(value="—")
        ttk.Label(top, textvariable=self.review_counts_var, style="Status.TLabel").pack(side="right")

        # --- left column: file list ---
        left = ttk.Frame(f)
        left.grid(row=1, column=0, sticky="nswe")
        left.rowconfigure(0, weight=1)
        self.review_list = tk.Listbox(
            left, width=22, activestyle="none", exportselection=False,
            background=THEME.BG_INPUT, foreground=THEME.TEXT_PRIMARY,
            selectbackground=THEME.ACCENT_AMBER, selectforeground=THEME.TEXT_ON_ACCENT,
            highlightthickness=1, highlightbackground=THEME.DIVIDER,
            highlightcolor=THEME.ACCENT_AMBER, borderwidth=0, relief="flat",
            font=THEME.FONT_BODY,
        )
        self.review_list.grid(row=0, column=0, sticky="nswe")
        lb_scroll = ttk.Scrollbar(left, orient="vertical", command=self.review_list.yview)
        lb_scroll.grid(row=0, column=1, sticky="ns")
        self.review_list.configure(yscrollcommand=lb_scroll.set)
        self.review_list.bind("<<ListboxSelect>>", self._on_review_list_select)

        list_btns = ttk.Frame(left)
        list_btns.grid(row=1, column=0, columnspan=2, sticky="we", pady=(PAD, 0))
        ttk.Button(list_btns, text="Reload", command=self._reload_review).pack(side="left")
        ttk.Button(list_btns, text="Save", command=self._save_review).pack(side="left", padx=(PAD, 0))

        # --- right column: image + editor ---
        right = ttk.Frame(f)
        right.grid(row=1, column=1, sticky="nswe", padx=(PAD, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # image preview
        self.review_image_label = ttk.Label(
            right, anchor="center", style="Preview.TLabel",
        )
        self.review_image_label.grid(row=0, column=0, sticky="we")

        # caption editor + chips + info
        editor = ttk.Frame(right)
        editor.grid(row=1, column=0, sticky="nswe", pady=(PAD, 0))
        editor.columnconfigure(1, weight=1)
        editor.rowconfigure(2, weight=1)

        self.review_include_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            editor, text="Include in training",
            variable=self.review_include_var,
            command=self._on_review_include_toggle,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(editor, text="Caption:").grid(row=1, column=0, sticky="nw", pady=(PAD, 2))
        self.review_caption_text = tk.Text(
            editor, height=4, wrap="word", font=THEME.FONT_BODY,
            background=THEME.BG_INPUT, foreground=THEME.TEXT_PRIMARY,
            insertbackground=THEME.ACCENT_CYAN, relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=THEME.DIVIDER,
            highlightcolor=THEME.ACCENT_AMBER,
        )
        self.review_caption_text.grid(row=1, column=1, rowspan=2, sticky="nswe", padx=(PAD, 0))
        self.review_caption_text.bind("<FocusOut>", lambda _e: self._capture_current_editor())

        # prompt chips
        chips = ttk.LabelFrame(editor, text="QUICK TAGS", padding=PAD)
        chips.grid(row=3, column=0, columnspan=2, sticky="we", pady=(PAD, 0))
        self.review_chips_frame = chips
        self._rebuild_chips([])

        # per-image info
        info = ttk.LabelFrame(editor, text="IMAGE INFO", padding=PAD)
        info.grid(row=4, column=0, columnspan=2, sticky="we", pady=(PAD, 0))
        info.columnconfigure(1, weight=1)
        self.review_info_var = tk.StringVar(value="—")
        ttk.Label(info, textvariable=self.review_info_var, justify="left").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        self.review_dupes_var = tk.StringVar(value="")
        ttk.Label(
            info, textvariable=self.review_dupes_var, style="Warn.TLabel", justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

        # notes
        ttk.Label(editor, text="Notes:").grid(row=5, column=0, sticky="nw", pady=(PAD, 2))
        self.review_notes_entry = ttk.Entry(editor)
        self.review_notes_entry.grid(row=5, column=1, sticky="we", padx=(PAD, 0), pady=(PAD, 2))

        # nav
        nav = ttk.Frame(right)
        nav.grid(row=2, column=0, sticky="we", pady=(PAD, 0))
        ttk.Button(nav, text="< Prev", command=lambda: self._review_step(-1)).pack(side="left")
        ttk.Button(nav, text="Next >", command=lambda: self._review_step(+1)).pack(side="left", padx=(PAD // 2, 0))
        ttk.Button(nav, text="Toggle include", command=self._toggle_review_include).pack(side="left", padx=(PAD, 0))
        ttk.Label(
            nav,
            text="Shortcuts: ← → (nav), I (toggle include), Ctrl+S (save)",
            style="Status.TLabel",
        ).pack(side="right")

        # Keyboard shortcuts on the tab frame.
        f.bind_all("<Control-s>", lambda _e: self._save_review())

        # state
        self._review: Optional[review_mod.Review] = None
        self._review_order: list[str] = []
        self._review_idx: int = -1
        # Ring buffer of recent PhotoImage refs. Single-slot caching leads to
        # the previous image being GC'd mid-render under fast navigation, so
        # we hold the last few.
        self._review_photo_cache: list = []
        self._review_dupes_index: dict[str, list[tuple[str, int]]] = {}
        self._review_stats_index: dict[str, dict] = {}

    def _build_train_tab(self) -> None:
        f = self.tab_train
        f.columnconfigure(0, weight=1)

        ttk.Label(f, text="TRAIN · LORA", style="Header.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, PAD)
        )
        ttk.Label(
            f,
            text=(
                "Uses settings from the Settings tab. Ctrl+C in the terminal OR the "
                "Stop button save a checkpoint before exiting — safe to resume."
            ),
            justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w")

        note_row = ttk.Frame(f)
        note_row.grid(row=2, column=0, columnspan=4, sticky="we", pady=(PAD, 0))
        ttk.Label(note_row, text="Journal note:").pack(side="left")
        self.train_note_var = tk.StringVar()
        ttk.Entry(note_row, textvariable=self.train_note_var).pack(
            side="left", fill="x", expand=True, padx=(PAD, 0)
        )

        bar_frame = ttk.Frame(f)
        bar_frame.grid(row=3, column=0, columnspan=4, sticky="we", pady=(PAD * 2, PAD // 2))
        bar_frame.columnconfigure(0, weight=1)
        self.train_progress = ttk.Progressbar(
            bar_frame, length=600, mode="determinate",
            style="Trainer.Horizontal.TProgressbar",
        )
        self.train_progress.grid(row=0, column=0, sticky="we")
        self.train_pct_var = tk.StringVar(value="0%")
        ttk.Label(bar_frame, textvariable=self.train_pct_var, width=6, anchor="e").grid(
            row=0, column=1, padx=(PAD, 0)
        )

        self.train_status_var = tk.StringVar(value="idle")
        ttk.Label(f, textvariable=self.train_status_var, style="Status.TLabel").grid(
            row=4, column=0, columnspan=4, sticky="w"
        )

        btns = ttk.Frame(f)
        btns.grid(row=5, column=0, columnspan=4, sticky="w", pady=PAD)
        ttk.Button(btns, text="Start training", style="Primary.TButton", command=self._on_train).pack(side="left")
        ttk.Button(btns, text="Resume training", command=self._on_train_resume).pack(side="left", padx=PAD)
        ttk.Button(btns, text="Stop (graceful)", style="Caution.TButton", command=self._on_train_stop).pack(side="left")

        log_btns = ttk.Frame(f)
        log_btns.grid(row=6, column=0, columnspan=4, sticky="w", pady=PAD)
        ttk.Button(log_btns, text="Open logs folder", command=self._open_logs_dir).pack(side="left")
        ttk.Button(log_btns, text="Open validation previews", command=self._open_validation_dir).pack(side="left", padx=PAD)
        ttk.Button(log_btns, text="Open latest log", command=self._open_latest_log).pack(side="left")
        ttk.Button(log_btns, text="Open journal", command=self._open_journal).pack(side="left", padx=(PAD, 0))

    def _build_generate_tab(self) -> None:
        f = self.tab_generate
        f.columnconfigure(1, weight=1)

        self.prompt_var = tk.StringVar(value="ohwx person, portrait, natural lighting")
        self.negative_var = tk.StringVar(value="")
        self.n_var = tk.StringVar(value="4")
        self.steps_var = tk.StringVar(value="30")
        self.guidance_var = tk.StringVar(value="7.0")
        self.seed_var = tk.StringVar(value="")

        ttk.Label(f, text="GENERATE · TRAINED LORA", style="Header.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, PAD)
        )

        ttk.Label(f, text="Prompt:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(f, textvariable=self.prompt_var).grid(row=1, column=1, columnspan=3, sticky="we", padx=PAD)

        ttk.Label(f, text="Negative:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(f, textvariable=self.negative_var).grid(row=2, column=1, columnspan=3, sticky="we", padx=PAD)

        ttk.Label(f, text="N images:").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(f, textvariable=self.n_var, width=8).grid(row=3, column=1, sticky="w", padx=PAD)
        ttk.Label(f, text="Steps:").grid(row=3, column=2, sticky="w", pady=2)
        ttk.Entry(f, textvariable=self.steps_var, width=8).grid(row=3, column=3, sticky="w", padx=PAD)

        ttk.Label(f, text="Guidance:").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Entry(f, textvariable=self.guidance_var, width=8).grid(row=4, column=1, sticky="w", padx=PAD)
        ttk.Label(f, text="Seed (blank = random):").grid(row=4, column=2, sticky="w", pady=2)
        ttk.Entry(f, textvariable=self.seed_var, width=12).grid(row=4, column=3, sticky="w", padx=PAD)

        btns = ttk.Frame(f)
        btns.grid(row=5, column=0, columnspan=4, sticky="w", pady=PAD)
        ttk.Button(btns, text="Generate", style="Primary.TButton", command=self._on_generate).pack(side="left")
        ttk.Button(btns, text="Open outputs folder", command=self._open_outputs_dir).pack(side="left", padx=PAD)

    # ---- project bar handlers ----

    def _change_projects_root(self) -> None:
        path = filedialog.askdirectory(title="Choose projects root folder")
        if not path:
            return
        self.projects_root = ProjectsRoot(Path(path))
        self.projects_root.ensure()
        self.projects_root_var.set(str(self.projects_root.root))
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

    def _new_project(self) -> None:
        name = _ask_string(self.root, "New project", "Project name (folder-friendly):")
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
        # Generate tab defaults also come from the project so they persist per LoRA.
        if not self.negative_var.get().strip():
            self.negative_var.set(project.default_negative_prompt)
        # Invalidate the Review tab state so it rebuilds for this project.
        self._review = None

    def _save_settings(self) -> None:
        if not self.current_project:
            messagebox.showerror("No project", "Create or open a project first.")
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
        self.status_var.set(f"settings saved to {p.config_path.name}")

    # ---- review tab ----

    def _focus_is_text_input(self) -> bool:
        """True when the user is typing in the caption or notes widget; we
        must not let Review shortcuts (←/→/I) steal those keystrokes."""
        try:
            w = self.root.focus_get()
        except Exception:
            return False
        return isinstance(w, (tk.Text, tk.Entry, ttk.Entry, ttk.Combobox))

    def _kbd(self, action):
        def handler(_event=None):
            if self._focus_is_text_input():
                return  # let the widget handle the key
            action()
        return handler

    def _on_tab_changed(self, _event) -> None:
        selected = self.nb.select()
        if selected == str(self.tab_review) and self.current_project is not None:
            # Load once when switching in; preserve edits if already loaded.
            if self._review is None:
                self._reload_review()
            else:
                self._refresh_review_list_ui()

        # Bind nav + toggle only while the Review tab is active, and guard
        # against firing when focus is inside the caption / notes widgets.
        if selected == str(self.tab_review):
            self.root.bind("<Left>", self._kbd(lambda: self._review_step(-1)))
            self.root.bind("<Right>", self._kbd(lambda: self._review_step(+1)))
            self.root.bind("<i>", self._kbd(self._toggle_review_include))
            self.root.bind("<I>", self._kbd(self._toggle_review_include))
        else:
            for key in ("<Left>", "<Right>", "<i>", "<I>"):
                try:
                    self.root.unbind(key)
                except Exception:
                    pass

    def _reload_review(self) -> None:
        if not self.current_project:
            messagebox.showerror("No project", "Create or open a project first.")
            return
        project = self.current_project
        self._review = review_mod.load(project)
        self._review_order = sorted(self._review.entries.keys())
        if not self._review_order:
            self.review_counts_var.set("no processed images yet")
            self.review_image_label.configure(image="", text="(run prep first)")
            self._review_idx = -1
            self.review_list.delete(0, "end")
            return

        # Pre-compute per-image stats + hashes in one pass (one file-open per
        # image), then derive near-duplicate pairs from the cached hashes.
        pngs = [project.processed_dir / f"{s}.png" for s in self._review_order]
        self._review_stats_index = {}
        hashes: list[tuple[str, int]] = []
        for p in pngs:
            st, h = insights.stats_and_hash(p)
            self._review_stats_index[p.stem] = st
            hashes.append((p.stem, h))
        self._review_dupes_index = {}
        for i in range(len(hashes)):
            stem_i, h_i = hashes[i]
            for j in range(i + 1, len(hashes)):
                stem_j, h_j = hashes[j]
                d = insights.hamming(h_i, h_j)
                if d <= 6:
                    self._review_dupes_index.setdefault(stem_i, []).append((stem_j, d))
                    self._review_dupes_index.setdefault(stem_j, []).append((stem_i, d))

        self._refresh_review_list_ui()
        self._review_idx = 0
        self._render_review_entry()

    def _refresh_review_list_ui(self) -> None:
        self.review_list.delete(0, "end")
        for stem in self._review_order:
            entry = self._review.entries[stem]
            mark = "✓" if entry.include else "✗"
            self.review_list.insert("end", f"{mark} {stem}")
        counts = f"{self._review.included_count()} in / {self._review.excluded_count()} out of {len(self._review_order)}"
        self.review_counts_var.set(counts)
        self._rebuild_chips(self.current_project.prompt_chips if self.current_project else [])

    def _rebuild_chips(self, chips: list) -> None:
        """Redraw the quick-tag chips bar from `chips` (a list of strings).

        The default chip palette in :attr:`Project.prompt_chips` includes
        framing, lighting, pose, clothing (SFW + NSFW), and camera tags —
        edit that list in the project's ``config.json`` to customize.
        """
        for w in self.review_chips_frame.winfo_children():
            w.destroy()
        if not chips:
            ttk.Label(self.review_chips_frame, text="(no chips configured)").pack(anchor="w")
            return
        # Wrap chips into rows of ~10 so a large palette (60+) stays compact.
        per_row = 10
        row = None
        for i, chip in enumerate(chips):
            if i % per_row == 0:
                row = ttk.Frame(self.review_chips_frame)
                row.pack(fill="x")
            ttk.Button(row, text=chip, command=lambda c=chip: self._append_chip(c)).pack(
                side="left", padx=(0, 4), pady=2
            )

    def _on_review_list_select(self, _event) -> None:
        sel = self.review_list.curselection()
        if not sel:
            return
        new_idx = sel[0]
        if new_idx == self._review_idx:
            return
        self._capture_current_editor()
        self._review_idx = new_idx
        self._render_review_entry()

    def _review_step(self, delta: int) -> None:
        if self._review is None or not self._review_order:
            return
        self._capture_current_editor()
        self._review_idx = max(0, min(len(self._review_order) - 1, self._review_idx + delta))
        self.review_list.selection_clear(0, "end")
        self.review_list.selection_set(self._review_idx)
        self.review_list.see(self._review_idx)
        self._render_review_entry()

    def _current_stem(self) -> Optional[str]:
        if 0 <= self._review_idx < len(self._review_order):
            return self._review_order[self._review_idx]
        return None

    def _render_review_entry(self) -> None:
        stem = self._current_stem()
        if not stem or self.current_project is None or self._review is None:
            return
        entry = self._review.entries[stem]

        # image preview
        png_path = self.current_project.processed_dir / f"{stem}.png"
        self._show_preview(png_path)

        # caption / include / notes
        self.review_include_var.set(entry.include)
        self.review_caption_text.delete("1.0", "end")
        self.review_caption_text.insert("1.0", entry.caption)
        self.review_notes_entry.delete(0, "end")
        self.review_notes_entry.insert(0, entry.notes)

        # info
        stats = self._review_stats_index.get(stem, {})
        info_bits = [f"{stem}.png"]
        if stats:
            info_bits.append(f"{stats['width']}×{stats['height']}")
            info_bits.append(f"brightness {stats['brightness']}")
            info_bits.append(f"sharpness {stats['sharpness']}")
        if stats:
            warn = insights.resolution_warning(stats["width"], stats["height"], self.current_project.resolution)
            if warn:
                info_bits.append(f"⚠ {warn}")
        self.review_info_var.set("   |   ".join(info_bits))

        dupes = self._review_dupes_index.get(stem, [])
        if dupes:
            text = "near-duplicates: " + ", ".join(f"{s} (d={d})" for s, d in dupes[:5])
            self.review_dupes_var.set(text)
        else:
            self.review_dupes_var.set("")

    def _show_preview(self, png_path: Path, max_side: int = 520) -> None:
        from PIL import Image, ImageTk

        try:
            img = Image.open(png_path).convert("RGB")
        except Exception as e:
            self.review_image_label.configure(image="", text=f"(failed to open: {e})")
            return
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._review_photo_cache.append(photo)
        # Keep the last few; lets us navigate fast without Tk GC-ing a
        # still-rendering image.
        if len(self._review_photo_cache) > 4:
            self._review_photo_cache = self._review_photo_cache[-4:]
        self.review_image_label.configure(image=photo, text="")

    def _capture_current_editor(self) -> None:
        """Flush the caption / include / notes widgets back into the in-memory
        Review object. Called before navigation and before save."""
        stem = self._current_stem()
        if not stem or self._review is None:
            return
        entry = self._review.entries[stem]
        entry.include = bool(self.review_include_var.get())
        entry.caption = self.review_caption_text.get("1.0", "end").strip()
        entry.notes = self.review_notes_entry.get().strip()
        # reflect include/exclude tick in the list row
        mark = "✓" if entry.include else "✗"
        self.review_list.delete(self._review_idx)
        self.review_list.insert(self._review_idx, f"{mark} {stem}")
        self.review_list.selection_clear(0, "end")
        self.review_list.selection_set(self._review_idx)
        counts = f"{self._review.included_count()} in / {self._review.excluded_count()} out of {len(self._review_order)}"
        self.review_counts_var.set(counts)

    def _on_review_include_toggle(self) -> None:
        self._capture_current_editor()

    def _toggle_review_include(self) -> None:
        self.review_include_var.set(not self.review_include_var.get())
        self._capture_current_editor()

    def _append_chip(self, chip: str) -> None:
        current = self.review_caption_text.get("1.0", "end").strip()
        new = review_mod.append_chip(current, chip)
        self.review_caption_text.delete("1.0", "end")
        self.review_caption_text.insert("1.0", new)
        self._capture_current_editor()

    def _save_review(self) -> None:
        if not self.current_project or self._review is None:
            messagebox.showerror("Nothing to save", "Reload the Review tab first.")
            return
        self._capture_current_editor()
        path = review_mod.save(self.current_project, self._review)
        self.status_var.set(f"review saved to {path.name}")
        self.log_queue.put(f"[review saved: {path}]\n")

    # ---- pickers / open folder ----

    def _pick_source_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose source folder of raw images")
        if path:
            self.source_dir_var.set(path)

    def _pick_base_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose base SDXL checkpoint",
            filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")],
        )
        if path:
            self.base_model_var.set(path)

    def _open_outputs_dir(self) -> None:
        if not self.current_project:
            return
        _open_folder(self.current_project.outputs_dir)

    def _open_validation_dir(self) -> None:
        if not self.current_project:
            return
        _open_folder(self.current_project.validation_dir)

    def _open_logs_dir(self) -> None:
        if not self.current_project:
            return
        _open_folder(self.current_project.logs_dir)

    def _open_latest_log(self) -> None:
        """Open the most recent ``training_<ts>.log`` in the OS default viewer.

        Handy when the GUI's live Log pane has scrolled past something you
        want to search, or when training has been running for hours.
        """
        if not self.current_project:
            return
        logs = list(self.current_project.logs_dir.glob("training_*.log"))
        if not logs:
            messagebox.showinfo("No logs yet", "No training_*.log files have been written yet.")
            return
        # Sort by mtime rather than name — filenames are ISO-timestamped
        # today, but mtime is robust to future format changes.
        logs.sort(key=lambda p: p.stat().st_mtime)
        _open_file(logs[-1])

    def _open_journal(self) -> None:
        """Open ``logs/journal.txt`` — one line per training run."""
        if not self.current_project:
            return
        journal = self.current_project.logs_dir / "journal.txt"
        if not journal.exists():
            journal.write_text("")
        _open_file(journal)

    # ---- command handlers ----

    def _require_project(self) -> Optional[Project]:
        if not self.current_project:
            messagebox.showerror("No project", "Create or open a project first.")
            return None
        return self.current_project

    def _spawn(self, args: list[str]) -> None:
        try:
            self.runner.start(args)
            self.status_var.set(f"running: {args[0]}")
        except RuntimeError:
            messagebox.showwarning("Busy", "Another step is still running.")

    def _on_prep(self) -> None:
        project = self._require_project()
        if not project:
            return
        self._save_settings()
        args = ["prep", str(project.root)]
        src = self.source_dir_var.get().strip()
        if src:
            args += ["--source", src]
        self._spawn(args)

    def _on_caption(self) -> None:
        project = self._require_project()
        if not project:
            return
        self._save_settings()
        self._spawn(["caption", str(project.root)])

    def _train_args(self, *, resume: bool) -> Optional[list[str]]:
        project = self._require_project()
        if not project:
            return None
        self._save_settings()
        args = ["train", str(project.root), "--max-steps", self.max_steps_var.get()]
        if resume:
            args.append("--resume")
        # Ensure base model is set on the project; the CLI blocks `--base` on
        # resume so we only forward it when not resuming.
        if not resume:
            base = self.base_model_var.get().strip()
            if base:
                args += ["--base", base]
        note = self.train_note_var.get().strip()
        if note:
            args += ["--note", note]
        return args

    def _on_train(self) -> None:
        args = self._train_args(resume=False)
        if args is None:
            return
        self.train_progress["value"] = 0
        self.train_pct_var.set("0%")
        self.train_status_var.set("starting...")
        self._spawn(args)

    def _on_train_resume(self) -> None:
        args = self._train_args(resume=True)
        if args is None:
            return
        self.train_status_var.set("resuming...")
        self._spawn(args)

    def _on_train_stop(self) -> None:
        if not self.runner.is_running():
            messagebox.showinfo("Nothing running", "No training process to stop.")
            return
        sent = self.runner.stop_graceful()
        if sent:
            self.train_status_var.set("stop requested; checkpointing before exit...")
            self.status_var.set("sent SIGINT to training subprocess")
            # Push a line into the log pane too. The subprocess's own
            # "Caught signal..." print will arrive when the training loop
            # notices the flag, which can take 10-30 seconds if a step is
            # mid-way; this gives the user immediate visible feedback.
            self.runner.log_queue.put(
                "[stop requested; waiting for training to checkpoint and exit "
                "— this can take up to ~30s while the current step finishes]\n"
            )
        else:
            messagebox.showerror(
                "Stop failed",
                "Couldn't send signal. Use Ctrl+C in the terminal that launched the GUI.",
            )

    def _on_generate(self) -> None:
        project = self._require_project()
        if not project:
            return
        self._save_settings()
        args = [
            "generate",
            str(project.root),
            "--prompt", self.prompt_var.get(),
            "--n", self.n_var.get() or "4",
            "--steps", self.steps_var.get() or "30",
            "--guidance", self.guidance_var.get() or "7.0",
        ]
        neg = self.negative_var.get().strip()
        if neg:
            args += ["--negative", neg]
        seed = self.seed_var.get().strip()
        if seed:
            args += ["--seed", seed]
        self._spawn(args)

    # ---- log pump ----

    def _drain_log(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.log.configure(state="normal")
                self.log.insert("end", line)
                self.log.see("end")
                self.log.configure(state="disabled")
                self._maybe_update_progress(line)
        except queue.Empty:
            pass
        if not self.runner.is_running() and self.status_var.get().startswith("running:"):
            self.status_var.set("ready")
        self.root.after(100, self._drain_log)

    def _maybe_update_progress(self, line: str) -> None:
        # Caching phase: `caching 3/20: 0002.png`
        if line.startswith("caching "):
            try:
                nm = line.split(" ", 2)[1]
                n, m = nm.split("/")
                n, m = int(n), int(m)
                self.train_progress["maximum"] = m
                self.train_progress["value"] = n
                self.train_pct_var.set(f"{int(100 * n / max(m, 1))}% (cache)")
                self.train_status_var.set(line.strip())
            except Exception:
                pass
            return
        # Training phase: `step 50/1500 loss=0.02 ...`
        if line.startswith("step ") and "/" in line:
            try:
                nm = line.split(" ", 2)[1]
                n, m = nm.split("/")
                n, m = int(n), int(m)
                self.train_progress["maximum"] = m
                self.train_progress["value"] = n
                self.train_pct_var.set(f"{int(100 * n / max(m, 1))}%")
                self.train_status_var.set(line.strip())
            except Exception:
                pass


# ---------- small utilities ----------

def _ask_string(parent: tk.Tk, title: str, prompt: str) -> str:
    from tkinter import simpledialog
    return simpledialog.askstring(title, prompt, parent=parent) or ""


def _platform_open(path: Path) -> None:
    """Best-effort 'open in default app' across OSes. Falls back to a
    message box when no opener is available (e.g. a minimal Linux box with
    no ``xdg-open``) so the Tk callback never crashes with an uncaught
    ``FileNotFoundError``."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(path)])
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")
    except FileNotFoundError:
        messagebox.showinfo(
            "Open failed",
            f"No system opener available. Path:\n{path}",
        )
    except Exception as e:
        messagebox.showerror("Open failed", f"{e}\nPath: {path}")


def _open_folder(path: Path) -> None:
    """Reveal `path` in the OS file manager. Creates the folder if missing."""
    path.mkdir(parents=True, exist_ok=True)
    _platform_open(path)


def _open_file(path: Path) -> None:
    """Open a file in the OS default viewer (e.g. text editor for .log/.txt)."""
    _platform_open(path)


_TAB_INDEX = {
    "settings": 0, "prep": 1, "caption": 2, "review": 3, "train": 4, "generate": 5,
}


def launch(
    initial_project_dir: Optional[Path] = None,
    initial_tab: Optional[str] = None,
) -> None:
    """Create the Tk root and run the main loop.

    Args:
        initial_project_dir: If set, the GUI opens this project on startup.
            If the project lives outside the default :class:`ProjectsRoot`,
            the GUI re-points its browser to the project's parent so the
            combobox reflects what's open. This is how
            ``trainer review <project>`` jumps straight to the right
            project.
        initial_tab: Optional key from :data:`_TAB_INDEX`
            (``"settings"``, ``"prep"``, ``"caption"``, ``"review"``,
            ``"train"``, ``"generate"``). The notebook is switched to that
            tab after the window is fully laid out.
    """
    root = tk.Tk()
    gui = TrainerGUI(root)

    if initial_project_dir is not None:
        try:
            project = Project.load(initial_project_dir)
        except Exception as e:
            gui.log_queue.put(f"[Could not open {initial_project_dir}: {e}]\n")
        else:
            # If the project is outside the default ProjectsRoot, re-point the
            # browser at its parent so the combobox shows the right thing
            # instead of a stale list from the default root.
            parent = project.root.parent
            if parent != gui.projects_root.root:
                gui.projects_root = ProjectsRoot(parent)
                gui.projects_root_var.set(str(parent))
            gui.current_project = project
            gui._load_settings_into_ui(project)
            gui._refresh_project_list()
            gui.project_combo.set(project.root.name)
            gui.log_queue.put(f"[Opened project via CLI: {project.root}]\n")

    if initial_tab and initial_tab in _TAB_INDEX:
        root.after(300, lambda: gui.nb.select(_TAB_INDEX[initial_tab]))

    root.mainloop()


if __name__ == "__main__":
    launch()
