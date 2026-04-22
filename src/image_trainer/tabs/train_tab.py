"""05 · TRAIN tab — run/resume/stop training with live ETA + loss sparkline.

Parses ``step N/M loss=...`` lines from the CLI subprocess to keep the
progress bar, ETA, step counter, and loss sparkline in sync. Polls
``nvidia-smi`` every ~2s for a VRAM readout when available.
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Optional

from .. import gui_helpers, gui_theme
from ..gui_widgets import Sparkline

if TYPE_CHECKING:
    from ..gui_app import TrainerGUI


def build(gui: "TrainerGUI") -> None:
    state = _TrainState(gui)
    gui.train_state = state
    state.build_ui(gui.tab_train)


class _TrainState:
    def __init__(self, gui: "TrainerGUI") -> None:
        self.gui = gui
        self.run_started_at: Optional[float] = None
        self.last_step: int = 0
        self.total_steps: int = 0
        self.last_vram_poll: float = 0.0

    def build_ui(self, root: ttk.Frame) -> None:
        PAD = gui_theme.PAD
        root.columnconfigure(0, weight=1)

        ttk.Label(root, text="Train · LoRA", style="Header.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, PAD)
        )
        ttk.Label(
            root,
            text=(
                "Uses settings from the Settings tab. Ctrl+C in the terminal "
                "OR the Stop button save a checkpoint before exiting — safe "
                "to resume."
            ),
            justify="left", style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=4, sticky="w")

        # Journal note row
        note_row = ttk.Frame(root)
        note_row.grid(row=2, column=0, columnspan=4, sticky="we", pady=(PAD, 0))
        ttk.Label(note_row, text="Journal note:").pack(side="left")
        self.note_var = tk.StringVar()
        self.gui.train_note_var = self.note_var
        ttk.Entry(note_row, textvariable=self.note_var).pack(
            side="left", fill="x", expand=True, padx=(PAD, 0)
        )

        # Progress + metrics row
        metrics = ttk.Frame(root)
        metrics.grid(row=3, column=0, columnspan=4, sticky="we", pady=(PAD * 2, PAD // 2))
        metrics.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(
            metrics, mode="determinate",
            style="Trainer.Horizontal.TProgressbar",
        )
        self.progress.grid(row=0, column=0, sticky="we")
        self.pct_var = tk.StringVar(value="0%")
        ttk.Label(metrics, textvariable=self.pct_var, width=7, anchor="e",
                  style="MonoHot.TLabel").grid(row=0, column=1, padx=(PAD, 0))

        # Second row: step counter | ETA | elapsed | VRAM
        row2 = ttk.Frame(root)
        row2.grid(row=4, column=0, columnspan=4, sticky="we", pady=(PAD // 2, 0))
        self.step_var = tk.StringVar(value="step —/—")
        self.eta_var = tk.StringVar(value="ETA --:--:--")
        self.elapsed_var = tk.StringVar(value="elapsed 00:00:00")
        self.vram_var = tk.StringVar(value="VRAM —/— MiB")
        ttk.Label(row2, textvariable=self.step_var, style="Mono.TLabel").pack(side="left")
        ttk.Label(row2, text="  ·  ", style="Status.TLabel").pack(side="left")
        ttk.Label(row2, textvariable=self.eta_var, style="Mono.TLabel").pack(side="left")
        ttk.Label(row2, text="  ·  ", style="Status.TLabel").pack(side="left")
        ttk.Label(row2, textvariable=self.elapsed_var, style="Mono.TLabel").pack(side="left")
        ttk.Label(row2, text="  ·  ", style="Status.TLabel").pack(side="left")
        ttk.Label(row2, textvariable=self.vram_var, style="Mono.TLabel").pack(side="left")

        # Third row: sparkline + status line
        row3 = ttk.Frame(root)
        row3.grid(row=5, column=0, columnspan=4, sticky="we", pady=(PAD, 0))
        row3.columnconfigure(1, weight=1)
        ttk.Label(row3, text="loss", style="Status.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, PAD)
        )
        self.sparkline = Sparkline(row3, width=420, height=54, maxlen=400)
        self.sparkline.grid(row=0, column=1, sticky="we")

        self.status_var = tk.StringVar(value="idle")
        self.gui.train_status_var = self.status_var
        ttk.Label(root, textvariable=self.status_var, style="Status.TLabel").grid(
            row=6, column=0, columnspan=4, sticky="w", pady=(PAD, 0)
        )

        # Action buttons
        btns = ttk.Frame(root)
        btns.grid(row=7, column=0, columnspan=4, sticky="w", pady=PAD)
        ttk.Button(btns, text="START TRAINING", style="Primary.TButton",
                   command=self.on_start).pack(side="left")
        ttk.Button(btns, text="RESUME TRAINING",
                   command=self.on_resume).pack(side="left", padx=PAD)
        ttk.Button(btns, text="STOP (GRACEFUL)", style="Caution.TButton",
                   command=self.on_stop).pack(side="left")

        # Log / journal / validation shortcuts
        log_btns = ttk.Frame(root)
        log_btns.grid(row=8, column=0, columnspan=4, sticky="w", pady=PAD)
        ttk.Button(log_btns, text="OPEN LOGS", style="Ghost.TButton",
                   command=self._open_logs).pack(side="left")
        ttk.Button(log_btns, text="OPEN VALIDATION", style="Ghost.TButton",
                   command=self._open_validation).pack(side="left", padx=PAD)
        ttk.Button(log_btns, text="OPEN LATEST LOG", style="Ghost.TButton",
                   command=self._open_latest_log).pack(side="left")
        ttk.Button(log_btns, text="OPEN JOURNAL", style="Ghost.TButton",
                   command=self._open_journal).pack(side="left", padx=(PAD, 0))

        # Make progress bar handles reachable from the GUI for legacy code paths.
        self.gui.train_progress = self.progress
        self.gui.train_pct_var = self.pct_var

    # ---- handlers ----

    def _train_args(self, *, resume: bool) -> Optional[list[str]]:
        project = self.gui.require_project()
        if not project:
            return None
        self.gui.save_settings_silent()
        args = ["train", str(project.root), "--max-steps", self.gui.max_steps_var.get()]
        if resume:
            args.append("--resume")
        if not resume:
            base = self.gui.base_model_var.get().strip()
            if base:
                args += ["--base", base]
        note = self.note_var.get().strip()
        if note:
            args += ["--note", note]
        return args

    def on_start(self) -> None:
        args = self._train_args(resume=False)
        if args is None:
            return
        self._reset_metrics()
        self.status_var.set("starting...")
        self.gui.spawn(args)

    def on_resume(self) -> None:
        args = self._train_args(resume=True)
        if args is None:
            return
        self._reset_metrics()
        self.status_var.set("resuming...")
        self.gui.spawn(args)

    def on_stop(self) -> None:
        if not self.gui.runner.is_running():
            messagebox.showinfo("Nothing running", "No training process to stop.")
            return
        if self.gui.runner.stop_graceful():
            self.status_var.set("stop requested; checkpointing before exit...")
            self.gui.log_queue.put(
                "[stop requested; waiting for training to checkpoint and exit "
                "— this can take up to ~30s while the current step finishes]\n"
            )
        else:
            messagebox.showerror(
                "Stop failed",
                "Couldn't send signal. Use Ctrl+C in the terminal that launched the GUI.",
            )

    # ---- live update hooks (called from TrainerGUI._drain_log) ----

    def on_progress_line(self, line: str) -> None:
        parsed = gui_helpers.parse_step_line(line)
        if parsed is None:
            return
        step = parsed["step"]
        total = parsed["total"]
        self.progress["maximum"] = total
        self.progress["value"] = step
        pct = int(100 * step / max(total, 1))
        phase = parsed.get("phase")
        suffix = " (cache)" if phase == "cache" else ""
        self.pct_var.set(f"{pct}%{suffix}")
        self.step_var.set(f"step {step}/{total}")
        self.status_var.set(line.strip())

        if phase == "train":
            if self.run_started_at is None or step < self.last_step:
                self.run_started_at = time.monotonic()
                self.sparkline.clear()
            self.last_step = step
            self.total_steps = total
            loss = parsed.get("loss")
            if isinstance(loss, (int, float)):
                self.sparkline.push(float(loss))
            elapsed = time.monotonic() - self.run_started_at
            self.elapsed_var.set(f"elapsed {gui_helpers.format_elapsed(elapsed)}")
            self.eta_var.set(gui_helpers.format_eta(elapsed, step, total))

    def tick(self) -> None:
        """Called by the main log-pump timer ~every 100ms."""
        # VRAM poll at most every 2s to avoid firing nvidia-smi every tick.
        now = time.monotonic()
        if now - self.last_vram_poll > 2.0:
            self.last_vram_poll = now
            v = gui_helpers.probe_vram()
            if v is not None:
                used, total = v
                self.vram_var.set(f"VRAM {used:>5}/{total} MiB")
            else:
                self.vram_var.set("VRAM —/— MiB")
        # Keep elapsed ticking while training is running.
        if self.run_started_at is not None and self.gui.runner.is_running():
            elapsed = time.monotonic() - self.run_started_at
            self.elapsed_var.set(f"elapsed {gui_helpers.format_elapsed(elapsed)}")
            if self.last_step > 0 and self.total_steps > 0:
                self.eta_var.set(
                    gui_helpers.format_eta(elapsed, self.last_step, self.total_steps)
                )

    def _reset_metrics(self) -> None:
        self.progress["value"] = 0
        self.pct_var.set("0%")
        self.step_var.set("step —/—")
        self.eta_var.set("ETA --:--:--")
        self.elapsed_var.set("elapsed 00:00:00")
        self.sparkline.clear()
        self.run_started_at = None
        self.last_step = 0
        self.total_steps = 0

    # ---- shortcuts ----

    def _open_logs(self) -> None:
        if not self.gui.current_project:
            return
        gui_helpers.open_folder(self.gui.current_project.logs_dir)

    def _open_validation(self) -> None:
        if not self.gui.current_project:
            return
        gui_helpers.open_folder(self.gui.current_project.validation_dir)

    def _open_latest_log(self) -> None:
        if not self.gui.current_project:
            return
        logs = list(self.gui.current_project.logs_dir.glob("training_*.log"))
        if not logs:
            messagebox.showinfo("No logs yet", "No training_*.log files have been written yet.")
            return
        logs.sort(key=lambda p: p.stat().st_mtime)
        gui_helpers.open_file(logs[-1])

    def _open_journal(self) -> None:
        if not self.gui.current_project:
            return
        journal = self.gui.current_project.logs_dir / "journal.txt"
        if not journal.exists():
            journal.write_text("")
        gui_helpers.open_file(journal)
