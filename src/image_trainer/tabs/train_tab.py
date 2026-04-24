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
from ..gui_widgets import Sparkline, info_icon

if TYPE_CHECKING:
    from ..gui_app import TrainerGUI


def build(gui: "TrainerGUI") -> None:
    state = _TrainState(gui)
    gui.train_state = state
    state.build_ui(gui.tab_train)


class _TrainState:
    def __init__(self, gui: "TrainerGUI") -> None:
        self.gui = gui
        # Wall-clock start of the current subprocess run. Set the moment we
        # fire the CLI so "elapsed" ticks during the (sometimes-long) setup
        # and caching phases — before the first ``step N/M`` line arrives.
        self.run_started_at: Optional[float] = None
        # Timestamp of the first real training-phase step. ETA is computed
        # from here so it doesn't include caching / model-load warmup time.
        self.train_started_at: Optional[float] = None
        self.train_start_step: int = 0
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
        info_icon(
            note_row,
            "Optional one-line note appended to logs/journal.txt for this run, "
            "alongside the timestamp + key settings (rank, resolution, LR, "
            "step count). Use it like a lab notebook so you can later remember "
            "what you tried — e.g. 'rank 32, pony base, more close-ups' or "
            "'first run after re-captioning'. Blank is fine.",
        ).pack(side="left")
        self.note_var = tk.StringVar()
        self.gui.train_note_var = self.note_var
        self.note_entry = ttk.Entry(note_row, textvariable=self.note_var)
        self.note_entry.pack(side="left", fill="x", expand=True, padx=(PAD, 0))
        # Subtle placeholder hint via the entry's secondary text — Tkinter
        # doesn't have native placeholders, so we fake it: prefill grey text
        # that clears on first focus.
        self._install_placeholder(
            self.note_entry, self.note_var,
            "e.g. rank 32, pony base, more close-ups",
        )

        # Scope / time-budget row — image-limit picker + Free VRAM button
        # live above the metrics so they're visible *before* you commit to a
        # run. Image limit answers "I have 15 minutes" by shrinking the
        # per-epoch dataset; Free VRAM kills any other GPU-using process so
        # the run doesn't OOM on init.
        scope_row = ttk.Frame(root)
        scope_row.grid(row=3, column=0, columnspan=4, sticky="we", pady=(PAD // 2, 0))
        ttk.Label(scope_row, text="Images per run:").pack(side="left")
        info_icon(
            scope_row,
            "Cap how many included images the run trains on. 'All' uses every "
            "image; pick a number when you only have a short window of PC "
            "time. A 1-image run finishes in minutes for a quick sanity test, "
            "but won't generalise. Cache stays valid for the next 'All' run.",
        ).pack(side="left")
        self.image_limit_var = tk.StringVar(value="All")
        self.image_limit_combo = ttk.Combobox(
            scope_row,
            textvariable=self.image_limit_var,
            values=["All", "1", "5", "10", "25", "50", "100"],
            width=8,
        )
        self.image_limit_combo.pack(side="left", padx=(PAD // 2, 0))
        ttk.Label(
            scope_row,
            text="(All = train on every included image; pick a number for a faster run)",
            style="Status.TLabel",
        ).pack(side="left", padx=(PAD, 0))
        free_btn = ttk.Button(
            scope_row, text="Free VRAM", style="Ghost.TButton",
            command=self.on_free_vram,
        )
        free_btn.pack(side="right")
        info_icon(
            scope_row,
            "Lists every process holding GPU memory (via nvidia-smi), excludes "
            "this app + the training subprocess if running, and kills the rest "
            "after you confirm. Use right before Start training to claw back "
            "VRAM from browsers, other ML jobs, etc. SIGTERM with a 2-second "
            "escalation to SIGKILL.",
        ).pack(side="right", padx=(0, PAD // 2))

        # Progress + metrics row
        metrics = ttk.Frame(root)
        metrics.grid(row=4, column=0, columnspan=4, sticky="we", pady=(PAD * 2, PAD // 2))
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
        row2.grid(row=5, column=0, columnspan=4, sticky="we", pady=(PAD // 2, 0))
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
        row3.grid(row=6, column=0, columnspan=4, sticky="we", pady=(PAD, 0))
        row3.columnconfigure(1, weight=1)
        ttk.Label(row3, text="loss", style="Status.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, PAD)
        )
        self.sparkline = Sparkline(row3, width=420, height=54, maxlen=400)
        self.sparkline.grid(row=0, column=1, sticky="we")

        self.status_var = tk.StringVar(value="idle")
        self.gui.train_status_var = self.status_var
        ttk.Label(root, textvariable=self.status_var, style="Status.TLabel").grid(
            row=7, column=0, columnspan=4, sticky="w", pady=(PAD, 0)
        )

        # Action buttons
        btns = ttk.Frame(root)
        btns.grid(row=8, column=0, columnspan=4, sticky="w", pady=PAD)
        ttk.Button(btns, text="Start training", style="Primary.TButton",
                   command=self.on_start).pack(side="left")
        ttk.Button(btns, text="Resume training",
                   command=self.on_resume).pack(side="left", padx=PAD)
        ttk.Button(btns, text="Stop (graceful)", style="Caution.TButton",
                   command=self.on_stop).pack(side="left")

        # Log / journal / validation shortcuts
        log_btns = ttk.Frame(root)
        log_btns.grid(row=9, column=0, columnspan=4, sticky="w", pady=PAD)
        ttk.Button(log_btns, text="Open logs", style="Ghost.TButton",
                   command=self._open_logs).pack(side="left")
        ttk.Button(log_btns, text="Open validation", style="Ghost.TButton",
                   command=self._open_validation).pack(side="left", padx=PAD)
        ttk.Button(log_btns, text="Open latest log", style="Ghost.TButton",
                   command=self._open_latest_log).pack(side="left")
        ttk.Button(log_btns, text="Open journal", style="Ghost.TButton",
                   command=self._open_journal).pack(side="left", padx=(PAD, 0))

        # Make progress bar handles reachable from the GUI for legacy code paths.
        self.gui.train_progress = self.progress
        self.gui.train_pct_var = self.pct_var

    # ---- helpers ----

    def _install_placeholder(
        self, entry: ttk.Entry, var: tk.StringVar, placeholder: str,
    ) -> None:
        """Fake a placeholder on a ttk.Entry — Tkinter doesn't ship one.

        Drops grey hint text into the entry while empty + unfocused. Clears
        on focus-in, restores on focus-out if the user typed nothing. The
        ``_is_placeholder`` attribute marks the state so handlers (like the
        spawn args builder) can ignore the hint as if the field were empty.
        """
        muted = gui_theme.THEME.TEXT_MUTED
        primary = gui_theme.THEME.TEXT_PRIMARY

        def show_placeholder() -> None:
            var.set(placeholder)
            entry.configure(foreground=muted)
            entry._is_placeholder = True  # type: ignore[attr-defined]

        def clear_if_placeholder(_e=None) -> None:
            if getattr(entry, "_is_placeholder", False):
                var.set("")
                entry.configure(foreground=primary)
                entry._is_placeholder = False  # type: ignore[attr-defined]

        def restore_if_empty(_e=None) -> None:
            if not var.get().strip():
                show_placeholder()

        entry.bind("<FocusIn>", clear_if_placeholder, add="+")
        entry.bind("<FocusOut>", restore_if_empty, add="+")
        # Initial state.
        if not var.get().strip():
            show_placeholder()

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
        # Image-limit picker. "All" (or empty / unparseable) means no limit.
        # An integer is forwarded as ``--limit-images N`` and the training
        # loop slices the included-stem list to that many before caching.
        limit_raw = self.image_limit_var.get().strip()
        if limit_raw and limit_raw.lower() != "all":
            try:
                n = int(limit_raw)
                if n > 0:
                    args += ["--limit-images", str(n)]
            except ValueError:
                # Silently ignore — user can recover by picking from the dropdown.
                self.gui.log_queue.put(
                    f"[ignoring non-integer image limit: {limit_raw!r}]\n"
                )
        # Skip the placeholder hint text — _install_placeholder marks the
        # entry with _is_placeholder while the hint is showing.
        if getattr(self.note_entry, "_is_placeholder", False):
            note = ""
        else:
            note = self.note_var.get().strip()
        if note:
            args += ["--note", note]
        return args

    # ---- Free VRAM (kill other GPU processes) ----

    def on_free_vram(self) -> None:
        """List the processes currently holding GPU memory, exclude this GUI
        and the active training subprocess if any, and prompt to kill the
        rest. Targeted: nothing happens to non-GPU apps."""
        import os

        procs = gui_helpers.list_gpu_processes()
        my_pid = os.getpid()
        # Exclude us + the trainer subprocess (if running) so we don't kill
        # ourselves mid-train. The runner stores the live Popen on
        # ``self.gui.runner.proc``; if it's not running, ``proc`` may be a
        # stale handle from the last invocation — guard with is_running().
        excluded = {my_pid}
        if self.gui.runner.is_running() and self.gui.runner.proc is not None:
            excluded.add(self.gui.runner.proc.pid)
        targets = [p for p in procs if p.pid not in excluded]

        if not procs:
            messagebox.showinfo(
                "No GPU processes",
                "nvidia-smi reported no processes holding VRAM (or isn't installed).",
            )
            return
        if not targets:
            messagebox.showinfo(
                "Nothing to free",
                "The only GPU processes are this GUI and the active training run "
                "— there's nothing safe to kill.",
            )
            return

        body_lines = [f"  pid {p.pid:>7}  ·  {p.used_mib:>5} MiB  ·  {p.name}"
                      for p in targets]
        total_mib = sum(p.used_mib for p in targets)
        body = "\n".join(body_lines)
        msg = (
            f"Send SIGTERM to {len(targets)} process(es) currently holding "
            f"~{total_mib} MiB of VRAM?\n\n"
            f"{body}\n\n"
            "Anything that doesn't exit within 2 seconds will get SIGKILL. "
            "You will lose unsaved work in these processes."
        )
        if not messagebox.askyesno("Free VRAM", msg):
            return

        results = gui_helpers.kill_processes([p.pid for p in targets])
        terminated = sum(1 for s in results.values() if s in ("terminated", "killed", "gone"))
        denied = [pid for pid, s in results.items() if s == "perm-denied"]
        failed = [pid for pid, s in results.items() if s.startswith("failed")]

        # Quick after-shot of free VRAM for the log so the user sees the win.
        v = gui_helpers.probe_vram()
        post = f" — VRAM now {v[0]}/{v[1]} MiB" if v else ""
        self.gui.log_queue.put(
            f"[Free VRAM: {terminated}/{len(targets)} cleared{post}]\n"
        )
        for pid, status in results.items():
            self.gui.log_queue.put(f"[  pid {pid}: {status}]\n")
        self.status_var.set(
            f"Freed {terminated}/{len(targets)} GPU process(es)"
            + (f"; {len(denied)} permission-denied" if denied else "")
            + (f"; {len(failed)} failed" if failed else "")
        )

    def on_start(self) -> None:
        if not self._preflight_te_lora_or_offer_fix():
            return
        args = self._train_args(resume=False)
        if args is None:
            return
        self._reset_metrics()
        # Start the elapsed clock the instant we dispatch — long cache /
        # model-load warmups are real wall time the user wants to see.
        self.run_started_at = time.monotonic()
        self.status_var.set("starting...")
        self.gui.spawn(args)

    def on_resume(self) -> None:
        if not self._preflight_te_lora_or_offer_fix():
            return
        args = self._train_args(resume=True)
        if args is None:
            return
        self._reset_metrics()
        self.run_started_at = time.monotonic()
        self.status_var.set("resuming...")
        self.gui.spawn(args)

    def _preflight_te_lora_or_offer_fix(self) -> bool:
        """Catch the TE-LoRA-on-small-VRAM combo BEFORE the subprocess spawns.

        Returns ``True`` when it's safe to proceed with the spawn, ``False``
        when the user cancelled out. If the user accepts the offered fix
        (disable TE LoRA), the toggle is flipped, settings are persisted, and
        we return ``True`` so the caller can immediately spawn the run.
        """
        # Read the current GUI state of the checkbox, not the on-disk project,
        # because the user may have toggled it without saving.
        if not self.gui.te_lora_var.get():
            return True
        v = gui_helpers.probe_vram()
        if v is None:
            # No nvidia-smi — let the CLI's own gate handle it. Don't block.
            return True
        _used, total_mib = v
        total_gb = total_mib / 1024.0
        if total_gb >= 12.0:
            return True

        msg = (
            f"Text-encoder LoRA is enabled but your card has only {total_gb:.1f} GB "
            f"of VRAM. The training run will OOM at unet.to(device) before the "
            f"first step.\n\n"
            f"Disable Text-encoder LoRA and start training now?\n\n"
            f"(You can re-enable it any time on the Settings tab. Quality is "
            f"slightly lower without TE LoRA but the run actually fits.)"
        )
        choice = messagebox.askyesnocancel("Won't fit on this card", msg)
        if choice is None or choice is False:
            self.status_var.set("training cancelled — TE LoRA / VRAM mismatch")
            return False

        # User said yes — flip + save.
        self.gui.te_lora_var.set(False)
        self.gui.save_settings_silent()
        self.gui.log_queue.put(
            f"[Disabled Text-encoder LoRA (card has {total_gb:.1f} GB; needs ~12 GB)]\n"
        )
        return True

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
            # First training step (or a resume that jumped backward) starts
            # the ETA clock. Elapsed keeps counting from subprocess launch.
            # We anchor ``train_start_step`` to the first step value we see,
            # not ``step - 1`` — otherwise a resume from step 500 would miss
            # a step's worth of rate and give a misleading ETA.
            if self.train_started_at is None or step < self.last_step:
                self.train_started_at = time.monotonic()
                self.train_start_step = step
                self.sparkline.clear()
            self.last_step = step
            self.total_steps = total
            loss = parsed.get("loss")
            if isinstance(loss, (int, float)):
                self.sparkline.push(float(loss))
            self._update_timers()

    def _update_timers(self) -> None:
        """Refresh the elapsed + ETA labels from the two timers."""
        if self.run_started_at is not None:
            elapsed = time.monotonic() - self.run_started_at
            self.elapsed_var.set(f"elapsed {gui_helpers.format_elapsed(elapsed)}")
        if (
            self.train_started_at is not None
            and self.last_step > 0
            and self.total_steps > 0
        ):
            train_elapsed = time.monotonic() - self.train_started_at
            steps_done = max(1, self.last_step - self.train_start_step)
            self.eta_var.set(
                gui_helpers.format_eta(train_elapsed, steps_done, self.total_steps - self.train_start_step)
            )

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
        # Keep elapsed + ETA ticking while the subprocess is alive, even
        # during the caching / model-load phase before the first training
        # step lands.
        if self.run_started_at is not None and self.gui.runner.is_running():
            self._update_timers()

    def _reset_metrics(self) -> None:
        self.progress["value"] = 0
        self.pct_var.set("0%")
        self.step_var.set("step —/—")
        self.eta_var.set("ETA --:--:--")
        self.elapsed_var.set("elapsed 00:00:00")
        self.sparkline.clear()
        self.run_started_at = None
        self.train_started_at = None
        self.train_start_step = 0
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
