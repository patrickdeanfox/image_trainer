"""08 · Video tab — overnight image-to-video pipeline scaffold.

Drives the user's overnight workflow:

    Wan2GP I2V → Real-ESRGAN upscale → RIFE interpolate → ffmpeg assemble

Wan2GP itself runs as its own Gradio app (it's a long, interactive
generation that the user typically supervises through Wan2GP's own UI), so
this tab does NOT try to drive the model directly. Instead it:

1. Detects which tools are installed on PATH (ffmpeg, realesrgan-ncnn-vulkan,
   rife-ncnn-vulkan, optionally a wan2gp script if the user added one).
2. Lets the user pick a starting still image (e.g. a Generate-tab output)
   and write a motion prompt.
3. Provides a "Free VRAM" + "Inhibit sleep" pre-flight pair so the overnight
   run actually finishes.
4. Hands off to Wan2GP for generation; once the raw .mp4 lands, the user
   clicks "Run post-pipeline" and phases 4-7 run automatically via the
   ``trainer video-post`` subcommand.

Everything writes to ``<project>/video/<timestamp>/`` so a crash mid-run
keeps every prior phase's output intact.
"""

from __future__ import annotations

import os
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Optional

from .. import gui_helpers, gui_theme, wan2gp_installer
from ..gui_widgets import FolderField, info_icon

if TYPE_CHECKING:
    from ..gui_app import TrainerGUI


# Tool dependency table — rendered as a pre-flight checklist at the top.
# Each row: (binary_name_on_path, friendly_label, install_hint).
_TOOLS = [
    (
        "ffmpeg", "ffmpeg",
        "sudo apt install ffmpeg",
    ),
    (
        "realesrgan-ncnn-vulkan", "realesrgan-ncnn-vulkan",
        "Download standalone binary from "
        "https://github.com/xinntao/Real-ESRGAN/releases (the *-ncnn-vulkan-* zip)",
    ),
    (
        "rife-ncnn-vulkan", "rife-ncnn-vulkan",
        "Download standalone binary from "
        "https://github.com/nihui/rife-ncnn-vulkan/releases",
    ),
    (
        "systemd-inhibit", "systemd-inhibit (sleep guard)",
        "Comes with systemd on most Linux distros; if missing you can replace "
        "the inhibit step with a manual `xset s off`.",
    ),
]


def build(gui: "TrainerGUI") -> None:
    state = _VideoState(gui)
    gui.video_state = state
    state.build_ui(gui.tab_video)


class _VideoState:
    def __init__(self, gui: "TrainerGUI") -> None:
        self.gui = gui
        self.tool_status: dict[str, ttk.Label] = {}
        # The currently-active sleep inhibitor process (systemd-inhibit). We
        # hold the Popen so the user can release it from the same tab.
        self.inhibitor: Optional[subprocess.Popen] = None
        # Wan2GP install/launch state
        self.wan2gp_proc: Optional[subprocess.Popen] = None
        self.wan2gp_installer_thread = None
        # Resolve the install path from per-user settings, defaulting to
        # ~/Apps/wan2gp/ — the same convention as projects root. If the user
        # picks a different path it gets persisted on Save.
        user = gui_helpers.load_user_settings(self.gui.projects_root.root)
        self.wan2gp_install_path_var: Optional[tk.StringVar] = None  # built later
        self._initial_wan2gp_path = user.get(
            "wan2gp_install_root",
            str(wan2gp_installer.DEFAULT_INSTALL_ROOT),
        )

    # ---- layout ----

    def build_ui(self, root: ttk.Frame) -> None:
        PAD = gui_theme.PAD
        root.columnconfigure(0, weight=1)
        # Make rows expand reasonably; the form section gets the most room.
        root.rowconfigure(3, weight=1)

        # Header
        ttk.Label(
            root, text="Video · overnight image-to-video", style="Header.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, PAD))
        ttk.Label(
            root,
            text=(
                "Pipeline: Wan2GP I2V → Real-ESRGAN upscale → RIFE interpolate → "
                "ffmpeg encode. Wan2GP itself runs as its own app — generate the "
                "raw .mp4 there overnight, then drop it here for upscale + RIFE."
            ),
            style="Muted.TLabel", justify="left", wraplength=900,
        ).grid(row=1, column=0, sticky="w", pady=(0, PAD))

        # Tool checklist
        tools_box = ttk.LabelFrame(root, text="Pre-flight · tools on PATH", padding=PAD)
        tools_box.grid(row=2, column=0, sticky="we", pady=(0, PAD))
        tools_box.columnconfigure(0, weight=1)
        self._build_tools_table(tools_box)

        # Notebook with two flows: A) prep + launch generation; B) post-process raw mp4.
        nb = ttk.Notebook(root)
        nb.grid(row=3, column=0, sticky="nswe", pady=(0, PAD))

        gen_tab = ttk.Frame(nb, padding=PAD)
        post_tab = ttk.Frame(nb, padding=PAD)
        nb.add(gen_tab, text="A · Generate (Wan2GP)")
        nb.add(post_tab, text="B · Post-process (upscale + RIFE)")

        self._build_generate_tab(gen_tab)
        self._build_post_tab(post_tab)

        # Status row
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(root, textvariable=self.status_var, style="Status.TLabel").grid(
            row=4, column=0, sticky="w", pady=(0, PAD),
        )

    # ---- tool detection ----

    def _build_tools_table(self, root: ttk.LabelFrame) -> None:
        PAD = gui_theme.PAD
        for r, (binary, label, hint) in enumerate(_TOOLS):
            row = ttk.Frame(root)
            row.grid(row=r, column=0, sticky="we", pady=2)
            row.columnconfigure(2, weight=1)
            status = ttk.Label(row, text="…", style="Mono.TLabel", width=10)
            status.grid(row=0, column=0, sticky="w")
            self.tool_status[binary] = status
            ttk.Label(row, text=label, style="Mono.TLabel").grid(
                row=0, column=1, sticky="w", padx=(0, PAD),
            )
            ttk.Label(
                row, text=hint, style="Status.TLabel", justify="left",
                wraplength=620,
            ).grid(row=0, column=2, sticky="w")
        ttk.Button(
            root, text="Re-check", style="Ghost.TButton",
            command=self.refresh_tool_status,
        ).grid(row=len(_TOOLS), column=0, sticky="w", pady=(PAD, 0))
        self.refresh_tool_status()

    def refresh_tool_status(self) -> None:
        t = gui_theme.THEME
        for binary, label in self.tool_status.items():
            present = gui_helpers.which(binary) is not None
            label.configure(
                text="✓ found" if present else "✗ missing",
                foreground=t.ACCENT_GREEN if present else t.ACCENT_RED,
            )

    # ---- Wan2GP install / launch panel ----

    def _build_wan2gp_panel(self, root: ttk.Frame, *, row: int) -> None:
        """Install + launch panel for Wan2GP, rendered at the top of flow A."""
        PAD = gui_theme.PAD
        box = ttk.LabelFrame(root, text="Wan2GP · install / launch", padding=PAD)
        box.grid(row=row, column=0, columnspan=4, sticky="we", pady=(0, PAD))
        box.columnconfigure(1, weight=1)

        # Install path row.
        ttk.Label(box, text="Install path:").grid(row=0, column=0, sticky="w", pady=2)
        info_icon(
            box,
            "Folder where Wan2GP will be cloned. The installer creates "
            "<this path>/Wan2GP/ with the source + an isolated venv. Keep "
            "this on a fast SSD with at least 30 GB free for source + "
            "venv + downloaded model weights.",
        ).grid(row=0, column=0, sticky="e")
        self.wan2gp_install_path_var = tk.StringVar(value=self._initial_wan2gp_path)
        FolderField(
            box,
            textvariable=self.wan2gp_install_path_var,
            browse_title="Pick Wan2GP install folder",
            file_mode=False,
        ).grid(row=0, column=1, columnspan=3, sticky="we", padx=PAD)

        # Status row + buttons.
        ctl = ttk.Frame(box)
        ctl.grid(row=1, column=0, columnspan=4, sticky="we", pady=(PAD, 0))
        self.wan2gp_status_var = tk.StringVar(value="status: checking…")
        ttk.Label(
            ctl, textvariable=self.wan2gp_status_var, style="Mono.TLabel",
        ).pack(side="left")
        ttk.Button(
            ctl, text="Re-check", style="Ghost.TButton",
            command=self.refresh_wan2gp_status,
        ).pack(side="left", padx=(PAD, 0))
        self.wan2gp_install_btn = ttk.Button(
            ctl, text="Install / update", style="Primary.TButton",
            command=self._on_install_wan2gp,
        )
        self.wan2gp_install_btn.pack(side="right")
        self.wan2gp_launch_btn = ttk.Button(
            ctl, text="Launch Wan2GP", style="Ghost.TButton",
            command=self._on_launch_wan2gp,
        )
        self.wan2gp_launch_btn.pack(side="right", padx=(0, PAD // 2))
        self.wan2gp_stop_btn = ttk.Button(
            ctl, text="Stop", style="Caution.TButton",
            command=self._on_stop_wan2gp,
        )
        self.wan2gp_stop_btn.pack(side="right", padx=(0, PAD // 2))
        info_icon(
            ctl,
            "Install: clones the Wan2GP repo, sets up an isolated Python "
            "venv, pip-installs dependencies. Idempotent — re-runs as a "
            "git pull + pip update. Takes 5-15 min on first run depending "
            "on connection. Model weights download on first generation, "
            "not at install time.",
        ).pack(side="right", padx=(0, PAD // 2))
        # Initial state is updated lazily — refresh on tab build.
        self.refresh_wan2gp_status()

    def refresh_wan2gp_status(self) -> None:
        """Probe the on-disk install state and recolour the buttons + label."""
        if self.wan2gp_install_path_var is None:
            return
        path = Path(self.wan2gp_install_path_var.get()).expanduser()
        status = wan2gp_installer.detect(path)

        # Build a one-line status string and colour cue.
        t = gui_theme.THEME
        if status.fully_installed:
            self.wan2gp_status_var.set(
                f"status: ✓ installed at {path}  ·  entry: {status.entry_script.name}"
            )
            try:
                self.wan2gp_launch_btn.state(["!disabled"])
                self.wan2gp_install_btn.configure(text="Update (re-pull + re-install)")
            except Exception:
                pass
        else:
            missing = []
            if not status.repo_present:
                missing.append("repo")
            if not status.venv_present:
                missing.append("venv")
            if status.entry_script is None and status.repo_present:
                missing.append("entry script")
            self.wan2gp_status_var.set(
                f"status: ✗ not installed  ·  missing: {', '.join(missing) or 'all'}"
            )
            try:
                self.wan2gp_launch_btn.state(["disabled"])
                self.wan2gp_install_btn.configure(text="Install Wan2GP")
            except Exception:
                pass
        # Always allow Stop only when a launched process is alive.
        live = self.wan2gp_proc is not None and self.wan2gp_proc.poll() is None
        try:
            self.wan2gp_stop_btn.state(["!disabled"] if live else ["disabled"])
        except Exception:
            pass

    def _persist_wan2gp_path(self) -> Path:
        """Save the current path to per-user settings + return resolved Path."""
        path = Path(self.wan2gp_install_path_var.get()).expanduser().resolve()
        gui_helpers.update_user_setting(
            self.gui.projects_root.root,
            "wan2gp_install_root",
            str(path),
        )
        return path

    def _on_install_wan2gp(self) -> None:
        if self.wan2gp_installer_thread is not None and self.wan2gp_installer_thread.is_alive():
            messagebox.showinfo(
                "Already running",
                "An install/update is already in progress. Watch the Telemetry pane.",
            )
            return
        if self.wan2gp_install_path_var is None:
            return
        path = self._persist_wan2gp_path()

        # Confirm before running — this clones/pulls + does a real pip install
        # (5-15 min, fairly heavy).
        ok = messagebox.askokcancel(
            "Install Wan2GP",
            f"Install (or update) Wan2GP into:\n  {path}/Wan2GP/\n\n"
            f"This will:\n"
            f"  • git clone (or pull) the Wan2GP repo\n"
            f"  • create a Python venv at {path}/Wan2GP/.venv/\n"
            f"  • pip-install Wan2GP's requirements (~5-15 min on first run)\n\n"
            f"Model weights are downloaded by Wan2GP itself on first generation, "
            f"not now.\n\nProceed?",
        )
        if not ok:
            return

        # Run install in a worker thread so the GUI stays responsive. Stream
        # output through gui.log_queue so it lands in the Telemetry pane,
        # exactly the same channel the trainer subprocess uses.
        import threading

        def _worker() -> None:
            def _emit(msg: str) -> None:
                self.gui.log_queue.put(msg + "\n")

            try:
                _emit("[wan2gp installer starting]")
                wan2gp_installer.install(path, progress=_emit)
                _emit("[wan2gp installer done]")
            except Exception as e:
                _emit(f"[wan2gp installer failed: {e}]")
            # Hop back to the Tk thread to refresh status.
            try:
                self.gui.root.after(0, self.refresh_wan2gp_status)
            except Exception:
                pass

        self.wan2gp_installer_thread = threading.Thread(target=_worker, daemon=True)
        self.wan2gp_installer_thread.start()
        self.status_var.set("Wan2GP installer running — watch the Telemetry pane")

    def _on_launch_wan2gp(self) -> None:
        if self.wan2gp_install_path_var is None:
            return
        path = self._persist_wan2gp_path()
        if self.wan2gp_proc is not None and self.wan2gp_proc.poll() is None:
            messagebox.showinfo(
                "Already running",
                "Wan2GP is already running — open the Gradio URL in your browser. "
                "Click Stop to terminate it.",
            )
            return
        try:
            # Capture output so we can stream Wan2GP's startup log into the
            # telemetry pane — handy because that's where the Gradio URL prints.
            self.wan2gp_proc = wan2gp_installer.launch(path, capture_output=True)
        except Exception as e:
            messagebox.showerror("Launch failed", str(e))
            self.refresh_wan2gp_status()
            return

        # Pump Wan2GP's output into the existing log queue on a daemon thread.
        import threading

        def _pump() -> None:
            assert self.wan2gp_proc is not None
            assert self.wan2gp_proc.stdout is not None
            for line in self.wan2gp_proc.stdout:
                self.gui.log_queue.put(f"[wan2gp] {line}")
            rc = self.wan2gp_proc.wait()
            self.gui.log_queue.put(f"[wan2gp exited rc={rc}]\n")
            try:
                self.gui.root.after(0, self.refresh_wan2gp_status)
            except Exception:
                pass

        threading.Thread(target=_pump, daemon=True).start()
        self.status_var.set(
            f"Wan2GP launched (pid {self.wan2gp_proc.pid}) — "
            "watch Telemetry for the Gradio URL"
        )
        self.refresh_wan2gp_status()

    def _on_stop_wan2gp(self) -> None:
        if self.wan2gp_proc is None or self.wan2gp_proc.poll() is not None:
            self.status_var.set("Wan2GP not running")
            self.refresh_wan2gp_status()
            return
        try:
            self.wan2gp_proc.terminate()
            try:
                self.wan2gp_proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.wan2gp_proc.kill()
            self.status_var.set(f"Wan2GP stopped (pid {self.wan2gp_proc.pid})")
        except Exception as e:
            messagebox.showerror("Stop failed", str(e))
        finally:
            self.refresh_wan2gp_status()

    # ---- A: generate tab ----

    def _build_generate_tab(self, root: ttk.Frame) -> None:
        PAD = gui_theme.PAD
        root.columnconfigure(1, weight=1)

        # Wan2GP install/launch panel — first thing in flow A so the user
        # knows whether they can actually generate before they fill the form.
        self._build_wan2gp_panel(root, row=0)

        # Source still
        ttk.Label(root, text="Source image:").grid(row=1, column=0, sticky="w", pady=2)
        info_icon(
            root,
            "The starting still that Wan2GP will animate. Best results: a "
            "Generate-tab output where the composition leaves room for motion. "
            "Avoid tight face crops and awkward hand poses (Wan still struggles "
            "with hands).",
        ).grid(row=1, column=0, sticky="e")
        self.source_image_var = tk.StringVar()
        FolderField(
            root, textvariable=self.source_image_var,
            browse_title="Pick a starting still",
            file_mode=True,
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")],
        ).grid(row=1, column=1, columnspan=3, sticky="we", padx=PAD)

        # Motion prompt
        ttk.Label(root, text="Motion prompt:").grid(row=2, column=0, sticky="nw", pady=2)
        info_icon(
            root,
            "Describe what changes frame-to-frame, not just what's in the "
            "image. 'person turns head slowly to the right, hair moves in "
            "gentle breeze, soft smile develops' beats 'a beautiful scene'. "
            "Wan 2.2 rewards specific motion language.",
        ).grid(row=2, column=0, sticky="e")
        self.motion_prompt = tk.Text(
            root, height=4, wrap="word",
            background=gui_theme.THEME.BG_INPUT,
            foreground=gui_theme.THEME.TEXT_PRIMARY,
            font=gui_theme.THEME.FONT_BODY, relief="flat", borderwidth=0,
            highlightthickness=1,
            highlightbackground=gui_theme.THEME.DIVIDER,
            highlightcolor=gui_theme.THEME.FOCUS_RING,
        )
        self.motion_prompt.grid(row=2, column=1, columnspan=3, sticky="we", padx=PAD)

        # Wan2GP knobs row 1: target seconds, framerate, lightning
        knob1 = ttk.Frame(root)
        knob1.grid(row=3, column=0, columnspan=4, sticky="we", pady=(PAD, 0))
        ttk.Label(knob1, text="Target seconds:").pack(side="left")
        info_icon(
            knob1,
            "Wall-clock seconds of generated video you want. With Lightning "
            "LoRA + Q4_K_M weights on a 10 GB card, ~10 seconds per hour of "
            "generation is realistic. 6-7 hours overnight ≈ 60-70s of raw "
            "video before upscale.",
        ).pack(side="left")
        self.target_seconds_var = tk.StringVar(value="60")
        ttk.Spinbox(
            knob1, from_=2, to=120, increment=2,
            textvariable=self.target_seconds_var, width=6,
        ).pack(side="left", padx=(PAD // 2, PAD))

        ttk.Label(knob1, text="Wan source fps:").pack(side="left")
        info_icon(
            knob1,
            "Wan 2.2's native framerate (16). Don't change unless your local "
            "Wan2GP build is configured differently. RIFE later doubles this "
            "to 32 fps.",
        ).pack(side="left")
        self.wan_fps_var = tk.StringVar(value="16")
        ttk.Spinbox(
            knob1, from_=8, to=30, increment=1,
            textvariable=self.wan_fps_var, width=6,
        ).pack(side="left", padx=(PAD // 2, PAD))

        ttk.Checkbutton(
            knob1, text="Lightning LoRA (6 steps)",
            variable=tk.BooleanVar(value=True),
        ).pack(side="left", padx=(PAD, 0))
        info_icon(
            knob1,
            "Alibaba's Lightning LoRAs cut sampling steps from 20-30 to 4-8 "
            "with barely any quality loss — typically a 3-5x speedup. "
            "Without it, an overnight run gets you maybe 20s of video; with "
            "it, 60-90+s. Always use it for overnight runs.",
        ).pack(side="left")

        # Wan2GP knobs row 2: block swap, window size
        knob2 = ttk.Frame(root)
        knob2.grid(row=4, column=0, columnspan=4, sticky="we", pady=(PAD // 2, 0))
        ttk.Label(knob2, text="Block swap:").pack(side="left")
        info_icon(
            knob2,
            "How many transformer blocks Wan2GP swaps between GPU and CPU. "
            "Higher = less VRAM per step but slower. On a 10 GB card with "
            "Wan 2.2 14B, set to 30+ to fit comfortably; 40+ for safety on "
            "longer generations.",
        ).pack(side="left")
        self.block_swap_var = tk.StringVar(value="35")
        ttk.Spinbox(
            knob2, from_=0, to=60, increment=5,
            textvariable=self.block_swap_var, width=6,
        ).pack(side="left", padx=(PAD // 2, PAD))

        ttk.Label(knob2, text="Window frames:").pack(side="left")
        info_icon(
            knob2,
            "Frames per sliding-window chunk. 81 (~5s at 16fps) is Wan 2.2's "
            "sweet spot and fits memory comfortably with offloading. Smaller "
            "windows are faster per chunk but introduce more transition seams.",
        ).pack(side="left")
        self.window_var = tk.StringVar(value="81")
        ttk.Spinbox(
            knob2, from_=33, to=121, increment=8,
            textvariable=self.window_var, width=6,
        ).pack(side="left", padx=(PAD // 2, PAD))

        ttk.Label(knob2, text="Window overlap:").pack(side="left")
        info_icon(
            knob2,
            "Frames each window overlaps with the previous one. Larger = "
            "smoother transitions between windows but more total work. 16 is "
            "the documented balance point for Wan 2.2.",
        ).pack(side="left")
        self.overlap_var = tk.StringVar(value="16")
        ttk.Spinbox(
            knob2, from_=4, to=32, increment=4,
            textvariable=self.overlap_var, width=6,
        ).pack(side="left", padx=(PAD // 2, PAD))

        # Pre-flight buttons row
        pf_row = ttk.Frame(root)
        pf_row.grid(row=5, column=0, columnspan=4, sticky="we", pady=(PAD * 2, 0))
        ttk.Button(
            pf_row, text="Free VRAM", style="Ghost.TButton",
            command=self._on_free_vram,
        ).pack(side="left")
        info_icon(
            pf_row,
            "Same kill list as the Train tab — closes other GPU-using "
            "processes so Wan2GP can claim the full card before generation "
            "starts. Run this right before kicking off the overnight job.",
        ).pack(side="left", padx=(0, PAD))
        ttk.Button(
            pf_row, text="Inhibit sleep (overnight)", style="Ghost.TButton",
            command=self._on_inhibit_sleep,
        ).pack(side="left")
        info_icon(
            pf_row,
            "Calls systemd-inhibit so the desktop doesn't suspend or blank "
            "the screen mid-run. Releases automatically when you click "
            "'Release inhibitor' or quit the app. Linux only.",
        ).pack(side="left")
        self.inhibit_btn_var = tk.StringVar(value="Inhibit sleep (overnight)")
        ttk.Button(
            pf_row, text="Release inhibitor", style="Ghost.TButton",
            command=self._on_release_inhibitor,
        ).pack(side="left", padx=(PAD, 0))
        ttk.Button(
            pf_row, text="Copy generation plan…", style="Primary.TButton",
            command=self._copy_plan_to_clipboard,
        ).pack(side="right")
        info_icon(
            pf_row,
            "Copies a ready-to-paste plan to your clipboard with all the "
            "Wan2GP settings filled in (image path, prompt, frame count, "
            "block swap, etc.). Open Wan2GP, paste, and start the job.",
        ).pack(side="right", padx=(0, PAD // 2))

        # Help block
        help_box = ttk.LabelFrame(root, text="How to drive Wan2GP", padding=PAD)
        help_box.grid(row=6, column=0, columnspan=4, sticky="we", pady=(PAD * 2, 0))
        help_box.columnconfigure(0, weight=1)
        ttk.Label(
            help_box,
            text=(
                "1. Launch your Wan2GP install in another terminal "
                "(./run.sh or `python wgp.py` in the Wan2GP folder).\n"
                "2. Click 'Copy generation plan' above and paste into "
                "Wan2GP's prompt + advanced fields.\n"
                "3. Set output folder so the raw .mp4 lands somewhere you "
                "can find it (any folder works — you'll point this tab at "
                "it in flow B).\n"
                "4. Hit 'Generate' in Wan2GP and walk away.\n"
                "5. In the morning: switch to flow B above, point at the "
                "raw .mp4, click 'Run post-process'."
            ),
            justify="left", wraplength=820,
        ).grid(row=0, column=0, sticky="w")

    # ---- B: post-process tab ----

    def _build_post_tab(self, root: ttk.Frame) -> None:
        PAD = gui_theme.PAD
        root.columnconfigure(1, weight=1)

        ttk.Label(root, text="Raw mp4 from Wan2GP:").grid(row=0, column=0, sticky="w", pady=2)
        self.raw_mp4_var = tk.StringVar()
        FolderField(
            root, textvariable=self.raw_mp4_var,
            browse_title="Pick the Wan2GP raw .mp4",
            file_mode=True,
            filetypes=[("Video", "*.mp4 *.mov *.mkv *.webm"), ("All files", "*.*")],
        ).grid(row=0, column=1, columnspan=3, sticky="we", padx=PAD)

        # Post-pipeline knobs
        knob_row = ttk.Frame(root)
        knob_row.grid(row=1, column=0, columnspan=4, sticky="we", pady=(PAD, 0))

        ttk.Label(knob_row, text="Upscale model:").pack(side="left")
        info_icon(
            knob_row,
            "Real-ESRGAN model. 'realesr-animevideov3' is the best default "
            "for AI-generated video (handles soft, painterly output well). "
            "For photoreal subjects try 'realesrgan-x4plus' (4x native; pair "
            "with scale=2 to net 2x).",
        ).pack(side="left")
        self.upscale_model_var = tk.StringVar(value="realesr-animevideov3")
        ttk.Combobox(
            knob_row, textvariable=self.upscale_model_var,
            values=["realesr-animevideov3", "realesrgan-x4plus", "realesrgan-x4plus-anime"],
            width=24,
        ).pack(side="left", padx=(PAD // 2, PAD))

        ttk.Label(knob_row, text="Upscale ×:").pack(side="left")
        self.upscale_scale_var = tk.StringVar(value="2")
        ttk.Combobox(
            knob_row, textvariable=self.upscale_scale_var,
            values=["2", "3", "4"], state="readonly", width=4,
        ).pack(side="left", padx=(PAD // 2, PAD))

        ttk.Label(knob_row, text="RIFE ×:").pack(side="left")
        info_icon(
            knob_row,
            "Frame interpolation factor. 2 = double the framerate (16→32fps), "
            "4 = quadruple (16→64fps; overkill for most uses).",
        ).pack(side="left")
        self.rife_mult_var = tk.StringVar(value="2")
        ttk.Combobox(
            knob_row, textvariable=self.rife_mult_var,
            values=["2", "4"], state="readonly", width=4,
        ).pack(side="left", padx=(PAD // 2, PAD))

        ttk.Label(knob_row, text="Final fps:").pack(side="left")
        self.final_fps_var = tk.StringVar(value="32")
        ttk.Spinbox(
            knob_row, from_=12, to=120, increment=2,
            textvariable=self.final_fps_var, width=6,
        ).pack(side="left", padx=(PAD // 2, 0))

        # Run button
        run_row = ttk.Frame(root)
        run_row.grid(row=2, column=0, columnspan=4, sticky="we", pady=(PAD * 2, 0))
        ttk.Button(
            run_row, text="Run post-process", style="Primary.TButton",
            command=self._on_run_post,
        ).pack(side="left")
        ttk.Button(
            run_row, text="Open project video folder", style="Ghost.TButton",
            command=self._open_video_dir,
        ).pack(side="left", padx=(PAD, 0))

    # ---- handlers ----

    def _on_free_vram(self) -> None:
        # Reuse the train-tab logic by importing its kill helper.
        from .. import gui_helpers as gh
        procs = gh.list_gpu_processes()
        my_pid = os.getpid()
        excluded = {my_pid}
        if self.gui.runner.is_running() and self.gui.runner.proc is not None:
            excluded.add(self.gui.runner.proc.pid)
        targets = [p for p in procs if p.pid not in excluded]
        if not targets:
            messagebox.showinfo(
                "Nothing to free",
                "No other GPU-using processes detected.",
            )
            return
        body = "\n".join(
            f"  pid {p.pid:>7}  ·  {p.used_mib:>5} MiB  ·  {p.name}" for p in targets
        )
        if not messagebox.askyesno(
            "Free VRAM",
            f"Send SIGTERM to {len(targets)} process(es)?\n\n{body}\n\n"
            "Anything not exiting in 2s gets SIGKILL.",
        ):
            return
        gh.kill_processes([p.pid for p in targets])
        v = gh.probe_vram()
        post = f" — VRAM now {v[0]}/{v[1]} MiB" if v else ""
        self.status_var.set(f"freed {len(targets)} process(es){post}")

    def _on_inhibit_sleep(self) -> None:
        if self.inhibitor is not None and self.inhibitor.poll() is None:
            messagebox.showinfo(
                "Already inhibited",
                "Sleep inhibitor is already running. Click 'Release inhibitor' "
                "to lift it.",
            )
            return
        if gui_helpers.which("systemd-inhibit") is None:
            messagebox.showerror(
                "systemd-inhibit not found",
                "Couldn't find systemd-inhibit on PATH. Either install systemd "
                "or use `xset s off; xset -dpms` manually before starting the run.",
            )
            return
        try:
            self.inhibitor = subprocess.Popen(
                [
                    "systemd-inhibit",
                    "--what=sleep:idle",
                    "--why=Image-trainer overnight video generation",
                    "sleep", "12h",
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.status_var.set(
                f"sleep inhibited (12h) — pid {self.inhibitor.pid}; "
                "release when you're done"
            )
        except Exception as e:
            messagebox.showerror("Inhibit failed", str(e))

    def _on_release_inhibitor(self) -> None:
        if self.inhibitor is None or self.inhibitor.poll() is not None:
            self.status_var.set("no active inhibitor")
            return
        try:
            self.inhibitor.terminate()
            self.inhibitor.wait(timeout=2)
        except Exception:
            try:
                self.inhibitor.kill()
            except Exception:
                pass
        self.inhibitor = None
        self.status_var.set("inhibitor released")

    def _copy_plan_to_clipboard(self) -> None:
        if not self.gui.current_project:
            messagebox.showerror("No project", "Open a project first.")
            return
        try:
            seconds = int(self.target_seconds_var.get())
            fps = int(self.wan_fps_var.get())
            target_frames = seconds * fps
        except ValueError:
            messagebox.showerror("Invalid number", "Target seconds and fps must be integers.")
            return
        prompt = self.motion_prompt.get("1.0", "end").strip() or "(write a motion prompt)"
        src = self.source_image_var.get().strip() or "(pick a source image)"
        plan = (
            f"Wan2GP overnight plan\n"
            f"---------------------\n"
            f"Source image:    {src}\n"
            f"Motion prompt:   {prompt}\n"
            f"Target frames:   {target_frames}  ({seconds}s × {fps}fps)\n"
            f"Resolution:      1280×720 (or 720×1280 for portrait)\n"
            f"Model quant:     Wan 2.2 I2V 14B GGUF Q4_K_M\n"
            f"Lightning LoRA:  on  (6 sampling steps; CFG 1.0)\n"
            f"Without Lightning: 20 steps; CFG 3.5-5.0\n"
            f"Window frames:   {self.window_var.get()}\n"
            f"Window overlap:  {self.overlap_var.get()}\n"
            f"Block swap:      {self.block_swap_var.get()}+\n"
            f"VAE tiling:      on\n"
            f"Save intermediates: on (critical for crash safety)\n"
            f"Output mp4 path: somewhere you can find it for post-processing\n"
        )
        try:
            self.gui.root.clipboard_clear()
            self.gui.root.clipboard_append(plan)
            self.gui.root.update()  # commit clipboard before window may lose focus
            self.status_var.set("plan copied to clipboard")
        except Exception as e:
            messagebox.showerror("Clipboard error", str(e))

    def _on_run_post(self) -> None:
        if not self.gui.current_project:
            messagebox.showerror("No project", "Open a project first.")
            return
        raw = self.raw_mp4_var.get().strip()
        if not raw or not Path(raw).exists():
            messagebox.showerror(
                "No raw video",
                "Pick the Wan2GP raw .mp4 first (Browse beside 'Raw mp4 from Wan2GP').",
            )
            return
        # Verify the binaries are present before we kick off a long job.
        missing = [
            b for b, _label, _hint in _TOOLS
            if b in ("ffmpeg", "realesrgan-ncnn-vulkan", "rife-ncnn-vulkan")
            and gui_helpers.which(b) is None
        ]
        if missing:
            messagebox.showerror(
                "Missing tools",
                "These tools need to be on PATH before post-processing:\n  "
                + "\n  ".join(missing)
                + "\n\nSee the pre-flight checklist at the top.",
            )
            return
        args = [
            "video-post",
            str(self.gui.current_project.root),
            raw,
            "--framerate", self.final_fps_var.get() or "32",
            "--rife-multiplier", self.rife_mult_var.get() or "2",
            "--upscale-model", self.upscale_model_var.get() or "realesr-animevideov3",
            "--upscale-scale", self.upscale_scale_var.get() or "2",
        ]
        self.gui.spawn(args)
        self.status_var.set("post-pipeline started — see Telemetry pane for progress")

    def _open_video_dir(self) -> None:
        if not self.gui.current_project:
            return
        d = self.gui.current_project.root / "video"
        d.mkdir(parents=True, exist_ok=True)
        gui_helpers.open_folder(d)
