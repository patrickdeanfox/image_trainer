"""07 · Storage tab — see what each project folder is using, delete safely.

The user can re-run prep / caption / train multiple times across the
lifetime of a project, which can leave stale ingested copies, cache
tensors, or training checkpoints lying around. This tab gives them a
single place to:

- See the size + file count of every per-project subfolder.
- Delete any one folder (with confirmation).
- Wipe the entire project (danger zone; requires typed confirmation).

The deletes go through ``shutil.rmtree`` directly rather than shelling
into ``trainer clean`` so we can update the row's size readout in place
without round-tripping a subprocess. Project-wide delete also drops the
project from the Recent menu and switches the GUI back to the picker.
"""

from __future__ import annotations

import shutil
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import TYPE_CHECKING, Callable, Optional

from .. import gui_helpers, gui_theme

if TYPE_CHECKING:
    from ..gui_app import TrainerGUI


# Each row: (folder attribute name on Project, display label, blurb, regenerable?)
# "regenerable" controls the colour of the Delete button — yellow ghost for
# safe-to-rebuild things (cache, checkpoints), red Caution button for
# destructive ones (raw, processed, lora, outputs).
_ROWS = [
    ("raw_dir",         "raw",         "Originals copied in by ingest. Safe to delete only if your sources still exist elsewhere.", False),
    ("processed_dir",   "processed",   "Resized PNGs + caption .txt files. Re-runnable from raw via Prep + Caption.",                False),
    ("cache_dir",       "cache",       "Pre-computed VAE latents + text embeddings. Auto-rebuilt on next training run.",            True),
    ("checkpoints_dir", "checkpoints", "Intermediate training checkpoints used by --resume. Safe to delete after training finishes.", True),
    ("lora_dir",        "lora",        "Final exported LoRA weights. Deleting these means you have to retrain.",                     False),
    ("outputs_dir",     "outputs",     "Generated images from the Generate tab.",                                                    False),
    ("logs_dir",        "logs",        "Training logs, journal, validation previews. Auto-recreated next run.",                     True),
]
# Two folders that aren't first-class on Project but commonly hang around:
_EXTRA_ROWS = [
    ("preview", "preview", "Dry-run thumbnails written by Prep --dry-run.", True),
]


def build(gui: "TrainerGUI") -> None:
    state = _StorageState(gui)
    gui.storage_state = state
    state.build_ui(gui.tab_storage)


class _StorageState:
    def __init__(self, gui: "TrainerGUI") -> None:
        self.gui = gui
        # rows: stem name -> {"path", "size_var", "count_var", "delete_btn"}
        self.rows: dict[str, dict] = {}

    def build_ui(self, root: ttk.Frame) -> None:
        PAD = gui_theme.PAD
        root.columnconfigure(0, weight=1)

        ttk.Label(
            root, text="Storage · what's on disk", style="Header.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, PAD))

        ttk.Label(
            root,
            text=(
                "Every per-project subfolder, with its size + file count. "
                "'Refresh' rescans from disk; 'Delete' wipes one folder; the "
                "danger zone at the bottom wipes the whole project."
            ),
            style="Muted.TLabel", justify="left", wraplength=900,
        ).grid(row=1, column=0, sticky="w", pady=(0, PAD))

        # Refresh + total summary row
        top = ttk.Frame(root)
        top.grid(row=2, column=0, sticky="we", pady=(0, PAD))
        ttk.Button(top, text="Refresh", style="Ghost.TButton",
                   command=self.refresh).pack(side="left")
        self.total_var = tk.StringVar(value="—")
        ttk.Label(top, textvariable=self.total_var, style="Mono.TLabel").pack(
            side="right",
        )

        # Table of folder rows.
        table = ttk.Frame(root)
        table.grid(row=3, column=0, sticky="nswe", pady=(0, PAD * 2))
        table.columnconfigure(0, weight=0, minsize=120)
        table.columnconfigure(1, weight=0, minsize=110)
        table.columnconfigure(2, weight=0, minsize=90)
        table.columnconfigure(3, weight=1)
        table.columnconfigure(4, weight=0)
        table.columnconfigure(5, weight=0)

        # Header row
        for col, label in enumerate(("Folder", "Size", "Files", "What it holds", "", "")):
            ttk.Label(
                table, text=label, style="SubHeader.TLabel",
            ).grid(row=0, column=col, sticky="w", padx=(0, PAD), pady=(0, 4))

        # Data rows. Build (project-path-attr, display name, blurb, regenerable).
        all_rows = list(_ROWS) + [(name, name, blurb, regen) for name, _disp, blurb, regen in _EXTRA_ROWS]

        for r, (key, display, blurb, regenerable) in enumerate(all_rows, start=1):
            size_var = tk.StringVar(value="—")
            count_var = tk.StringVar(value="—")
            ttk.Label(table, text=display, style="Mono.TLabel").grid(
                row=r, column=0, sticky="w", padx=(0, PAD), pady=2,
            )
            ttk.Label(table, textvariable=size_var, style="Mono.TLabel").grid(
                row=r, column=1, sticky="w", padx=(0, PAD), pady=2,
            )
            ttk.Label(table, textvariable=count_var, style="Mono.TLabel").grid(
                row=r, column=2, sticky="w", padx=(0, PAD), pady=2,
            )
            ttk.Label(
                table, text=blurb, style="Status.TLabel",
                wraplength=520, justify="left",
            ).grid(row=r, column=3, sticky="w", padx=(0, PAD), pady=2)
            btn_style = "Ghost.TButton" if regenerable else "Caution.TButton"
            del_btn = ttk.Button(
                table, text="Delete", style=btn_style,
                command=lambda k=key, d=display: self._on_delete_folder(k, d),
            )
            del_btn.grid(row=r, column=4, sticky="e", padx=(PAD, 0), pady=2)
            open_btn = ttk.Button(
                table, text="Open", style="Ghost.TButton",
                command=lambda k=key: self._on_open_folder(k),
            )
            open_btn.grid(row=r, column=5, sticky="e", padx=(PAD // 2, 0), pady=2)
            self.rows[key] = {
                "size_var": size_var,
                "count_var": count_var,
                "delete_btn": del_btn,
                "open_btn": open_btn,
                "display": display,
            }

        # Danger zone — entire-project delete.
        danger = ttk.LabelFrame(root, text="Danger zone", padding=PAD)
        danger.grid(row=4, column=0, sticky="we", pady=(PAD, 0))
        danger.columnconfigure(0, weight=1)
        ttk.Label(
            danger,
            text=(
                "Delete the entire project — every folder above plus config.json "
                "and review.json. The project root itself is removed and won't "
                "be in your Recent list anymore. This cannot be undone."
            ),
            style="Warn.TLabel", justify="left", wraplength=820,
        ).grid(row=0, column=0, sticky="w", pady=(0, PAD))
        ttk.Button(
            danger, text="Delete entire project…", style="Caution.TButton",
            command=self._on_delete_project,
        ).grid(row=1, column=0, sticky="w")

    # ---- external API ----

    def on_tab_enter(self) -> None:
        if self.gui.current_project is None:
            return
        self.refresh()

    # ---- helpers ----

    def _resolve_path(self, key: str) -> Optional[Path]:
        project = self.gui.current_project
        if project is None:
            return None
        if hasattr(project, key):
            return getattr(project, key)
        # _EXTRA_ROWS entries live directly under root by name.
        return project.root / key

    def refresh(self) -> None:
        if not self.gui.current_project:
            self.total_var.set("(no project loaded)")
            for row in self.rows.values():
                row["size_var"].set("—")
                row["count_var"].set("—")
            return
        total_bytes = 0
        total_files = 0
        for key, row in self.rows.items():
            path = self._resolve_path(key)
            if path is None:
                continue
            b, n = gui_helpers.folder_size_and_count(path)
            row["size_var"].set(gui_helpers.format_bytes(b))
            row["count_var"].set(f"{n}")
            total_bytes += b
            total_files += n
        self.total_var.set(
            f"project total · {gui_helpers.format_bytes(total_bytes)} · {total_files} files"
        )

    def _on_open_folder(self, key: str) -> None:
        path = self._resolve_path(key)
        if path is None:
            return
        if not path.exists():
            # Create empty so the OS can open it; matches gui_helpers.open_folder behaviour.
            messagebox.showinfo(
                "Empty folder",
                f"{key} doesn't exist yet. Run the relevant pipeline step first.",
            )
            return
        gui_helpers.open_folder(path)

    def _on_delete_folder(self, key: str, display: str) -> None:
        path = self._resolve_path(key)
        if path is None:
            return
        if not path.exists():
            messagebox.showinfo(
                "Nothing to delete", f"{display}/ doesn't exist.",
            )
            return
        b, n = gui_helpers.folder_size_and_count(path)
        ok = messagebox.askokcancel(
            f"Delete {display}/",
            f"Delete {path}\n"
            f"({gui_helpers.format_bytes(b)} · {n} files)?\n\n"
            f"This cannot be undone.",
        )
        if not ok:
            return
        try:
            shutil.rmtree(path)
        except Exception as e:
            messagebox.showerror("Delete failed", f"{e}")
            return
        self.gui.log_queue.put(
            f"[deleted {path} ({gui_helpers.format_bytes(b)}, {n} files)]\n"
        )
        self.refresh()
        self.gui.refresh_step_status()

    def _on_delete_project(self) -> None:
        project = self.gui.current_project
        if project is None:
            messagebox.showerror("No project", "Nothing to delete — no project loaded.")
            return
        root = project.root
        b, n = gui_helpers.folder_size_and_count(root)

        # Two-step confirmation. The first messagebox sets expectations, the
        # second requires the user to type the project name. Belt + braces
        # because this wipes a directory the user spent hours captioning.
        ok = messagebox.askokcancel(
            "Delete entire project",
            f"This will permanently delete:\n\n  {root}\n\n"
            f"Total on disk: {gui_helpers.format_bytes(b)} across {n} files.\n\n"
            f"You will lose every image, caption, checkpoint, and trained LoRA "
            f"under this project. Continue?",
        )
        if not ok:
            return
        typed = simpledialog.askstring(
            "Confirm by typing the project name",
            f"To confirm, type the project folder name exactly:\n\n  {root.name}",
            parent=self.gui.root,
        )
        if (typed or "").strip() != root.name:
            messagebox.showinfo("Cancelled", "Project name didn't match — nothing deleted.")
            return
        try:
            shutil.rmtree(root)
        except Exception as e:
            messagebox.showerror("Delete failed", f"{e}")
            return

        self.gui.log_queue.put(
            f"[deleted project {root} ({gui_helpers.format_bytes(b)}, {n} files)]\n"
        )

        # Drop from recent + clear current project + refresh combo.
        try:
            from .. import gui_helpers as _gh
            recents = _gh.load_recent(self.gui.projects_root.root)
            recents = [p for p in recents if Path(p).resolve() != root.resolve()]
            _gh.recent_path(self.gui.projects_root.root).write_text(
                __import__("json").dumps(recents, indent=2)
            )
            self.gui.recent_projects = recents
            self.gui._rebuild_recent_menu()
        except Exception:
            pass

        self.gui.current_project = None
        self.gui._refresh_project_list()
        self.refresh()
        self.gui.status_var.set(f"deleted project {root.name}")
