"""04 · REVIEW tab — 3-pane list/preview/editor with a thumbnail grid toggle.

Per-image `{include, caption, notes}` is persisted to `review.json` at the
project root via :mod:`pipeline.review`. The grid view surfaces all
thumbnails at once so near-duplicates and exposure outliers jump out
immediately; the detail view is the old 3-pane workflow rebuilt for wider
screens.

Keyboard shortcuts (bound globally while this tab is active):
    ←/→ prev/next   I toggle include   Ctrl+S save
"""

from __future__ import annotations

import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Optional

from .. import gui_theme
from ..gui_widgets import ThumbnailGrid
from ..pipeline import insights, review as review_mod

if TYPE_CHECKING:
    from ..gui_app import TrainerGUI


def build(gui: "TrainerGUI") -> None:
    state = _ReviewState(gui)
    gui.review_state = state
    state.build_ui(gui.tab_review)


class _ReviewState:
    """All per-tab state + layout. Held on the GUI via ``gui.review_state``.

    Keeping the state object separate from the GUI singleton avoids the
    previous style of bolting 15+ attributes onto :class:`TrainerGUI`.
    """

    def __init__(self, gui: "TrainerGUI") -> None:
        self.gui = gui
        self.review: Optional[review_mod.Review] = None
        self.order: list[str] = []
        self.idx: int = -1
        self.photo_cache: deque = deque(maxlen=4)
        self.dupes_index: dict[str, list[tuple[str, int]]] = {}
        self.stats_index: dict[str, dict] = {}
        self.mode: str = "detail"   # or "grid"

    # ---- layout ----

    def build_ui(self, root: ttk.Frame) -> None:
        t = gui_theme.THEME
        PAD = gui_theme.PAD
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        top = ttk.Frame(root)
        top.grid(row=0, column=0, sticky="we", pady=(0, PAD))
        ttk.Label(top, text="Review · caption each image", style="Header.TLabel").pack(side="left")
        self.counts_var = tk.StringVar(value="—")
        ttk.Label(top, textvariable=self.counts_var, style="Status.TLabel").pack(side="right")

        # View toggle
        self.mode_var = tk.StringVar(value="DETAIL VIEW")
        self.mode_btn = ttk.Button(
            top, textvariable=self.mode_var, style="Ghost.TButton",
            command=self._toggle_mode,
        )
        self.mode_btn.pack(side="right", padx=PAD)
        ttk.Button(top, text="RELOAD", style="Ghost.TButton",
                   command=self.reload).pack(side="right", padx=(0, PAD))
        ttk.Button(top, text="SAVE", style="Ghost.TButton",
                   command=self.save).pack(side="right", padx=(0, PAD))

        # Two stacked frames, only one packed at a time.
        self.detail_frame = ttk.Frame(root)
        self.grid_frame = ttk.Frame(root, style="Panel.TFrame")
        self.detail_frame.grid(row=1, column=0, sticky="nswe")

        self._build_detail(self.detail_frame)
        self._build_grid(self.grid_frame)

        # shortcuts
        root.bind_all("<Control-s>", lambda _e: self.save())

    def _build_detail(self, root: ttk.Frame) -> None:
        t = gui_theme.THEME
        PAD = gui_theme.PAD
        root.columnconfigure(0, weight=0, minsize=220)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=1)
        root.rowconfigure(0, weight=1)

        # --- column 1: file list ---
        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nswe")
        left.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(
            left, width=22, activestyle="none", exportselection=False,
            background=t.BG_INPUT, foreground=t.TEXT_PRIMARY,
            selectbackground=t.ACCENT_AMBER, selectforeground=t.TEXT_ON_ACCENT,
            highlightthickness=1, highlightbackground=t.DIVIDER,
            highlightcolor=t.FOCUS_RING, borderwidth=0, relief="flat",
            font=t.FONT_BODY,
        )
        self.listbox.grid(row=0, column=0, sticky="nswe")
        sb = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        # --- column 2: preview ---
        mid = ttk.Frame(root)
        mid.grid(row=0, column=1, sticky="nswe", padx=(PAD, 0))
        mid.rowconfigure(0, weight=1)
        mid.columnconfigure(0, weight=1)
        self.preview = ttk.Label(mid, anchor="center", style="Preview.TLabel")
        self.preview.grid(row=0, column=0, sticky="nswe")

        nav = ttk.Frame(mid)
        nav.grid(row=1, column=0, sticky="we", pady=(PAD, 0))
        ttk.Button(nav, text="< PREV", style="Ghost.TButton",
                   command=lambda: self.step(-1)).pack(side="left")
        ttk.Button(nav, text="NEXT >", style="Ghost.TButton",
                   command=lambda: self.step(+1)).pack(side="left", padx=(PAD // 2, 0))
        ttk.Button(nav, text="TOGGLE INCLUDE", style="Ghost.TButton",
                   command=self.toggle_include).pack(side="left", padx=(PAD, 0))
        ttk.Label(
            nav, text="← → nav · I toggle · Ctrl+S save",
            style="Status.TLabel",
        ).pack(side="right")

        # --- column 3: editor ---
        right = ttk.Frame(root)
        right.grid(row=0, column=2, sticky="nswe", padx=(PAD, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        self.include_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            right, text="Include in training",
            variable=self.include_var,
            command=self._capture,
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(right, text="Caption:").grid(row=1, column=0, sticky="w", pady=(PAD, 2))
        self.caption_text = tk.Text(
            right, height=5, wrap="word", font=t.FONT_BODY,
            background=t.BG_INPUT, foreground=t.TEXT_PRIMARY,
            insertbackground=t.ACCENT_CYAN, relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=t.DIVIDER,
            highlightcolor=t.FOCUS_RING,
        )
        self.caption_text.grid(row=2, column=0, sticky="nswe")
        self.caption_text.bind("<FocusOut>", lambda _e: self._capture())

        chips = ttk.LabelFrame(right, text="QUICK TAGS", padding=PAD)
        chips.grid(row=3, column=0, sticky="we", pady=(PAD, 0))
        self.chips_frame = chips
        self._rebuild_chips([])

        info = ttk.LabelFrame(right, text="IMAGE INFO", padding=PAD)
        info.grid(row=4, column=0, sticky="we", pady=(PAD, 0))
        info.columnconfigure(0, weight=1)
        self.info_var = tk.StringVar(value="—")
        ttk.Label(info, textvariable=self.info_var, justify="left",
                  style="Mono.TLabel").grid(row=0, column=0, sticky="w")
        self.dupes_var = tk.StringVar(value="")
        ttk.Label(info, textvariable=self.dupes_var, style="Warn.TLabel",
                  justify="left").grid(row=1, column=0, sticky="w", pady=(2, 0))

        ttk.Label(right, text="Notes:").grid(row=5, column=0, sticky="w", pady=(PAD, 2))
        self.notes_entry = ttk.Entry(right)
        self.notes_entry.grid(row=6, column=0, sticky="we")

    def _build_grid(self, root: ttk.Frame) -> None:
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.grid = ThumbnailGrid(
            root, cols=5, cell_size=160,
            on_click=self._on_grid_click,
        )
        self.grid.grid(row=0, column=0, sticky="nswe")

    # ---- mode toggle ----

    def _toggle_mode(self) -> None:
        if self.mode == "detail":
            self.mode = "grid"
            self.mode_var.set("DETAIL VIEW")
            self.detail_frame.grid_forget()
            self.grid_frame.grid(row=1, column=0, sticky="nswe")
            self._populate_grid()
        else:
            self.mode = "detail"
            self.mode_var.set("GRID VIEW")
            self.grid_frame.grid_forget()
            self.detail_frame.grid(row=1, column=0, sticky="nswe")

    def _populate_grid(self) -> None:
        if not self.review or not self.gui.current_project:
            return
        project = self.gui.current_project
        items = []
        meta = {}
        for stem in self.order:
            items.append((stem, project.processed_dir / f"{stem}.png"))
            entry = self.review.entries[stem]
            meta[stem] = {
                "include": entry.include,
                "dupes": len(self.dupes_index.get(stem, [])),
            }
        self.grid.populate(items, metadata=meta)

    def _on_grid_click(self, stem: str) -> None:
        if stem in self.order:
            self._capture()
            self.idx = self.order.index(stem)
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(self.idx)
            self.listbox.see(self.idx)
            self._render()
            # switch back to detail view
            if self.mode == "grid":
                self._toggle_mode()

    # ---- external API called by TrainerGUI ----

    def on_tab_enter(self) -> None:
        """Called when user switches into this tab."""
        if self.gui.current_project is None:
            return
        if self.review is None:
            self.reload()
        else:
            self._refresh_listbox()
        self._bind_shortcuts(True)

    def on_tab_leave(self) -> None:
        self._bind_shortcuts(False)

    def _bind_shortcuts(self, enable: bool) -> None:
        keys = ("<Left>", "<Right>", "<i>", "<I>")
        if enable:
            self.gui.root.bind("<Left>", self._kbd(lambda: self.step(-1)))
            self.gui.root.bind("<Right>", self._kbd(lambda: self.step(+1)))
            self.gui.root.bind("<i>", self._kbd(self.toggle_include))
            self.gui.root.bind("<I>", self._kbd(self.toggle_include))
        else:
            for k in keys:
                try:
                    self.gui.root.unbind(k)
                except Exception:
                    pass

    def _kbd(self, action):
        def handler(_e=None):
            w = self.gui.root.focus_get()
            if isinstance(w, (tk.Text, tk.Entry, ttk.Entry, ttk.Combobox)):
                return
            action()
        return handler

    # ---- reload / render ----

    def reload(self) -> None:
        if not self.gui.current_project:
            messagebox.showerror("No project", "Create or open a project first.")
            return
        project = self.gui.current_project
        self.review = review_mod.load(project)
        self.order = sorted(self.review.entries.keys())
        if not self.order:
            self.counts_var.set("no processed images yet")
            self.preview.configure(image="", text="(run prep first)")
            self.idx = -1
            self.listbox.delete(0, "end")
            return

        # one-pass stats + hashes
        pngs = [project.processed_dir / f"{s}.png" for s in self.order]
        self.stats_index = {}
        hashes: list[tuple[str, int]] = []
        for p in pngs:
            st, h = insights.stats_and_hash(p)
            self.stats_index[p.stem] = st
            hashes.append((p.stem, h))
        self.dupes_index = {}
        for i in range(len(hashes)):
            si, hi = hashes[i]
            for j in range(i + 1, len(hashes)):
                sj, hj = hashes[j]
                d = insights.hamming(hi, hj)
                if d <= 6:
                    self.dupes_index.setdefault(si, []).append((sj, d))
                    self.dupes_index.setdefault(sj, []).append((si, d))

        self._refresh_listbox()
        self.idx = 0
        self._render()
        self.gui.refresh_step_status()

    def _refresh_listbox(self) -> None:
        self.listbox.delete(0, "end")
        assert self.review is not None
        for stem in self.order:
            entry = self.review.entries[stem]
            mark = "✓" if entry.include else "✗"
            self.listbox.insert("end", f"{mark} {stem}")
        counts = (
            f"{self.review.included_count()} in  ·  "
            f"{self.review.excluded_count()} out  ·  "
            f"{len(self.order)} total"
        )
        self.counts_var.set(counts)
        self._rebuild_chips(
            self.gui.current_project.prompt_chips if self.gui.current_project else []
        )

    def _rebuild_chips(self, chips: list) -> None:
        for w in self.chips_frame.winfo_children():
            w.destroy()
        if not chips:
            ttk.Label(self.chips_frame, text="(no chips configured)").pack(anchor="w")
            return
        per_row = 10
        row = None
        for i, chip in enumerate(chips):
            if i % per_row == 0:
                row = ttk.Frame(self.chips_frame)
                row.pack(fill="x")
            ttk.Button(
                row, text=chip, style="Ghost.TButton",
                command=lambda c=chip: self._append_chip(c),
            ).pack(side="left", padx=(0, 4), pady=2)

    def _on_list_select(self, _e) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        new_idx = sel[0]
        if new_idx == self.idx:
            return
        self._capture()
        self.idx = new_idx
        self._render()

    def step(self, delta: int) -> None:
        if self.review is None or not self.order:
            return
        self._capture()
        self.idx = max(0, min(len(self.order) - 1, self.idx + delta))
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(self.idx)
        self.listbox.see(self.idx)
        self._render()

    def _current_stem(self) -> Optional[str]:
        if 0 <= self.idx < len(self.order):
            return self.order[self.idx]
        return None

    def _render(self) -> None:
        stem = self._current_stem()
        if not stem or self.gui.current_project is None or self.review is None:
            return
        entry = self.review.entries[stem]
        png_path = self.gui.current_project.processed_dir / f"{stem}.png"
        self._show_preview(png_path)

        self.include_var.set(entry.include)
        self.caption_text.delete("1.0", "end")
        self.caption_text.insert("1.0", entry.caption)
        self.notes_entry.delete(0, "end")
        self.notes_entry.insert(0, entry.notes)

        stats = self.stats_index.get(stem, {})
        bits = [f"{stem}.png"]
        if stats:
            bits.append(f"{stats['width']}×{stats['height']}")
            bits.append(f"brightness {stats['brightness']}")
            bits.append(f"sharpness {stats['sharpness']}")
            warn = insights.resolution_warning(
                stats["width"], stats["height"], self.gui.current_project.resolution
            )
            if warn:
                bits.append(f"⚠ {warn}")
        self.info_var.set("  ·  ".join(bits))

        dupes = self.dupes_index.get(stem, [])
        if dupes:
            self.dupes_var.set(
                "near-duplicates: "
                + ", ".join(f"{s} (d={d})" for s, d in dupes[:5])
            )
        else:
            self.dupes_var.set("")

    def _show_preview(self, png_path: Path, max_side: int = 560) -> None:
        from PIL import Image, ImageTk
        try:
            img = Image.open(png_path).convert("RGB")
        except Exception as e:
            self.preview.configure(image="", text=f"(failed to open: {e})")
            return
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.photo_cache.append(photo)
        self.preview.configure(image=photo, text="")

    def _capture(self) -> None:
        stem = self._current_stem()
        if not stem or self.review is None:
            return
        entry = self.review.entries[stem]
        entry.include = bool(self.include_var.get())
        entry.caption = self.caption_text.get("1.0", "end").strip()
        entry.notes = self.notes_entry.get().strip()
        mark = "✓" if entry.include else "✗"
        self.listbox.delete(self.idx)
        self.listbox.insert(self.idx, f"{mark} {stem}")
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(self.idx)
        counts = (
            f"{self.review.included_count()} in  ·  "
            f"{self.review.excluded_count()} out  ·  "
            f"{len(self.order)} total"
        )
        self.counts_var.set(counts)

    def toggle_include(self) -> None:
        self.include_var.set(not self.include_var.get())
        self._capture()

    def _append_chip(self, chip: str) -> None:
        current = self.caption_text.get("1.0", "end").strip()
        new = review_mod.append_chip(current, chip)
        self.caption_text.delete("1.0", "end")
        self.caption_text.insert("1.0", new)
        self._capture()

    def save(self) -> None:
        if not self.gui.current_project or self.review is None:
            messagebox.showerror("Nothing to save", "Reload the Review tab first.")
            return
        self._capture()
        path = review_mod.save(self.gui.current_project, self.review)
        self.gui.status_var.set(f"review saved to {path.name}")
        self.gui.log_queue.put(f"[review saved: {path}]\n")
        self.gui.refresh_step_status()
