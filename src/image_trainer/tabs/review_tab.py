"""04 · REVIEW tab — dual-column include/exclude lists with preview/editor.

Per-image ``{include, caption, notes, face_detected}`` is persisted to
``review.json`` at the project root via :mod:`pipeline.review`. The grid view
surfaces all thumbnails at once so near-duplicates and exposure outliers
jump out immediately; the detail view is a dual-pane "Included / Excluded"
workflow so the in/out decision is visible at a glance instead of needing
to read a ✓/✗ prefix.

A "Faces / No-face / All" filter narrows both columns — useful on datasets
where prep detected faces on only some images.

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
        # ``order`` is the full (unfiltered) list of stems in on-disk sort
        # order. It's the master list the filter + column split work against.
        self.order: list[str] = []
        # Currently-visible stems, split by include state. These are the
        # actual indices the two listboxes render. ``current_stem`` plus the
        # active column (``active_side``) reconstructs selection.
        self.included_visible: list[str] = []
        self.excluded_visible: list[str] = []
        self.active_side: str = "in"   # "in" | "out" — which list has focus.
        self.current_stem: Optional[str] = None
        self.photo_cache: deque = deque(maxlen=4)
        self.dupes_index: dict[str, list[tuple[str, int]]] = {}
        self.stats_index: dict[str, dict] = {}
        self.mode: str = "detail"   # or "grid"
        # Face filter: "all" | "face" | "noface" | "unknown"
        self.face_filter: str = "all"

    # ---- layout ----

    def build_ui(self, root: ttk.Frame) -> None:
        t = gui_theme.THEME
        PAD = gui_theme.PAD
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        top = ttk.Frame(root)
        top.grid(row=0, column=0, sticky="we", pady=(0, PAD))
        ttk.Label(top, text="Review · curate the set", style="Header.TLabel").pack(side="left")
        self.counts_var = tk.StringVar(value="—")
        ttk.Label(top, textvariable=self.counts_var, style="Status.TLabel").pack(side="right")

        # View toggle
        self.mode_var = tk.StringVar(value="Detail view")
        self.mode_btn = ttk.Button(
            top, textvariable=self.mode_var, style="Ghost.TButton",
            command=self._toggle_mode,
        )
        self.mode_btn.pack(side="right", padx=PAD)
        ttk.Button(top, text="Reload", style="Ghost.TButton",
                   command=self.reload).pack(side="right", padx=(0, PAD))
        ttk.Button(top, text="Save", style="Ghost.TButton",
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
        # Left pane (dual-column list + filter bar) gets a minimum width so the
        # two columns are always comfortable. Preview + editor split the rest.
        root.columnconfigure(0, weight=0, minsize=360)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=1)
        root.rowconfigure(0, weight=1)

        # --- column 1: filter bar + two side-by-side listboxes ---
        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nswe")
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)
        left.rowconfigure(2, weight=1)

        # Filter/sort bar spans both columns.
        filter_bar = ttk.Frame(left)
        filter_bar.grid(row=0, column=0, columnspan=2, sticky="we", pady=(0, PAD // 2))
        ttk.Label(filter_bar, text="filter", style="SubHeader.TLabel").pack(
            side="left", padx=(0, PAD // 2)
        )
        self.face_filter_var = tk.StringVar(value="all")
        for label, value in (
            ("All", "all"),
            ("Faces", "face"),
            ("No-face", "noface"),
            ("Unknown", "unknown"),
        ):
            ttk.Radiobutton(
                filter_bar, text=label, value=value,
                variable=self.face_filter_var,
                command=self._on_filter_change,
            ).pack(side="left", padx=(0, PAD // 2))

        # Column headers.
        self.in_header_var = tk.StringVar(value="Included (—)")
        self.out_header_var = tk.StringVar(value="Excluded (—)")
        ttk.Label(left, textvariable=self.in_header_var,
                  style="SubHeader.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, PAD // 2), pady=(0, 2)
        )
        ttk.Label(left, textvariable=self.out_header_var,
                  style="SubHeader.TLabel").grid(
            row=1, column=1, sticky="w", padx=(PAD // 2, 0), pady=(0, 2)
        )

        # Included column.
        in_frame = ttk.Frame(left)
        in_frame.grid(row=2, column=0, sticky="nswe", padx=(0, PAD // 2))
        in_frame.rowconfigure(0, weight=1)
        in_frame.columnconfigure(0, weight=1)
        self.listbox_in = tk.Listbox(
            in_frame, activestyle="none", exportselection=False,
            # selectmode="extended" enables click + shift-click range
            # selection AND ctrl-click for non-contiguous picking, both
            # via Tk's built-in handlers.
            selectmode="extended",
            background=t.BG_INPUT, foreground=t.TEXT_PRIMARY,
            selectbackground=t.ACCENT_GREEN, selectforeground=t.TEXT_ON_ACCENT,
            highlightthickness=1, highlightbackground=t.DIVIDER,
            highlightcolor=t.FOCUS_RING, borderwidth=0, relief="flat",
            font=t.FONT_BODY,
        )
        self.listbox_in.grid(row=0, column=0, sticky="nswe")
        sb_in = ttk.Scrollbar(in_frame, orient="vertical",
                              command=self.listbox_in.yview)
        sb_in.grid(row=0, column=1, sticky="ns")
        self.listbox_in.configure(yscrollcommand=sb_in.set)
        self.listbox_in.bind(
            "<<ListboxSelect>>",
            lambda _e: self._on_list_select("in"),
        )

        # Excluded column.
        out_frame = ttk.Frame(left)
        out_frame.grid(row=2, column=1, sticky="nswe", padx=(PAD // 2, 0))
        out_frame.rowconfigure(0, weight=1)
        out_frame.columnconfigure(0, weight=1)
        self.listbox_out = tk.Listbox(
            out_frame, activestyle="none", exportselection=False,
            selectmode="extended",
            background=t.BG_INPUT, foreground=t.TEXT_MUTED,
            selectbackground=t.ACCENT_RED, selectforeground=t.TEXT_ON_ACCENT,
            highlightthickness=1, highlightbackground=t.DIVIDER,
            highlightcolor=t.FOCUS_RING, borderwidth=0, relief="flat",
            font=t.FONT_BODY,
        )
        self.listbox_out.grid(row=0, column=0, sticky="nswe")
        sb_out = ttk.Scrollbar(out_frame, orient="vertical",
                               command=self.listbox_out.yview)
        sb_out.grid(row=0, column=1, sticky="ns")
        self.listbox_out.configure(yscrollcommand=sb_out.set)
        self.listbox_out.bind(
            "<<ListboxSelect>>",
            lambda _e: self._on_list_select("out"),
        )

        # --- column 2: preview ---
        mid = ttk.Frame(root)
        mid.grid(row=0, column=1, sticky="nswe", padx=(PAD, 0))
        mid.rowconfigure(0, weight=1)
        mid.columnconfigure(0, weight=1)
        self.preview = ttk.Label(mid, anchor="center", style="Preview.TLabel")
        self.preview.grid(row=0, column=0, sticky="nswe")

        nav = ttk.Frame(mid)
        nav.grid(row=1, column=0, sticky="we", pady=(PAD, 0))
        ttk.Button(nav, text="‹ Prev", style="Ghost.TButton",
                   command=lambda: self.step(-1)).pack(side="left")
        ttk.Button(nav, text="Next ›", style="Ghost.TButton",
                   command=lambda: self.step(+1)).pack(side="left", padx=(PAD // 2, 0))
        ttk.Button(nav, text="Toggle include", style="Ghost.TButton",
                   command=self.toggle_include).pack(side="left", padx=(PAD, 0))
        # Bulk-move buttons act on whatever's currently highlighted in
        # either listbox via shift-click range / ctrl-click set selection.
        ttk.Button(
            nav, text="Exclude selected ▶", style="Ghost.TButton",
            command=self.bulk_exclude,
        ).pack(side="left", padx=(PAD, 0))
        ttk.Button(
            nav, text="◀ Include selected", style="Ghost.TButton",
            command=self.bulk_include,
        ).pack(side="left", padx=(PAD // 2, 0))
        ttk.Label(
            nav, text="click + shift-click range · ctrl-click set",
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

        chips = ttk.LabelFrame(right, text="Quick tags", padding=PAD)
        chips.grid(row=3, column=0, sticky="we", pady=(PAD, 0))
        self.chips_frame = chips
        self._rebuild_chips([])

        info = ttk.LabelFrame(right, text="Image info", padding=PAD)
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
            self.mode_var.set("Detail view")
            self.detail_frame.grid_forget()
            self.grid_frame.grid(row=1, column=0, sticky="nswe")
            self._populate_grid()
        else:
            self.mode = "detail"
            self.mode_var.set("Grid view")
            self.grid_frame.grid_forget()
            self.detail_frame.grid(row=1, column=0, sticky="nswe")

    def _populate_grid(self) -> None:
        if not self.review or not self.gui.current_project:
            return
        project = self.gui.current_project
        items = []
        meta = {}
        for stem in self.order:
            entry = self.review.entries[stem]
            if not self._face_filter_matches(entry):
                continue
            items.append((stem, project.processed_dir / f"{stem}.png"))
            meta[stem] = {
                "include": entry.include,
                "dupes": len(self.dupes_index.get(stem, [])),
                "face_detected": entry.face_detected,
            }
        self.grid.populate(items, metadata=meta)

    def _on_grid_click(self, stem: str) -> None:
        if self.review is None or stem not in self.review.entries:
            return
        self._capture()
        entry = self.review.entries[stem]
        # Grid ignores the face filter (it always shows everything), so
        # clicking a stem that isn't currently in the filtered columns should
        # reset the filter to "all" — otherwise the selection would appear
        # to do nothing in the detail view.
        if not self._face_filter_matches(entry):
            self.face_filter = "all"
            self.face_filter_var.set("all")
            self._refresh_listbox()
        self.current_stem = stem
        self.active_side = "in" if entry.include else "out"
        self._reflect_selection()
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
            self.current_stem = None
            self.included_visible = []
            self.excluded_visible = []
            self.listbox_in.delete(0, "end")
            self.listbox_out.delete(0, "end")
            return

        # Stats + perceptual-hash work used to run inline here, opening every
        # PNG in the project before the UI rendered. With 186 1024x1024 PNGs
        # that's ~750 MB of decoded RGB and several seconds of stalled UI per
        # tab visit (which spins fans and feels broken). Render the columns
        # immediately with empty stats; a daemon thread fills them in and the
        # next selection / dup-warning render picks up the new data
        # automatically.
        self.stats_index = {}
        self.dupes_index = {}

        self._refresh_listbox()
        # Seed initial selection to the first visible stem (prefer the
        # Included column so the user starts where they'll spend most time).
        if self.included_visible:
            self.active_side = "in"
            self.current_stem = self.included_visible[0]
        elif self.excluded_visible:
            self.active_side = "out"
            self.current_stem = self.excluded_visible[0]
        else:
            self.current_stem = None
        self._reflect_selection()
        self._render()
        self.gui.refresh_step_status()
        # Kick off background insight build. Re-renders the active stem when
        # done so its stats panel + near-dup line populate without the user
        # touching anything.
        self._start_insights_thread(project)

    # ---- background stats / hashes ----

    def _start_insights_thread(self, project) -> None:
        """Compute per-image stats + perceptual hashes off the UI thread."""
        import threading

        # Snapshot the order so we don't race against a project switch.
        snap_order = list(self.order)
        snap_processed = project.processed_dir

        def _worker() -> None:
            try:
                local_stats: dict = {}
                hashes: list[tuple[str, int]] = []
                for stem in snap_order:
                    p = snap_processed / f"{stem}.png"
                    try:
                        st, h = insights.stats_and_hash(p)
                    except Exception:
                        continue
                    local_stats[stem] = st
                    hashes.append((stem, h))
                local_dupes: dict[str, list[tuple[str, int]]] = {}
                for i in range(len(hashes)):
                    si, hi = hashes[i]
                    for j in range(i + 1, len(hashes)):
                        sj, hj = hashes[j]
                        d = insights.hamming(hi, hj)
                        if d <= 6:
                            local_dupes.setdefault(si, []).append((sj, d))
                            local_dupes.setdefault(sj, []).append((si, d))
            except Exception as e:
                self.gui.log_queue.put(f"[review insights worker failed: {e}]\n")
                return

            # Hop back to the Tk thread to publish the results.
            def _publish() -> None:
                # Bail if the user switched projects mid-flight.
                if self.gui.current_project is None:
                    return
                if self.order != snap_order:
                    return
                self.stats_index = local_stats
                self.dupes_index = local_dupes
                # Re-render so the active selection gets the new data.
                self._render()

            try:
                self.gui.root.after(0, _publish)
            except Exception:
                # Tk may be torn down — ignore.
                pass

        threading.Thread(target=_worker, daemon=True).start()

    # ---- filter / columns ----

    def _face_filter_matches(self, entry: review_mod.ReviewEntry) -> bool:
        f = self.face_filter
        if f == "all":
            return True
        if f == "face":
            return entry.face_detected is True
        if f == "noface":
            return entry.face_detected is False
        if f == "unknown":
            return entry.face_detected is None
        return True

    def _on_filter_change(self) -> None:
        self.face_filter = self.face_filter_var.get()
        if self.review is None:
            return
        self._capture()
        self._refresh_listbox()
        # Keep current stem if still visible; otherwise fall back to the
        # first visible stem on either side so the preview never goes blank.
        if self.current_stem and (
            self.current_stem in self.included_visible
            or self.current_stem in self.excluded_visible
        ):
            pass
        elif self.included_visible:
            self.active_side = "in"
            self.current_stem = self.included_visible[0]
        elif self.excluded_visible:
            self.active_side = "out"
            self.current_stem = self.excluded_visible[0]
        else:
            self.current_stem = None
        self._reflect_selection()
        self._render()

    def _refresh_listbox(self) -> None:
        assert self.review is not None
        self.listbox_in.delete(0, "end")
        self.listbox_out.delete(0, "end")
        self.included_visible = []
        self.excluded_visible = []
        for stem in self.order:
            entry = self.review.entries[stem]
            if not self._face_filter_matches(entry):
                continue
            face_mark = self._face_mark(entry.face_detected)
            label = f"{face_mark} {stem}" if face_mark else stem
            if entry.include:
                self.listbox_in.insert("end", label)
                self.included_visible.append(stem)
            else:
                self.listbox_out.insert("end", label)
                self.excluded_visible.append(stem)
        self.in_header_var.set(f"Included ({len(self.included_visible)})")
        self.out_header_var.set(f"Excluded ({len(self.excluded_visible)})")
        counts = (
            f"{self.review.included_count()} in  ·  "
            f"{self.review.excluded_count()} out  ·  "
            f"{self.review.face_count()} faces  ·  "
            f"{self.review.non_face_count()} no-face  ·  "
            f"{len(self.order)} total"
        )
        self.counts_var.set(counts)
        self._rebuild_chips(
            self.gui.current_project.prompt_chips if self.gui.current_project else []
        )

    @staticmethod
    def _face_mark(face_detected: Optional[bool]) -> str:
        if face_detected is True:
            return "☺"
        if face_detected is False:
            return "·"
        return " "

    def _reflect_selection(self) -> None:
        """Mirror ``current_stem`` + ``active_side`` into the listbox widgets."""
        self.listbox_in.selection_clear(0, "end")
        self.listbox_out.selection_clear(0, "end")
        if not self.current_stem:
            return
        if self.active_side == "in" and self.current_stem in self.included_visible:
            i = self.included_visible.index(self.current_stem)
            self.listbox_in.selection_set(i)
            self.listbox_in.see(i)
        elif self.active_side == "out" and self.current_stem in self.excluded_visible:
            i = self.excluded_visible.index(self.current_stem)
            self.listbox_out.selection_set(i)
            self.listbox_out.see(i)

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

    def _on_list_select(self, side: str) -> None:
        lb = self.listbox_in if side == "in" else self.listbox_out
        visible = self.included_visible if side == "in" else self.excluded_visible
        sel = lb.curselection()
        if not sel:
            return
        new_idx = sel[0]
        if not (0 <= new_idx < len(visible)):
            return
        new_stem = visible[new_idx]
        if new_stem == self.current_stem and side == self.active_side:
            return
        self._capture()
        self.current_stem = new_stem
        self.active_side = side
        # Clear the other column's selection so it's visually obvious which
        # item is the active one.
        other = self.listbox_out if side == "in" else self.listbox_in
        other.selection_clear(0, "end")
        self._render()

    def step(self, delta: int) -> None:
        """Walk forward/back through the active column. When we fall off the
        end of one column, hop to the neighbour column if it has items — this
        makes ←/→ feel like a single unified list even though we render two.
        """
        if self.review is None or not self.order:
            return
        self._capture()
        visible = (
            self.included_visible if self.active_side == "in"
            else self.excluded_visible
        )
        if not visible:
            # Current column is empty — jump to the other one.
            other = (
                self.excluded_visible if self.active_side == "in"
                else self.included_visible
            )
            if not other:
                return
            self.active_side = "out" if self.active_side == "in" else "in"
            self.current_stem = other[0]
            self._reflect_selection()
            self._render()
            return
        try:
            idx = visible.index(self.current_stem) if self.current_stem else 0
        except ValueError:
            idx = 0
        new_idx = idx + delta
        if 0 <= new_idx < len(visible):
            self.current_stem = visible[new_idx]
        else:
            # Crossed a column edge — hop to the neighbour.
            other_side = "out" if self.active_side == "in" else "in"
            other_visible = (
                self.excluded_visible if other_side == "out"
                else self.included_visible
            )
            if other_visible:
                self.active_side = other_side
                # When walking forward, land on the first item in the other
                # column; when walking backward, land on the last.
                self.current_stem = (
                    other_visible[0] if delta > 0 else other_visible[-1]
                )
            else:
                # Other column is empty — stay at the edge.
                self.current_stem = visible[max(0, min(len(visible) - 1, new_idx))]
        self._reflect_selection()
        self._render()

    def _current_stem(self) -> Optional[str]:
        return self.current_stem

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
        """Write the editor widgets back into ``review.entries``. If the
        include state flipped, move the stem across columns and keep the
        selection sitting on the item that just moved."""
        stem = self._current_stem()
        if not stem or self.review is None:
            return
        entry = self.review.entries[stem]
        was_included = entry.include
        entry.include = bool(self.include_var.get())
        entry.caption = self.caption_text.get("1.0", "end").strip()
        entry.notes = self.notes_entry.get().strip()

        if was_included != entry.include:
            # State flipped — rebuild the columns so the stem jumps sides.
            # We keep ``current_stem`` pointing at the same image; the user's
            # eye tracks it across the split.
            self.active_side = "in" if entry.include else "out"
            self._refresh_listbox()
            self._reflect_selection()
        else:
            # State unchanged — cheap in-place update of just the visible
            # label on the active side.
            self._update_visible_label(stem)
            self._refresh_counts_only()

    def _update_visible_label(self, stem: str) -> None:
        assert self.review is not None
        entry = self.review.entries[stem]
        face_mark = self._face_mark(entry.face_detected)
        label = f"{face_mark} {stem}" if face_mark else stem
        if entry.include and stem in self.included_visible:
            i = self.included_visible.index(stem)
            self.listbox_in.delete(i)
            self.listbox_in.insert(i, label)
            self.listbox_in.selection_clear(0, "end")
            self.listbox_in.selection_set(i)
        elif not entry.include and stem in self.excluded_visible:
            i = self.excluded_visible.index(stem)
            self.listbox_out.delete(i)
            self.listbox_out.insert(i, label)
            self.listbox_out.selection_clear(0, "end")
            self.listbox_out.selection_set(i)

    def _refresh_counts_only(self) -> None:
        assert self.review is not None
        self.in_header_var.set(f"Included ({len(self.included_visible)})")
        self.out_header_var.set(f"Excluded ({len(self.excluded_visible)})")
        counts = (
            f"{self.review.included_count()} in  ·  "
            f"{self.review.excluded_count()} out  ·  "
            f"{self.review.face_count()} faces  ·  "
            f"{self.review.non_face_count()} no-face  ·  "
            f"{len(self.order)} total"
        )
        self.counts_var.set(counts)

    def toggle_include(self) -> None:
        self.include_var.set(not self.include_var.get())
        self._capture()

    # ---- bulk select / move ----

    def _selected_stems(self, side: str) -> list[str]:
        """Return the stems currently highlighted in the given column."""
        lb = self.listbox_in if side == "in" else self.listbox_out
        visible = self.included_visible if side == "in" else self.excluded_visible
        out: list[str] = []
        for idx in lb.curselection():
            if 0 <= idx < len(visible):
                out.append(visible[idx])
        return out

    def bulk_exclude(self) -> None:
        """Move every selected stem in the Included column → Excluded.

        No-op if nothing is selected on the included side. Saves the user's
        current edits to the visible stem first so the in-flight caption /
        notes don't get lost when the row jumps columns.
        """
        if self.review is None:
            return
        # Persist anything the user typed in the editor before the rebuild.
        self._capture()
        stems = self._selected_stems("in")
        if not stems:
            messagebox.showinfo(
                "Nothing selected",
                "Click a stem in the Included column (shift-click for a "
                "range, ctrl-click to add) before pressing 'Exclude "
                "selected ▶'.",
            )
            return
        for stem in stems:
            entry = self.review.entries.get(stem)
            if entry is not None:
                entry.include = False
        # Track the last-moved stem so the cursor lands somewhere sensible.
        last = stems[-1]
        self.current_stem = last
        self.active_side = "out"
        self._refresh_listbox()
        self._reflect_selection()
        self._render()
        self.gui.status_var.set(
            f"Excluded {len(stems)} image(s)"
        )

    def bulk_include(self) -> None:
        """Mirror of :meth:`bulk_exclude` for the Excluded → Included direction."""
        if self.review is None:
            return
        self._capture()
        stems = self._selected_stems("out")
        if not stems:
            messagebox.showinfo(
                "Nothing selected",
                "Click a stem in the Excluded column (shift-click for a "
                "range, ctrl-click to add) before pressing '◀ Include "
                "selected'.",
            )
            return
        for stem in stems:
            entry = self.review.entries.get(stem)
            if entry is not None:
                entry.include = True
        last = stems[-1]
        self.current_stem = last
        self.active_side = "in"
        self._refresh_listbox()
        self._reflect_selection()
        self._render()
        self.gui.status_var.set(
            f"Included {len(stems)} image(s)"
        )

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
