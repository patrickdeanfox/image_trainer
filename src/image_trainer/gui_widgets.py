"""Custom Tk widgets used across tabs.

All widgets defined here are pure Tk/ttk — no model or training dependencies.
They pull their palette from :mod:`gui_theme` so recolouring the whole GUI
happens in one place.

Contents:
- :class:`StatusDot` — 10px round indicator for pipeline-step completion.
- :class:`Sparkline` — tiny line chart (used for training loss).
- :class:`FolderField` — Entry + Browse + Open composite for path inputs.
- :class:`CollapsibleFrame` — header row with a caret that toggles the body.
- :class:`ThumbnailGrid` — scrollable grid of PIL images (Review tab).
"""

from __future__ import annotations

import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable, Iterable, Optional, Sequence

from . import gui_helpers, gui_theme


# ---------- StatusDot ----------

class StatusDot(tk.Canvas):
    """A 10px circle that communicates step state in the compact header.

    States: ``"pending"`` (muted), ``"active"`` (cyan), ``"done"`` (green),
    ``"warn"`` (gold). Re-colour via :meth:`set_state`.
    """

    SIZE = 12

    _STATE_COLORS = {
        "pending": None,   # filled at instance time from theme
        "active": None,
        "done": None,
        "warn": None,
    }

    def __init__(self, master: tk.Misc, state: str = "pending"):
        t = gui_theme.THEME
        super().__init__(
            master,
            width=self.SIZE,
            height=self.SIZE,
            highlightthickness=0,
            background=t.BG_PANEL,
        )
        self._state = state
        self._oval = self.create_oval(
            2, 2, self.SIZE - 2, self.SIZE - 2,
            fill=self._color_for(state), outline="",
        )

    def _color_for(self, state: str) -> str:
        t = gui_theme.THEME
        return {
            "pending": t.TEXT_MUTED,
            "active": t.ACCENT_CYAN,
            "done": t.ACCENT_GREEN,
            "warn": t.ACCENT_GOLD,
            "error": t.ACCENT_RED,
        }.get(state, t.TEXT_MUTED)

    def set_state(self, state: str) -> None:
        self._state = state
        self.itemconfigure(self._oval, fill=self._color_for(state))


# ---------- Sparkline ----------

class Sparkline(tk.Canvas):
    """Compact line chart rendered with Canvas primitives.

    Designed for the training-loss readout: call :meth:`push` on every new
    loss value, the widget keeps the last :attr:`maxlen` points and redraws.
    """

    def __init__(self, master: tk.Misc, *, width: int = 280, height: int = 52,
                 maxlen: int = 300):
        t = gui_theme.THEME
        super().__init__(
            master,
            width=width,
            height=height,
            highlightthickness=1,
            highlightbackground=t.DIVIDER,
            background=t.BG_INPUT,
        )
        # NB: don't store width on ``self._w`` — that's Tk's internal widget
        # command name; overwriting it breaks every subsequent Tcl call from
        # this widget with an "invalid command name" error. Same story for
        # ``self._h`` — avoid names that Tkinter uses internally.
        self._canvas_w = width
        self._canvas_h = height
        self._points: deque = deque(maxlen=maxlen)
        self._label = self.create_text(
            width - 4, 4,
            text="",
            anchor="ne",
            fill=t.TEXT_MUTED,
            font=gui_theme.THEME.FONT_MONO_SMALL,
        )

    def push(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            return
        self._points.append(float(value))
        self._redraw()

    def clear(self) -> None:
        self._points.clear()
        self._redraw()

    def _redraw(self) -> None:
        # Remove old line segments (keep the label text item).
        for item in self.find_withtag("line"):
            self.delete(item)
        if len(self._points) < 2:
            self.itemconfigure(self._label, text="")
            return

        t = gui_theme.THEME
        pad_x, pad_y = 6, 10
        w = self._canvas_w - pad_x * 2
        h = self._canvas_h - pad_y * 2

        vals = list(self._points)
        vmin = min(vals)
        vmax = max(vals)
        rng = vmax - vmin or 1.0

        step_x = w / max(1, len(vals) - 1)
        coords: list[float] = []
        for i, v in enumerate(vals):
            x = pad_x + i * step_x
            # y axis inverted: lower loss = higher pixel
            y = pad_y + (1 - (v - vmin) / rng) * h
            coords.extend([x, y])

        self.create_line(
            *coords,
            fill=t.ACCENT_AMBER,
            width=1.4,
            smooth=True,
            tags=("line",),
        )
        last = vals[-1]
        self.itemconfigure(
            self._label,
            text=f"loss={last:.4f}  min={vmin:.4f}  n={len(vals)}",
        )


# ---------- FolderField ----------

class FolderField(ttk.Frame):
    """Entry + Browse + Open composite for a directory-valued setting.

    Used everywhere a path is configurable. Keeps the Entry+buttons layout
    identical across tabs (Settings/Prep/Generate/etc.).
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        textvariable: tk.StringVar,
        browse_title: str = "Choose folder",
        file_mode: bool = False,
        filetypes: Optional[Sequence[tuple[str, str]]] = None,
    ):
        super().__init__(master)
        self._var = textvariable
        self._browse_title = browse_title
        self._file_mode = file_mode
        self._filetypes = list(filetypes or [("All files", "*.*")])

        self.columnconfigure(0, weight=1)
        self._entry = ttk.Entry(self, textvariable=textvariable)
        self._entry.grid(row=0, column=0, sticky="we")
        ttk.Button(self, text="BROWSE…", style="Ghost.TButton",
                   command=self._browse).grid(row=0, column=1, padx=(gui_theme.PAD // 2, 0))
        ttk.Button(self, text="OPEN", style="Ghost.TButton",
                   command=self._open).grid(row=0, column=2, padx=(gui_theme.PAD // 2, 0))

    def _browse(self) -> None:
        if self._file_mode:
            path = filedialog.askopenfilename(
                title=self._browse_title, filetypes=self._filetypes
            )
        else:
            path = filedialog.askdirectory(title=self._browse_title)
        if path:
            self._var.set(path)

    def _open(self) -> None:
        v = self._var.get().strip()
        if not v:
            return
        p = Path(v).expanduser()
        if self._file_mode:
            # Open containing folder; selecting a file is not cross-platform.
            if p.exists():
                gui_helpers.open_folder(p.parent)
            else:
                gui_helpers.open_folder(p.parent if p.parent.exists() else Path.home())
        else:
            gui_helpers.open_folder(p if p.exists() else Path.home())


# ---------- CollapsibleFrame ----------

class CollapsibleFrame(ttk.Frame):
    """Header row with a caret and label that toggles a body frame.

    The body is any ttk.Frame exposed via :attr:`body`. Use for the log
    pane so it doesn't always steal space.
    """

    def __init__(self, master: tk.Misc, *, text: str, start_open: bool = True):
        super().__init__(master)
        self._open = start_open

        header = ttk.Frame(self)
        header.pack(fill="x")

        self._caret_var = tk.StringVar(value="▾" if start_open else "▸")
        self._btn = ttk.Button(
            header,
            textvariable=self._caret_var,
            style="Ghost.TButton",
            width=2,
            command=self.toggle,
        )
        self._btn.pack(side="left")

        ttk.Label(header, text=text, style="SubHeader.TLabel",
                  background=gui_theme.THEME.BG_ROOT).pack(side="left", padx=(4, 0))

        self.body = ttk.Frame(self)
        if start_open:
            self.body.pack(fill="both", expand=True, pady=(4, 0))

    def toggle(self) -> None:
        self._open = not self._open
        if self._open:
            self.body.pack(fill="both", expand=True, pady=(4, 0))
            self._caret_var.set("▾")
        else:
            self.body.pack_forget()
            self._caret_var.set("▸")


# ---------- ThumbnailGrid ----------

class ThumbnailGrid(ttk.Frame):
    """Scrollable grid of thumbnails. Used by Review tab's grid-view toggle.

    Construct with a list of (stem, image_path) plus an optional metadata
    dict for include/exclude marks and near-dup counts. Click callback fires
    with the stem.
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        cols: int = 5,
        cell_size: int = 160,
        on_click: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(master, style="Panel.TFrame")
        self._cols = cols
        self._cell = cell_size
        self._on_click = on_click
        self._photo_refs: deque = deque()  # keep photo image refs alive

        t = gui_theme.THEME
        self._canvas = tk.Canvas(
            self, background=t.BG_PANEL, highlightthickness=0, borderwidth=0,
        )
        self._canvas.pack(side="left", fill="both", expand=True)

        self._vsb = ttk.Scrollbar(
            self, orient="vertical", command=self._canvas.yview,
            style="Vertical.TScrollbar",
        )
        self._vsb.pack(side="right", fill="y")
        self._canvas.configure(yscrollcommand=self._vsb.set)

        self._inner = ttk.Frame(self._canvas, style="Panel.TFrame")
        self._win = self._canvas.create_window(
            (0, 0), window=self._inner, anchor="nw",
        )

        self._inner.bind(
            "<Configure>",
            lambda _e: self._canvas.configure(
                scrollregion=self._canvas.bbox("all")
            ),
        )
        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfigure(self._win, width=e.width),
        )
        # Mouse wheel scroll — both conventions.
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind_all("<Button-4>", lambda _e: self._canvas.yview_scroll(-3, "units"))
        self._canvas.bind_all("<Button-5>", lambda _e: self._canvas.yview_scroll(3, "units"))

    def _on_mousewheel(self, event) -> None:
        # Windows/macOS: event.delta is +/- multiples of 120.
        delta = -1 if event.delta > 0 else 1
        self._canvas.yview_scroll(delta * 3, "units")

    def clear(self) -> None:
        for child in self._inner.winfo_children():
            child.destroy()
        self._photo_refs.clear()

    def populate(
        self,
        items: Iterable[tuple[str, Path]],
        *,
        metadata: Optional[dict[str, dict]] = None,
    ) -> int:
        """Render `items` as a grid. Returns the number of cells rendered.

        `metadata[stem]` may carry ``{"include": bool, "dupes": int}`` and
        the grid decorates each cell accordingly.
        """
        from PIL import Image, ImageTk  # local import so gui_widgets loads without PIL

        self.clear()
        meta = metadata or {}
        count = 0
        for idx, (stem, path) in enumerate(items):
            r, c = divmod(idx, self._cols)
            cell = ttk.Frame(self._inner, style="Elevated.TFrame", padding=4)
            cell.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            self._inner.columnconfigure(c, weight=1)

            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((self._cell, self._cell), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
            except Exception:
                continue
            self._photo_refs.append(photo)

            lbl = tk.Label(
                cell, image=photo, borderwidth=0,
                background=gui_theme.THEME.BG_ELEVATED,
                cursor="hand2",
            )
            lbl.pack()
            if self._on_click:
                lbl.bind("<Button-1>", lambda _e, s=stem: self._on_click(s))

            m = meta.get(stem, {})
            mark = "✓" if m.get("include", True) else "✗"
            mark_color = (
                gui_theme.THEME.ACCENT_GREEN if m.get("include", True)
                else gui_theme.THEME.ACCENT_RED
            )
            caption = f"{mark}  {stem}"
            if m.get("dupes"):
                caption += f"  · dup({m['dupes']})"
            cap_label = tk.Label(
                cell, text=caption,
                background=gui_theme.THEME.BG_ELEVATED,
                foreground=mark_color,
                font=gui_theme.THEME.FONT_MONO_SMALL,
            )
            cap_label.pack()
            count += 1
        return count
