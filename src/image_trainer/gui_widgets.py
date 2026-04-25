"""Custom Tk widgets used across tabs.

All widgets defined here are pure Tk/ttk — no model or training dependencies.
They pull their palette from :mod:`gui_theme` so recolouring the whole GUI
happens in one place.

Contents:
- :class:`StatusDot` — 10px round indicator for pipeline-step completion.
- :class:`Sparkline` — tiny line chart (used for training loss).
- :class:`FolderField` — Entry + Browse + Open composite for path inputs.
- :class:`CollapsibleFrame` — header row with a caret that toggles the body.
- :class:`ScrollableFrame` — vertically scrollable container with a managed
  inner frame + mouse-wheel scroll. Use this whenever a tab grows taller
  than the window.
- :class:`ThumbnailGrid` — scrollable grid of PIL images (Review tab).
- :class:`Tooltip` — hover popup with a one-sentence explanation.
- :func:`info_icon` — convenience builder that drops a ⓘ glyph + tooltip
  next to a label so technical fields can self-explain.
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
        ttk.Button(self, text="Browse…", style="Ghost.TButton",
                   command=self._browse).grid(row=0, column=1, padx=(gui_theme.PAD // 2, 0))
        ttk.Button(self, text="Open", style="Ghost.TButton",
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


# ---------- ScrollableFrame ----------

class ScrollableFrame(ttk.Frame):
    """Vertically scrollable container with a managed inner frame.

    Use when a tab's content can grow taller than the window and you want
    a regular vertical scroll instead of cramping or hiding fields. The
    pattern is the standard "Canvas + Scrollbar + inner Frame + bind
    <Configure>" recipe; this widget just packages it once so each tab
    isn't reinventing it.

    Usage:
        scroll = ScrollableFrame(parent_frame)
        scroll.pack(fill="both", expand=True)
        # Build your widgets inside scroll.body, treating it as a
        # regular ttk.Frame.
        ttk.Label(scroll.body, text="Hello").pack()

    Notes:
    - The inner frame's width tracks the canvas's width, so packed/grid
      children that expand horizontally stay full-width.
    - Mouse wheel scroll is bound globally on the canvas while the
      pointer is over it, mirroring native scroll feel on Linux/Windows.
    - We deliberately do NOT bind a horizontal scrollbar — for tab
      content, vertical scroll is what users expect; horizontal scroll
      hides content unpredictably.
    """

    def __init__(self, master: tk.Misc, *, panel_style: bool = False) -> None:
        super().__init__(master)
        t = gui_theme.THEME

        bg = t.BG_PANEL if panel_style else t.BG_ROOT

        # Explicit, modest minimum width/height. Without these the Canvas
        # asks the layout system for whatever its inner content needs,
        # which on a tab with a tall scrollable form (e.g. the Generate
        # tab) bubbles up to the surrounding ttk.Notebook and pushes the
        # telemetry pane off the bottom of the window. The canvas can
        # still expand beyond these via fill="both" expand=True; the
        # numbers are just the *minimum* it requests.
        # Bumped to 600 (was 200) — the Generate tab's form is taller
        # since the right sidebar was removed, and 200 px gave only
        # ~6 lines of visible content before scroll kicked in. 600 px
        # shows roughly the prompt body + 2-3 picker groups before
        # the user has to scroll.
        self._canvas = tk.Canvas(
            self,
            background=bg,
            highlightthickness=0,
            borderwidth=0,
            width=700,
            height=600,
        )
        self._canvas.pack(side="left", fill="both", expand=True)

        self._vsb = ttk.Scrollbar(
            self, orient="vertical", command=self._canvas.yview,
            style="Vertical.TScrollbar",
        )
        self._vsb.pack(side="right", fill="y")
        self._canvas.configure(yscrollcommand=self._vsb.set)

        # The frame the caller actually drops widgets into.
        self.body = ttk.Frame(self._canvas, style="Panel.TFrame" if panel_style else "TFrame")
        self._win = self._canvas.create_window(
            (0, 0), window=self.body, anchor="nw",
        )

        # Re-compute scrollregion whenever inner content reflows.
        self.body.bind("<Configure>", self._on_inner_configure)
        # Keep inner frame's width matched to the canvas so horizontal
        # expansion of children works as expected.
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        # Wheel scrolling is bound on enter / unbound on leave so multiple
        # ScrollableFrames in the same window don't fight for events.
        self._canvas.bind("<Enter>", self._bind_wheel)
        self._canvas.bind("<Leave>", self._unbind_wheel)

    # ---- internal handlers ----

    def _on_inner_configure(self, _e=None) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, e) -> None:
        # Match inner-frame width to the canvas; this is what makes packed
        # children with fill="x" actually fill the visible area.
        self._canvas.itemconfigure(self._win, width=e.width)

    def _bind_wheel(self, _e=None) -> None:
        # Windows / macOS deliver MouseWheel with .delta in multiples of 120.
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # X11 delivers scroll as Button-4 / Button-5.
        self._canvas.bind_all("<Button-4>", self._on_button4)
        self._canvas.bind_all("<Button-5>", self._on_button5)

    def _unbind_wheel(self, _e=None) -> None:
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                self._canvas.unbind_all(ev)
            except Exception:
                pass

    def _on_mousewheel(self, e) -> None:
        delta = -1 if e.delta > 0 else 1
        self._canvas.yview_scroll(delta * 3, "units")

    def _on_button4(self, _e=None) -> None:
        self._canvas.yview_scroll(-3, "units")

    def _on_button5(self, _e=None) -> None:
        self._canvas.yview_scroll(3, "units")

    # ---- convenience ----

    def scroll_to_top(self) -> None:
        self._canvas.yview_moveto(0)


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


# ---------- Tooltip + info icon ----------

class Tooltip:
    """A delayed-hover tooltip that pops a small parchment label near the
    cursor. Use via :meth:`bind` (any widget) or :func:`info_icon` (which
    bakes the glyph + tooltip into a tiny clickable label).

    Honors the active :data:`gui_theme.THEME` so it reads as part of the
    interface, not as a generic OS tooltip.
    """

    DELAY_MS = 350          # hover dwell before showing — short feels responsive
    OFFSET = (14, 18)       # cursor → tooltip nudge in pixels
    MAX_WIDTH = 340

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self._after_id: Optional[str] = None
        self._tip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<Motion>", self._on_motion, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    @classmethod
    def bind(cls, widget: tk.Widget, text: str) -> "Tooltip":
        """Convenience constructor — same as ``Tooltip(widget, text)`` but
        reads more naturally at call sites: ``Tooltip.bind(entry, "...")``."""
        return cls(widget, text)

    def _on_enter(self, _e=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.DELAY_MS, self._show)

    def _on_motion(self, _e=None) -> None:
        if self._tip is None:
            self._cancel()
            self._after_id = self.widget.after(self.DELAY_MS, self._show)

    def _on_leave(self, _e=None) -> None:
        self._cancel()
        self._hide()

    def _cancel(self) -> None:
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self._tip is not None:
            return
        t = gui_theme.THEME
        # Skip if the underlying widget has been destroyed (e.g. tab rebuilt).
        try:
            x = self.widget.winfo_pointerx() + self.OFFSET[0]
            y = self.widget.winfo_pointery() + self.OFFSET[1]
        except Exception:
            return

        tip = tk.Toplevel(self.widget)
        # overrideredirect removes window decorations; transient + topmost +
        # lift make sure the popup actually surfaces above the main window
        # on every platform we care about. Without these the popup can render
        # behind the main window on tiling WMs / older Tk builds and look
        # like the tooltip system "doesn't work."
        try:
            tip.wm_overrideredirect(True)
        except Exception:
            pass
        try:
            tip.wm_attributes("-topmost", True)
        except Exception:
            pass
        try:
            tip.transient(self.widget.winfo_toplevel())
        except Exception:
            pass
        tip.wm_geometry(f"+{x}+{y}")

        # A two-layer frame fakes a 1px outline since ttk.Frame can't easily
        # carry a coloured border on Tk's clam theme. Use a high-contrast
        # cream-on-burgundy border so the popup reads regardless of which
        # theme is active.
        outer = tk.Frame(tip, background=t.ACCENT_AMBER, padx=1, pady=1)
        outer.pack()
        inner = tk.Frame(outer, background=t.BG_ELEVATED, padx=10, pady=8)
        inner.pack()
        lbl = tk.Label(
            inner,
            text=self.text,
            background=t.BG_ELEVATED,
            foreground=t.TEXT_PRIMARY,
            font=t.FONT_BODY,
            wraplength=self.MAX_WIDTH,
            justify="left",
        )
        lbl.pack()
        try:
            tip.lift()
        except Exception:
            pass
        self._tip = tip

    def _hide(self) -> None:
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def info_icon(parent: tk.Widget, text: str) -> ttk.Label:
    """Drop a tiny ⓘ glyph that shows ``text`` on hover. Use right next to a
    label or input. Returns the label so the caller can grid/pack it.

    Usage:
        ttk.Label(row, text="Resolution:").pack(side="left")
        info_icon(row, "Square edge in pixels...").pack(side="left")
        ttk.Combobox(row, ...).pack(side="left")
    """
    t = gui_theme.THEME
    glyph = ttk.Label(
        parent, text="ⓘ", style="InfoIcon.TLabel",
    )
    # Configure the style once on first use; harmless to re-apply.
    style = ttk.Style()
    style.configure(
        "InfoIcon.TLabel",
        background=t.BG_ROOT,
        foreground=t.ACCENT_VIOLET,
        font=t.FONT_BODY,
        padding=(2, 0, 6, 0),
        cursor="question_arrow",
    )
    Tooltip.bind(glyph, text)
    return glyph
