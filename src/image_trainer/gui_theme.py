"""Theme, font resolution, and ttk styling for the Tkinter GUI.

The GUI has one aesthetic POV — LCARS-inspired obsidian/glass — expressed
through a single source of truth (`Theme`) and applied via `apply_style`.
Keep ad-hoc hex strings OUT of the rest of the GUI; reach into `THEME`
instead.

Two things live here that used to be inline in `gui.py`:

1. :class:`Theme` — palette, type scale, spacing. Replaces the previous
   :class:`_Theme` with slightly better contrast (AA) and a new
   ``FOCUS_RING`` token. :meth:`Theme.dark` is a factory that returns the
   current palette; :meth:`Theme.light` is a stub for a future light mode
   (no widget actually switches theme at runtime yet).
2. :func:`apply_style` — configures every ttk widget class the GUI uses
   plus the raw-tk widgets that Tk can't reach via styles.

Font resolution is platform-aware: we look up :attr:`Theme.DISPLAY_CANDIDATES`
and :attr:`Theme.MONO_CANDIDATES` against ``tkFont.families()`` at startup
and pick the first installed face. No more silent fallback to generic sans.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, field
from tkinter import font as tkfont
from tkinter import ttk
from typing import Tuple


#: Uniform widget padding in pixels. Tweak here to rescale the whole layout.
PAD = 8


# ---------- font resolution ----------

#: Ordered candidates for the primary display typeface. We prefer a
#: condensed/geometric sans for the LCARS feel; fall back to classic
#: interface sans only if nothing better is on this machine.
DISPLAY_CANDIDATES: Tuple[str, ...] = (
    "Eurostile Extended",
    "Bahnschrift",
    "Helvetica Neue",
    "HelveticaNeue",
    "SF Pro Display",
    "Segoe UI",
    "Ubuntu",
    "DejaVu Sans",
    "Helvetica",
    "Arial",
)

#: Ordered candidates for the monospace typeface used in the log pane and
#: numeric readouts (loss values, step counters, VRAM).
MONO_CANDIDATES: Tuple[str, ...] = (
    "JetBrains Mono",
    "Fira Code",
    "Source Code Pro",
    "SF Mono",
    "Menlo",
    "Consolas",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "Courier New",
)


def _resolve_face(candidates: Tuple[str, ...], fallback: str) -> str:
    """Return the first candidate installed on this system, else ``fallback``.

    Must be called after a :class:`tk.Tk` root has been created so
    :func:`tkfont.families` can enumerate. That's why the GUI calls
    :func:`apply_style` *after* instantiating the root.
    """
    try:
        installed = set(tkfont.families())
    except Exception:
        installed = set()
    for name in candidates:
        if name in installed:
            return name
    return fallback


# ---------- theme ----------

@dataclass(frozen=True)
class Theme:
    """Palette + type scale + spacing. One instance per runtime.

    Surfaces are layered near-black tiers (root < panel < elevated < input)
    so depth is communicated through tonal shift rather than borders.
    Accents are amber (dominant), cyan (secondary), violet (tertiary),
    gold (caution), red (danger).
    """

    # --- surfaces ---
    BG_ROOT: str = "#05070D"
    BG_PANEL: str = "#0B1020"
    BG_ELEVATED: str = "#121A33"
    BG_INPUT: str = "#0E1526"
    BG_HOVER: str = "#1A2340"
    BG_PRESSED: str = "#22305A"

    # --- accents ---
    ACCENT_AMBER: str = "#FFB74D"
    ACCENT_CYAN: str = "#6FD6FF"
    ACCENT_VIOLET: str = "#B79CFF"
    ACCENT_RED: str = "#FF6B6B"
    ACCENT_GOLD: str = "#F2C14E"
    ACCENT_GREEN: str = "#8BE9A7"  # used for completed-step dots

    # --- structural strokes ---
    DIVIDER: str = "#1F2A47"
    BORDER_SOFT: str = "#2A3656"
    FOCUS_RING: str = "#FFB74D"  # new in this refactor

    # --- text ---
    TEXT_PRIMARY: str = "#E8EEF8"
    TEXT_SECONDARY: str = "#C4CEE3"   # bumped from #B7C0D8 (AA safe on BG_ROOT)
    TEXT_MUTED: str = "#8A94B2"        # bumped from #6B7591 for AA contrast
    TEXT_ON_ACCENT: str = "#0A0E1A"

    # --- fonts (concrete faces resolved via Theme.configure_fonts) ---
    _display_face: str = field(default="Helvetica")
    _mono_face: str = field(default="Courier New")

    # font tuples — derived lazily via properties so they can be rebuilt
    # after face resolution
    @property
    def FONT_BODY(self) -> Tuple:
        return (self._display_face, 10)

    @property
    def FONT_HEADER(self) -> Tuple:
        return (self._display_face, 11, "bold")

    @property
    def FONT_LCARS(self) -> Tuple:
        return (self._display_face, 10, "bold")

    @property
    def FONT_DISPLAY(self) -> Tuple:
        return (self._display_face, 16, "bold")

    @property
    def FONT_TITLE(self) -> Tuple:
        return (self._display_face, 20, "bold")

    @property
    def FONT_MONO(self) -> Tuple:
        return (self._mono_face, 10)

    @property
    def FONT_MONO_SMALL(self) -> Tuple:
        return (self._mono_face, 9)

    # --- factories ---

    @classmethod
    def dark(cls) -> "Theme":
        """The canonical LCARS dark palette. Current default."""
        return cls()

    @classmethod
    def light(cls) -> "Theme":
        """Stub for a future light mode. Not wired to any toggle yet.

        Returning an honest light palette here lets callers write
        ``Theme.light()`` without crashing, and makes future wiring of a
        toggle a drop-in change in one place.
        """
        return cls(
            BG_ROOT="#F5F2EC",
            BG_PANEL="#EAE4D8",
            BG_ELEVATED="#D8CFBE",
            BG_INPUT="#FFFFFF",
            BG_HOVER="#E0D6C0",
            BG_PRESSED="#C7BBA0",
            DIVIDER="#B9AF98",
            BORDER_SOFT="#9C927B",
            FOCUS_RING="#C97B1A",
            TEXT_PRIMARY="#14110B",
            TEXT_SECONDARY="#3C3627",
            TEXT_MUTED="#6A6147",
            TEXT_ON_ACCENT="#0A0E1A",
            ACCENT_AMBER="#C97B1A",
            ACCENT_CYAN="#2E7BA4",
            ACCENT_VIOLET="#6E4BB0",
            ACCENT_RED="#B84848",
            ACCENT_GOLD="#A98221",
            ACCENT_GREEN="#3B8F55",
        )

    def with_fonts(self, display_face: str, mono_face: str) -> "Theme":
        """Return a copy of this theme with resolved concrete font faces."""
        # dataclass is frozen; use dataclasses.replace-style copy via __class__.
        # The private fields on a frozen dataclass require object.__setattr__
        # on a fresh instance.
        new = self.__class__(
            BG_ROOT=self.BG_ROOT,
            BG_PANEL=self.BG_PANEL,
            BG_ELEVATED=self.BG_ELEVATED,
            BG_INPUT=self.BG_INPUT,
            BG_HOVER=self.BG_HOVER,
            BG_PRESSED=self.BG_PRESSED,
            ACCENT_AMBER=self.ACCENT_AMBER,
            ACCENT_CYAN=self.ACCENT_CYAN,
            ACCENT_VIOLET=self.ACCENT_VIOLET,
            ACCENT_RED=self.ACCENT_RED,
            ACCENT_GOLD=self.ACCENT_GOLD,
            ACCENT_GREEN=self.ACCENT_GREEN,
            DIVIDER=self.DIVIDER,
            BORDER_SOFT=self.BORDER_SOFT,
            FOCUS_RING=self.FOCUS_RING,
            TEXT_PRIMARY=self.TEXT_PRIMARY,
            TEXT_SECONDARY=self.TEXT_SECONDARY,
            TEXT_MUTED=self.TEXT_MUTED,
            TEXT_ON_ACCENT=self.TEXT_ON_ACCENT,
            _display_face=display_face,
            _mono_face=mono_face,
        )
        return new


#: Live theme instance. Populated by :func:`apply_style`; safe to read after
#: that call. Modules that need palette access import this name.
THEME: Theme = Theme.dark()


# ---------- style application ----------

def apply_style(root: tk.Tk) -> Theme:
    """Resolve fonts, apply the LCARS visual language, return the live theme.

    Must be called *after* the :class:`tk.Tk` root is created so we can
    inspect installed font families.
    """
    global THEME

    display_face = _resolve_face(DISPLAY_CANDIDATES, fallback="Helvetica")
    mono_face = _resolve_face(MONO_CANDIDATES, fallback="Courier New")
    THEME = Theme.dark().with_fonts(display_face, mono_face)
    t = THEME

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    root.configure(background=t.BG_ROOT)

    # ---- raw-tk defaults ----
    root.option_add("*Font", t.FONT_BODY)
    root.option_add("*background", t.BG_ROOT)
    root.option_add("*foreground", t.TEXT_PRIMARY)

    for w in ("Text", "Listbox", "ScrolledText"):
        root.option_add(f"*{w}.background", t.BG_INPUT)
        root.option_add(f"*{w}.foreground", t.TEXT_PRIMARY)
        root.option_add(f"*{w}.borderWidth", 0)
        root.option_add(f"*{w}.relief", "flat")
        root.option_add(f"*{w}.highlightThickness", 1)
        root.option_add(f"*{w}.highlightBackground", t.DIVIDER)
        root.option_add(f"*{w}.highlightColor", t.FOCUS_RING)
        root.option_add(f"*{w}.selectBackground", t.ACCENT_AMBER)
        root.option_add(f"*{w}.selectForeground", t.TEXT_ON_ACCENT)
    root.option_add("*Text.insertBackground", t.ACCENT_CYAN)
    root.option_add("*ScrolledText.insertBackground", t.ACCENT_CYAN)
    root.option_add("*Listbox.activeStyle", "none")

    root.option_add("*Menu.background", t.BG_ELEVATED)
    root.option_add("*Menu.foreground", t.TEXT_PRIMARY)
    root.option_add("*Menu.activeBackground", t.ACCENT_AMBER)
    root.option_add("*Menu.activeForeground", t.TEXT_ON_ACCENT)
    root.option_add("*Menu.borderWidth", 0)

    # ---- frames ----
    style.configure("TFrame", background=t.BG_ROOT)
    style.configure("Panel.TFrame", background=t.BG_PANEL)
    style.configure("Elevated.TFrame", background=t.BG_ELEVATED)
    style.configure("Input.TFrame", background=t.BG_INPUT)

    # ---- labels ----
    style.configure(
        "TLabel",
        background=t.BG_ROOT,
        foreground=t.TEXT_PRIMARY,
        font=t.FONT_BODY,
        padding=(2, 2),
    )
    style.configure("Panel.TLabel", background=t.BG_PANEL, foreground=t.TEXT_PRIMARY)
    style.configure("Elevated.TLabel", background=t.BG_ELEVATED, foreground=t.TEXT_PRIMARY)
    style.configure(
        "Header.TLabel",
        background=t.BG_ROOT,
        foreground=t.ACCENT_AMBER,
        font=t.FONT_HEADER,
    )
    style.configure(
        "Display.TLabel",
        background=t.BG_PANEL,
        foreground=t.ACCENT_AMBER,
        font=t.FONT_DISPLAY,
    )
    style.configure(
        "Title.TLabel",
        background=t.BG_PANEL,
        foreground=t.ACCENT_AMBER,
        font=t.FONT_TITLE,
    )
    style.configure(
        "SubHeader.TLabel",
        background=t.BG_PANEL,
        foreground=t.ACCENT_CYAN,
        font=t.FONT_LCARS,
    )
    style.configure(
        "Status.TLabel",
        background=t.BG_ROOT,
        foreground=t.TEXT_MUTED,
        font=t.FONT_BODY,
    )
    style.configure(
        "Muted.TLabel",
        background=t.BG_ROOT,
        foreground=t.TEXT_SECONDARY,
        font=t.FONT_BODY,
    )
    style.configure(
        "Warn.TLabel",
        background=t.BG_ROOT,
        foreground=t.ACCENT_GOLD,
        font=t.FONT_BODY,
    )
    style.configure(
        "Mono.TLabel",
        background=t.BG_ROOT,
        foreground=t.TEXT_SECONDARY,
        font=t.FONT_MONO,
    )
    style.configure(
        "MonoHot.TLabel",
        background=t.BG_ROOT,
        foreground=t.ACCENT_AMBER,
        font=t.FONT_MONO,
    )
    style.configure(
        "Preview.TLabel",
        background=t.BG_INPUT,
        foreground=t.TEXT_MUTED,
        borderwidth=1,
        relief="flat",
    )

    # ---- buttons ----
    style.configure(
        "TButton",
        background=t.BG_ELEVATED,
        foreground=t.ACCENT_AMBER,
        font=t.FONT_LCARS,
        padding=(14, 6),
        borderwidth=0,
        relief="flat",
        focuscolor=t.BG_ELEVATED,
    )
    style.map(
        "TButton",
        background=[
            ("pressed", t.BG_PRESSED),
            ("active", t.BG_HOVER),
            ("disabled", t.BG_PANEL),
        ],
        foreground=[
            ("disabled", t.TEXT_MUTED),
            ("pressed", t.TEXT_PRIMARY),
            ("active", t.ACCENT_CYAN),
        ],
    )
    style.configure(
        "Primary.TButton",
        background=t.ACCENT_AMBER,
        foreground=t.TEXT_ON_ACCENT,
        font=t.FONT_LCARS,
        padding=(18, 7),
        borderwidth=0,
        relief="flat",
        focuscolor=t.ACCENT_AMBER,
    )
    style.map(
        "Primary.TButton",
        background=[
            ("pressed", t.ACCENT_VIOLET),
            ("active", t.ACCENT_CYAN),
            ("disabled", t.BG_PANEL),
        ],
        foreground=[
            ("disabled", t.TEXT_MUTED),
            ("active", t.TEXT_ON_ACCENT),
            ("pressed", t.TEXT_ON_ACCENT),
        ],
    )
    style.configure(
        "Caution.TButton",
        background=t.BG_ELEVATED,
        foreground=t.ACCENT_RED,
        font=t.FONT_LCARS,
        padding=(14, 6),
        borderwidth=0,
        relief="flat",
    )
    style.map(
        "Caution.TButton",
        background=[("active", t.BG_HOVER), ("pressed", t.BG_PRESSED)],
        foreground=[("active", t.ACCENT_AMBER)],
    )
    # Small Ghost button for inline "Open" / "Reveal" actions next to fields.
    style.configure(
        "Ghost.TButton",
        background=t.BG_ROOT,
        foreground=t.ACCENT_CYAN,
        font=t.FONT_LCARS,
        padding=(8, 4),
        borderwidth=0,
        relief="flat",
    )
    style.map(
        "Ghost.TButton",
        background=[("active", t.BG_HOVER), ("pressed", t.BG_PRESSED)],
        foreground=[("active", t.ACCENT_AMBER)],
    )

    # ---- entries / comboboxes ----
    style.configure(
        "TEntry",
        fieldbackground=t.BG_INPUT,
        foreground=t.TEXT_PRIMARY,
        insertcolor=t.ACCENT_CYAN,
        bordercolor=t.DIVIDER,
        lightcolor=t.DIVIDER,
        darkcolor=t.DIVIDER,
        borderwidth=1,
        relief="flat",
        padding=(8, 5),
    )
    style.map(
        "TEntry",
        bordercolor=[("focus", t.FOCUS_RING)],
        lightcolor=[("focus", t.FOCUS_RING)],
        darkcolor=[("focus", t.FOCUS_RING)],
    )

    style.configure(
        "TCombobox",
        fieldbackground=t.BG_INPUT,
        background=t.BG_ELEVATED,
        foreground=t.TEXT_PRIMARY,
        arrowcolor=t.ACCENT_AMBER,
        bordercolor=t.DIVIDER,
        lightcolor=t.DIVIDER,
        darkcolor=t.DIVIDER,
        borderwidth=1,
        relief="flat",
        padding=(6, 4),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", t.BG_INPUT)],
        foreground=[("readonly", t.TEXT_PRIMARY), ("disabled", t.TEXT_MUTED)],
        bordercolor=[("focus", t.FOCUS_RING)],
        arrowcolor=[("active", t.ACCENT_CYAN)],
    )

    # ---- checkbutton ----
    style.configure(
        "TCheckbutton",
        background=t.BG_ROOT,
        foreground=t.TEXT_PRIMARY,
        indicatorbackground=t.BG_INPUT,
        indicatorforeground=t.ACCENT_AMBER,
        focuscolor=t.BG_ROOT,
        padding=(4, 2),
    )
    style.map(
        "TCheckbutton",
        background=[("active", t.BG_ROOT)],
        foreground=[("active", t.ACCENT_AMBER), ("disabled", t.TEXT_MUTED)],
        indicatorcolor=[
            ("selected", t.ACCENT_AMBER),
            ("!selected", t.BG_INPUT),
        ],
    )

    # ---- labelframe ----
    style.configure(
        "TLabelframe",
        background=t.BG_ROOT,
        bordercolor=t.DIVIDER,
        lightcolor=t.DIVIDER,
        darkcolor=t.DIVIDER,
        borderwidth=1,
        relief="flat",
        padding=PAD,
    )
    style.configure(
        "TLabelframe.Label",
        background=t.BG_ROOT,
        foreground=t.ACCENT_CYAN,
        font=t.FONT_LCARS,
    )

    # ---- notebook ----
    style.configure(
        "TNotebook",
        background=t.BG_ROOT,
        bordercolor=t.BG_ROOT,
        lightcolor=t.BG_ROOT,
        darkcolor=t.BG_ROOT,
        borderwidth=0,
        tabmargins=(0, 4, 0, 0),
    )
    style.configure(
        "TNotebook.Tab",
        background=t.BG_PANEL,
        foreground=t.TEXT_SECONDARY,
        font=t.FONT_LCARS,
        padding=(18, 9),
        borderwidth=0,
        focuscolor=t.BG_PANEL,
    )
    style.map(
        "TNotebook.Tab",
        background=[
            ("selected", t.BG_ELEVATED),
            ("active", t.BG_HOVER),
        ],
        foreground=[
            ("selected", t.ACCENT_AMBER),
            ("active", t.ACCENT_CYAN),
        ],
    )

    # ---- progressbar ----
    style.configure(
        "Trainer.Horizontal.TProgressbar",
        troughcolor=t.BG_INPUT,
        background=t.ACCENT_AMBER,
        bordercolor=t.BG_INPUT,
        lightcolor=t.ACCENT_AMBER,
        darkcolor=t.ACCENT_AMBER,
        thickness=14,
    )

    # ---- scrollbars ----
    for orient in ("Vertical", "Horizontal"):
        style.configure(
            f"{orient}.TScrollbar",
            background=t.BG_PANEL,
            troughcolor=t.BG_ROOT,
            bordercolor=t.BG_ROOT,
            arrowcolor=t.ACCENT_AMBER,
            lightcolor=t.BG_PANEL,
            darkcolor=t.BG_PANEL,
            borderwidth=0,
            relief="flat",
        )
        style.map(
            f"{orient}.TScrollbar",
            background=[("active", t.BG_HOVER)],
            arrowcolor=[("active", t.ACCENT_CYAN)],
        )

    # ---- separators ----
    style.configure("TSeparator", background=t.DIVIDER)

    return t
