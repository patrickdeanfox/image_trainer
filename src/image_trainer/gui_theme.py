"""Theme, font resolution, and ttk styling for the Tkinter GUI.

The GUI's aesthetic POV is **"alluring rumors"** — a warm jewel-tone
palette built around Behr's 2025 colour of the year. Cream parchment
ground, burgundy as the dominant accent, olive as the secondary, deep
slate for text, soft gold for highlights. Read it as a darkroom-atelier /
field-notes mood, not a server console.

Single source of truth: every colour the GUI uses is on :class:`Theme`,
applied via :func:`apply_style`. Don't sprinkle hex strings around the
rest of the GUI — reach into :data:`THEME` instead.

Two things live here:

1. :class:`Theme` — palette, type scale, spacing. :meth:`Theme.parchment`
   is the new default (warm cream); :meth:`Theme.dark` is kept as an
   opt-in dim-room variant that maps the same five jewel tones onto a
   warm near-black ground for late-night work.
2. :func:`apply_style` — configures every ttk widget class the GUI uses
   plus the raw-tk widgets that Tk can't reach via styles.

Font resolution is platform-aware: we look up :data:`DISPLAY_CANDIDATES`
and :data:`MONO_CANDIDATES` against ``tkFont.families()`` at startup and
pick the first installed face. No silent fallback to generic sans.
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
#: warm humanist serif (atelier feel); fall back through interface sans
#: families that ship on most desktops.
DISPLAY_CANDIDATES: Tuple[str, ...] = (
    "Cormorant Garamond",
    "EB Garamond",
    "Spectral",
    "Source Serif Pro",
    "Source Serif 4",
    "Georgia",
    "Helvetica Neue",
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

    Five base colours from Behr's "Alluring Rumors" 2025 palette plus
    derived light/dark tints for surfaces. Surfaces stack as
    parchment → cream → warm-beige → near-white-input so depth is
    communicated through tonal shift rather than borders. Accents are
    burgundy (dominant), olive (secondary), slate (tertiary), gold
    (highlight). The field names below intentionally keep their old
    hue-coded shape (``ACCENT_AMBER`` etc.) so we don't have to rename
    every call site — they just point to the new jewel tones now.
    """

    # --- surfaces (warm parchment, light by default) ---
    BG_ROOT: str = "#F1EDE2"        # base parchment (Behr cream)
    BG_PANEL: str = "#E8E1D0"        # panel — slightly deeper cream
    BG_ELEVATED: str = "#DCD2BB"     # raised cards
    BG_INPUT: str = "#FBF8EE"        # input fields — almost-white
    BG_HOVER: str = "#D4C8AC"        # hover wash on buttons
    BG_PRESSED: str = "#C2B392"      # pressed state

    # --- accents (the five Alluring Rumors hues) ---
    # ``ACCENT_AMBER`` is the legacy name for the dominant accent — it
    # now carries the burgundy. ``ACCENT_CYAN`` is the secondary (olive),
    # ``ACCENT_VIOLET`` the tertiary (slate). Renaming the fields would
    # cascade across every tab module, so we just remap meanings here.
    ACCENT_AMBER: str = "#7B4C4F"    # burgundy — primary / focus / progress
    ACCENT_CYAN: str = "#717051"     # olive — secondary / sub-headers
    ACCENT_VIOLET: str = "#4E505F"   # slate — tertiary / muted accent
    ACCENT_RED: str = "#5C383B"      # deeper burgundy for caution buttons
    ACCENT_GOLD: str = "#C99738"     # warmer mustard (deeper than #F6C886
                                      # so it reads on cream); the soft gold
                                      # itself lives on ``ACCENT_GOLD_SOFT``
    ACCENT_GREEN: str = "#717051"    # olive doubles as success/done
    ACCENT_GOLD_SOFT: str = "#F6C886"  # soft butter — used as a highlight
                                       # wash, not for fg/bg pairings

    # --- structural strokes ---
    DIVIDER: str = "#C9BDA2"         # warm beige rule
    BORDER_SOFT: str = "#B5AB94"     # field borders, softer than divider
    FOCUS_RING: str = "#7B4C4F"      # burgundy ring on focus

    # --- text (deep slate; #4E505F mid-stop and a deeper tint for body) ---
    TEXT_PRIMARY: str = "#2A2C36"    # body — derived from #4E505F
    TEXT_SECONDARY: str = "#4E505F"  # the slate as-is; secondary copy
    TEXT_MUTED: str = "#7A7468"      # warm grey for hints / placeholders
    TEXT_ON_ACCENT: str = "#F1EDE2"  # cream on burgundy / olive / slate

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
    def parchment(cls) -> "Theme":
        """Default — warm cream parchment with jewel-tone accents."""
        return cls()

    @classmethod
    def dark(cls) -> "Theme":
        """Dim-room variant — same five jewel hues over warm near-black.

        Kept around for late-night curation sessions. Not wired to a
        runtime toggle yet; switch by changing the call in
        :func:`apply_style`.
        """
        return cls(
            BG_ROOT="#1F1A14",      # deep warm umber
            BG_PANEL="#2A231B",      # slightly raised
            BG_ELEVATED="#3A2F23",   # cards
            BG_INPUT="#15110C",      # input wells (deeper than root)
            BG_HOVER="#3F3327",
            BG_PRESSED="#544230",
            ACCENT_AMBER="#C99090",  # burgundy lifted for dark bg readability
            ACCENT_CYAN="#A6A578",   # olive lifted
            ACCENT_VIOLET="#A0A4B5", # slate lifted
            ACCENT_RED="#8E5358",
            ACCENT_GOLD="#F6C886",   # soft gold reads great on warm dark
            ACCENT_GREEN="#A6A578",
            ACCENT_GOLD_SOFT="#F6C886",
            DIVIDER="#3F3327",
            BORDER_SOFT="#544230",
            FOCUS_RING="#F6C886",
            TEXT_PRIMARY="#F1EDE2",
            TEXT_SECONDARY="#D8CFBE",
            TEXT_MUTED="#9C927B",
            TEXT_ON_ACCENT="#1F1A14",
        )

    @classmethod
    def light(cls) -> "Theme":
        """Alias kept for backwards compatibility — returns parchment."""
        return cls.parchment()

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
            ACCENT_GOLD_SOFT=self.ACCENT_GOLD_SOFT,
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
THEME: Theme = Theme.parchment()


# ---------- style application ----------

def apply_style(root: tk.Tk) -> Theme:
    """Resolve fonts, apply the parchment visual language, return the live theme.

    Must be called *after* the :class:`tk.Tk` root is created so we can
    inspect installed font families.
    """
    global THEME

    display_face = _resolve_face(DISPLAY_CANDIDATES, fallback="Georgia")
    mono_face = _resolve_face(MONO_CANDIDATES, fallback="Courier New")
    THEME = Theme.parchment().with_fonts(display_face, mono_face)
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
