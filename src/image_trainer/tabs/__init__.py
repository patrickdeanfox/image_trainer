"""Per-tab GUI modules.

Each tab exposes a single ``build(gui)`` function that wires widgets into
the tab frame and returns a small object (or ``None``) holding any
per-tab state the main GUI might need to poke at (e.g. progress bars,
review state).

Splitting the GUI this way keeps each file small enough to edit without
AI context loss and makes tab-specific changes reviewable in isolation.
"""

from __future__ import annotations
