"""Back-compat shim.

All GUI code now lives in :mod:`image_trainer.gui_app` and the
``gui_theme`` / ``gui_runner`` / ``gui_helpers`` / ``gui_widgets`` siblings,
with per-step implementations in :mod:`image_trainer.tabs`.

Importers should migrate to ``from image_trainer.gui_app import launch`` —
this shim preserves the ``from .gui import launch`` call-site in
:mod:`image_trainer.cli`.
"""

from .gui_app import TrainerGUI, launch  # noqa: F401

__all__ = ["TrainerGUI", "launch"]
