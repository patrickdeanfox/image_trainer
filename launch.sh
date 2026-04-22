#!/usr/bin/env bash
# Image Trainer launcher.
# - Bootstraps a local .venv and `pip install -e .` on first run.
# - Activates the venv and launches the Tkinter wizard via `trainer gui`.
#
# Usage:
#   ./launch.sh           # normal launch
#   ./launch.sh --reset   # wipe .venv and reinstall

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

VENV_DIR="$REPO_DIR/.venv"

if [[ "${1:-}" == "--reset" ]]; then
  echo "[launch.sh] --reset: removing $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[launch.sh] First run: creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  echo "[launch.sh] Upgrading pip"
  python -m pip install --upgrade pip
  echo "[launch.sh] Installing image_trainer in editable mode (this can take a while — torch + friends)"
  python -m pip install -e .
else
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

echo "[launch.sh] Launching GUI..."
exec trainer gui
