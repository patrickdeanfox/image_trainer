"""Small cross-tab helpers: file dialogs, OS openers, ETA, config diff.

Nothing here owns state — everything is a pure function or a trivial
wrapper around stdlib/Tk. If it could plausibly live in two places, put
it here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import messagebox
from typing import Any, Iterable, Optional


# ---------- dialogs ----------

def ask_string(parent: tk.Tk, title: str, prompt: str) -> str:
    """Modal string-input dialog. Empty on cancel (simpler for callers)."""
    from tkinter import simpledialog
    return simpledialog.askstring(title, prompt, parent=parent) or ""


# ---------- OS openers ----------

def _platform_open(path: Path) -> None:
    """Best-effort 'open in default app'. Soft-fails with a message box."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(path)])
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")
    except FileNotFoundError:
        messagebox.showinfo(
            "Open failed",
            f"No system opener available. Path:\n{path}",
        )
    except Exception as e:
        messagebox.showerror("Open failed", f"{e}\nPath: {path}")


def open_folder(path: Path) -> None:
    """Reveal `path` in the OS file manager. Creates the folder if missing."""
    path.mkdir(parents=True, exist_ok=True)
    _platform_open(path)


def open_file(path: Path) -> None:
    """Open a file in the OS default viewer."""
    _platform_open(path)


# ---------- ETA ----------

def format_eta(elapsed_s: float, done: int, total: int) -> str:
    """Return a compact 'ETA 01:23:45' / 'ETA --:--:--' string.

    Uses a simple step-rate projection. Guarded against divide-by-zero for
    the early steps where the rate is too noisy to trust.
    """
    if done <= 0 or total <= 0 or elapsed_s <= 0:
        return "ETA --:--:--"
    remaining = total - done
    if remaining <= 0:
        return "ETA 00:00:00"
    rate = done / elapsed_s  # steps/sec
    if rate <= 0:
        return "ETA --:--:--"
    secs = int(remaining / rate)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"ETA {h:02d}:{m:02d}:{s:02d}"


def format_elapsed(elapsed_s: float) -> str:
    """Compact 'HH:MM:SS' wall-clock string."""
    secs = max(0, int(elapsed_s))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------- config diff ----------

#: Fields we never show in a diff (derived, rarely meaningful to the user,
#: or structurally noisy like the whole prompt_chips list).
_DIFF_IGNORED: frozenset = frozenset({
    "root",
    "prompt_chips",       # too long, edited via config.json
    "default_negative_prompt",
})


def config_diff(old: Any, new: Any) -> list[tuple[str, Any, Any]]:
    """Return list of (field, old_value, new_value) for changed fields.

    Accepts either two :class:`Project` instances (the usual case) or two
    dict snapshots. Paths are stringified for display. Ignored fields are
    skipped.
    """
    if not isinstance(old, dict):
        old_d = _snapshot(old)
    else:
        old_d = old
    if not isinstance(new, dict):
        new_d = _snapshot(new)
    else:
        new_d = new

    out: list[tuple[str, Any, Any]] = []
    for k in sorted(set(old_d) | set(new_d)):
        if k in _DIFF_IGNORED:
            continue
        ov = old_d.get(k)
        nv = new_d.get(k)
        if ov != nv:
            out.append((k, ov, nv))
    return out


def _snapshot(obj: Any) -> dict:
    """Convert a Project (or dataclass) to a plain serializable dict."""
    try:
        data = asdict(obj)
    except TypeError:
        data = dict(obj.__dict__) if hasattr(obj, "__dict__") else {}
    # stringify any Path values so diffs compare as strings
    for k, v in list(data.items()):
        if isinstance(v, Path):
            data[k] = str(v)
    return data


def format_config_diff(diff: Iterable[tuple[str, Any, Any]]) -> str:
    """Render a diff list as a user-friendly multi-line string."""
    lines: list[str] = []
    for field, old, new in diff:
        lines.append(f"• {field}:  {_fmt(old)}  →  {_fmt(new)}")
    if not lines:
        return "(no changes)"
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, str) and len(v) > 60:
        return v[:57] + "…"
    return str(v)


# ---------- recent projects storage ----------

_RECENT_FILENAME = ".recent_projects.json"
_RECENT_MAX = 8


def recent_path(projects_root: Path) -> Path:
    return projects_root / _RECENT_FILENAME


def load_recent(projects_root: Path) -> list[str]:
    """Return the MRU list of absolute project paths, newest first."""
    p = recent_path(projects_root)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return [str(x) for x in data][:_RECENT_MAX]
    except Exception:
        pass
    return []


def touch_recent(projects_root: Path, project_dir: Path) -> list[str]:
    """Mark `project_dir` as the most-recently-used; persist and return list."""
    projects_root.mkdir(parents=True, exist_ok=True)
    pd = str(project_dir.resolve())
    existing = [x for x in load_recent(projects_root) if x != pd]
    new_list = [pd] + existing
    new_list = new_list[:_RECENT_MAX]
    try:
        recent_path(projects_root).write_text(json.dumps(new_list, indent=2))
    except OSError:
        pass
    return new_list


# ---------- training log parsing ----------

def parse_step_line(line: str) -> Optional[dict]:
    """Parse a training progress line. Returns None if not recognized.

    Recognizes two formats emitted by ``pipeline/train.py``:

    - ``step 50/1500 loss=0.0231 lr=9.87e-05`` — the main training loop.
    - ``caching latents 3/20: 0002.png`` / ``caching embeds 3/20: ...`` —
      the one-time cache populates. Phase tag is ``"cache"``.

    The implementation scans for the first ``N/M`` token rather than
    relying on fixed word positions, so extra prefix words (``latents``,
    ``embeds``, or anything else we add later) don't need a parser bump.
    """
    line = line.strip()
    if line.startswith("step ") and "/" in line:
        try:
            parts = line.split()
            n, m = parts[1].split("/")
            out: dict = {"phase": "train", "step": int(n), "total": int(m)}
            for tok in parts[2:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
            return out
        except Exception:
            return None
    if line.startswith("caching "):
        for tok in line.split():
            # The 'N/M' token may carry a trailing ':' in real output
            # like ``caching latents 3/20: 0002.png``.
            clean = tok.rstrip(":")
            if "/" in clean:
                a, b = clean.split("/", 1)
                try:
                    return {"phase": "cache", "step": int(a), "total": int(b)}
                except ValueError:
                    continue
        return None
    return None


# ---------- nvidia-smi probe ----------

def probe_vram() -> Optional[tuple[int, int]]:
    """Return (used_MiB, total_MiB) from nvidia-smi, or None if unavailable.

    Deliberately short timeout — this runs on the Tk main loop timer.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=1.5,
            text=True,
        )
        first = out.strip().splitlines()[0]
        used_s, total_s = first.split(",")
        return int(used_s.strip()), int(total_s.strip())
    except Exception:
        return None
