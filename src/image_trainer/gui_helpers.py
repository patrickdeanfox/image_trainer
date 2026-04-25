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


# ---------- user settings (per-user, not per-project) ----------

#: Per-user settings file. Lives in the projects root so it travels with
#: the projects collection but isn't tied to any single project. Used for
#: things that span projects: Wan2GP install path, light/dark preference
#: (future), etc.
_USER_SETTINGS_FILENAME = ".user_settings.json"


def _user_settings_path(projects_root: Path) -> Path:
    return projects_root / _USER_SETTINGS_FILENAME


def load_user_settings(projects_root: Path) -> dict:
    """Return the per-user settings dict, or empty dict if missing/corrupt."""
    p = _user_settings_path(projects_root)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_user_settings(projects_root: Path, data: dict) -> None:
    """Persist per-user settings, swallowing OS errors (best-effort)."""
    projects_root.mkdir(parents=True, exist_ok=True)
    try:
        _user_settings_path(projects_root).write_text(json.dumps(data, indent=2))
    except OSError:
        pass


def update_user_setting(projects_root: Path, key: str, value) -> dict:
    """Read-modify-write helper — merges {key: value} and returns the new dict.

    Passing ``value=None`` removes the key entirely (as opposed to
    persisting a literal ``null``). Lets callers use ``None`` as a "clear
    this saved default" signal — e.g. the Generate tab's
    ``Clear my defaults`` button.
    """
    data = load_user_settings(projects_root)
    if value is None:
        data.pop(key, None)
    else:
        data[key] = value
    save_user_settings(projects_root, data)
    return data


# ---------- tool discovery ----------

def which(name: str) -> Optional[str]:
    """Cross-platform shutil.which wrapper that always returns Optional[str]."""
    import shutil as _sh
    return _sh.which(name)


# ---------- shared LoRA library ----------

#: Folder under the projects root where community / civitai LoRAs live.
#: The Generate tab discovers .safetensors files here and offers to stack
#: them on top of the trained LoRA. Lives outside any project so a single
#: download is reusable across all of your projects.
SHARED_LORAS_DIRNAME = "shared_loras"


def shared_loras_dir(projects_root: Path) -> Path:
    """Path to the shared LoRA library; created on first use."""
    p = projects_root / SHARED_LORAS_DIRNAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_shared_loras(projects_root: Path) -> list[Path]:
    """Every .safetensors file directly under the shared LoRA folder.

    Sorted by mtime (newest first) so freshly-imported LoRAs surface at the
    top of the Generate tab list.
    """
    d = shared_loras_dir(projects_root)
    files = [p for p in d.iterdir() if p.is_file() and p.suffix == ".safetensors"]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def list_base_checkpoints(bases_dir: Path) -> list[Path]:
    """Every .safetensors / .ckpt file directly under the bases folder.

    Used by the Generate-tab base-model override picker. Skips zero-byte
    files (failed downloads) so the dropdown doesn't include broken
    files. Sorted alphabetically for predictable order.
    """
    if not bases_dir.exists():
        return []
    files: list[Path] = []
    for p in bases_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".safetensors", ".ckpt"):
            continue
        try:
            if p.stat().st_size < 100_000:
                # Almost certainly a failed download / HTML error page.
                # Real SDXL checkpoints are 5+ GB.
                continue
        except OSError:
            continue
        files.append(p)
    files.sort(key=lambda p: p.name.lower())
    return files


# ---------- folder inventory ----------

def folder_size_and_count(path: Path) -> tuple[int, int]:
    """Return (total_bytes, file_count) for ``path``.

    Walks the tree without following symlinks so we never count outside the
    project root. Missing directories return (0, 0) — Storage tab calls this
    even for folders that may not exist yet.
    """
    total_bytes = 0
    total_files = 0
    if not path.exists():
        return (0, 0)
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            try:
                total_bytes += os.lstat(os.path.join(dirpath, name)).st_size
                total_files += 1
            except OSError:
                pass
    return (total_bytes, total_files)


def format_bytes(n: int) -> str:
    """Human-friendly size — '1.4 GB' / '320 MB' / '4.2 KB' / '0 B'."""
    if n <= 0:
        return "0 B"
    for unit, divisor in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= divisor:
            return f"{n / divisor:.1f} {unit}"
    return f"{n} B"


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


# ---------- GPU process listing + kill ----------

class GpuProc:
    """One row of ``nvidia-smi --query-compute-apps``.

    Plain class (not a dataclass) to keep this file dependency-free at module
    load time — gui_helpers imports must stay cheap because every tab pulls it
    in eagerly.
    """

    __slots__ = ("pid", "name", "used_mib")

    def __init__(self, pid: int, name: str, used_mib: int) -> None:
        self.pid = pid
        self.name = name
        self.used_mib = used_mib

    def __repr__(self) -> str:  # pragma: no cover — debug aid only
        return f"GpuProc(pid={self.pid}, name={self.name!r}, mib={self.used_mib})"


def list_gpu_processes() -> list[GpuProc]:
    """Return every process currently holding GPU memory, per ``nvidia-smi``.

    Empty list if nvidia-smi isn't installed, returns no rows, or errors out
    — callers should treat that as "nothing to free" rather than a hard
    failure (the user might be on a CPU-only machine debugging the GUI).
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            text=True,
        )
    except Exception:
        return []
    rows: list[GpuProc] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            name = parts[1]
            used = int(parts[2])
        except ValueError:
            continue
        rows.append(GpuProc(pid=pid, name=name, used_mib=used))
    return rows


def kill_processes(pids: Iterable[int], *, escalate_after_s: float = 2.0) -> dict[int, str]:
    """SIGTERM each pid; if it's still alive after ``escalate_after_s``, SIGKILL.

    Returns ``{pid: status}`` where status is one of:
    ``"terminated"`` (died on SIGTERM),
    ``"killed"`` (needed SIGKILL),
    ``"gone"`` (already dead before we tried),
    ``"perm-denied"`` (we don't own the process — usually means root/system),
    ``"failed: <message>"`` (anything else).

    Designed for the "Free VRAM" button — non-blocking from the user's POV
    is achieved by the caller dispatching this to a thread; the function
    itself is synchronous so behaviour is predictable.
    """
    import os
    import signal
    import time

    out: dict[int, str] = {}
    pids = [int(p) for p in pids]

    # Phase 1: graceful term.
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            out[pid] = "gone"
        except PermissionError:
            out[pid] = "perm-denied"
        except Exception as e:  # pragma: no cover — defensive
            out[pid] = f"failed: {e}"

    if escalate_after_s > 0:
        time.sleep(escalate_after_s)

    # Phase 2: confirm + escalate.
    for pid in pids:
        if pid in out and out[pid] not in (None,):  # noqa: E711 — explicit
            # Already classified (gone / perm-denied / failed). Skip.
            if out[pid] != "perm-denied" and out[pid] != "gone":
                pass
        try:
            os.kill(pid, 0)  # alive?
        except ProcessLookupError:
            out.setdefault(pid, "terminated")
            if out.get(pid) not in ("perm-denied", "gone"):
                out[pid] = "terminated"
            continue
        except PermissionError:
            out[pid] = "perm-denied"
            continue
        # Still alive — escalate.
        try:
            os.kill(pid, signal.SIGKILL)
            out[pid] = "killed"
        except ProcessLookupError:
            out[pid] = "terminated"
        except PermissionError:
            out[pid] = "perm-denied"
        except Exception as e:  # pragma: no cover — defensive
            out[pid] = f"failed: {e}"
    return out
