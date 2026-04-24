"""Helpers to install + launch Wan2GP from inside the Image Trainer app.

Wan2GP is the community wrapper around Alibaba's Wan 2.2 video model that
adds GGUF quantisation, sliding windows, and block-swap CPU offloading so
the model fits on 10-12 GB cards. We don't bundle it — instead we:

1. Detect whether it's installed (check for the repo + venv).
2. Provide a one-shot installer that clones the repo, creates an isolated
   venv, and pip-installs the requirements.
3. Provide a launcher that starts the Gradio app in a background process
   so the user can drive it in their browser.

Everything runs through ``subprocess`` so the GUI's existing CLIRunner
pattern can stream stdout into the telemetry pane. The installer is
intentionally idempotent — re-running on an already-installed location
just reports "already installed" and exits clean.

Layout chosen for the install::

    <install_root>/
        Wan2GP/                    # the cloned repo
        Wan2GP/.venv/              # isolated venv (NOT shared with image-trainer)
        Wan2GP/loras_i2v/          # where Lightning LoRA goes (Wan2GP convention)

The default ``install_root`` is ``~/Apps/wan2gp/`` but the user can pick
another folder; the path is persisted to user settings so the GUI
remembers it across launches.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

#: Where Wan2GP's source lives upstream.
WAN2GP_REPO_URL = "https://github.com/deepbeepmeep/Wan2GP.git"

#: Default install root if the user hasn't picked one.
DEFAULT_INSTALL_ROOT = Path(os.path.expanduser("~/Apps/wan2gp"))

#: Sub-directory created inside ``install_root`` by ``git clone``.
REPO_DIRNAME = "Wan2GP"

#: Name of the venv inside the cloned repo.
VENV_DIRNAME = ".venv"

#: Conventional Wan2GP entry-point script. The repo currently calls this
#: ``wgp.py`` (some forks rename to ``app.py`` / ``run.py``); we probe a
#: small list and pick the first one that exists.
ENTRY_CANDIDATES = ("wgp.py", "app.py", "main.py")

#: Optional progress callback signature for log streaming.
ProgressCb = Callable[[str], None]


@dataclass
class InstallStatus:
    """Snapshot of the on-disk install state for the GUI to render."""

    install_root: Path
    repo_present: bool
    venv_present: bool
    entry_script: Optional[Path]   # absolute path to the picked entry script
    git_available: bool
    pip_available: bool

    @property
    def fully_installed(self) -> bool:
        return self.repo_present and self.venv_present and self.entry_script is not None


def detect(install_root: Path) -> InstallStatus:
    """Inspect ``install_root`` and report what's there + what's missing."""
    repo_dir = install_root / REPO_DIRNAME
    venv_dir = repo_dir / VENV_DIRNAME

    entry: Optional[Path] = None
    if repo_dir.is_dir():
        for cand in ENTRY_CANDIDATES:
            p = repo_dir / cand
            if p.is_file():
                entry = p
                break

    return InstallStatus(
        install_root=install_root,
        repo_present=repo_dir.is_dir() and (repo_dir / ".git").exists(),
        venv_present=venv_dir.is_dir() and (venv_dir / "bin" / "python").exists(),
        entry_script=entry,
        git_available=shutil.which("git") is not None,
        pip_available=shutil.which("pip") is not None or shutil.which("pip3") is not None,
    )


def venv_python(install_root: Path) -> Path:
    """Path to the interpreter inside Wan2GP's isolated venv."""
    return install_root / REPO_DIRNAME / VENV_DIRNAME / "bin" / "python"


def install(
    install_root: Path,
    *,
    progress: Optional[ProgressCb] = None,
    branch: Optional[str] = None,
) -> InstallStatus:
    """Install Wan2GP into ``install_root``. Idempotent.

    Steps:
      1. ``mkdir -p install_root``
      2. ``git clone`` (or ``git pull`` if repo already there)
      3. Create venv via ``python -m venv``
      4. Upgrade pip inside the venv
      5. ``pip install -r requirements.txt``

    Each step prints progress through ``progress`` (defaults to print()).
    Raises :class:`RuntimeError` on any subprocess failure with the failing
    command + exit code embedded in the message.

    Note: model weights are NOT downloaded here — Wan2GP downloads them
    on first launch into its own model cache. We just get the source +
    Python deps in place.
    """
    if progress is None:
        def progress(msg: str) -> None:
            print(msg, flush=True)

    install_root = Path(install_root).expanduser().resolve()
    install_root.mkdir(parents=True, exist_ok=True)
    repo_dir = install_root / REPO_DIRNAME

    progress(f"=== Installing Wan2GP into {install_root} ===")

    # Step 1: clone or pull.
    if not (repo_dir / ".git").exists():
        progress(f"Cloning {WAN2GP_REPO_URL} ...")
        cmd = ["git", "clone"]
        if branch:
            cmd += ["--branch", branch]
        cmd += [WAN2GP_REPO_URL, str(repo_dir)]
        _run(cmd, progress)
    else:
        progress(f"Repo already present; pulling latest into {repo_dir}")
        _run(["git", "-C", str(repo_dir), "pull", "--ff-only"], progress)

    # Step 2: venv.
    venv_dir = repo_dir / VENV_DIRNAME
    if not (venv_dir / "bin" / "python").exists():
        progress(f"Creating venv at {venv_dir}")
        _run([sys.executable, "-m", "venv", str(venv_dir)], progress)
    else:
        progress(f"Venv already present at {venv_dir}; skipping create")

    py = venv_dir / "bin" / "python"

    # Step 3: pip + requirements.
    progress("Upgrading pip inside Wan2GP venv")
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip"], progress)

    req_file = repo_dir / "requirements.txt"
    if req_file.exists():
        progress(f"Installing requirements from {req_file} (this can take a while)")
        _run(
            [str(py), "-m", "pip", "install", "-r", str(req_file)],
            progress,
        )
    else:
        progress(
            "WARNING: requirements.txt not found at the expected path. "
            "Wan2GP's repo layout may have changed; you may need to install "
            "deps manually. Skipping the pip-install step."
        )

    progress("=== Wan2GP install complete ===")
    return detect(install_root)


def launch(
    install_root: Path,
    *,
    extra_args: Optional[list[str]] = None,
    capture_output: bool = False,
) -> subprocess.Popen:
    """Spawn Wan2GP's Gradio entry point in the background.

    Returns the Popen so the caller can later .terminate() it. ``capture_output``
    controls whether the child's stdout/stderr is piped (for telemetry) or
    inherited from the parent (for users running through a terminal).

    Raises :class:`RuntimeError` if the install isn't complete enough to
    launch — call :func:`detect` first and surface the error to the user.
    """
    status = detect(install_root)
    if not status.fully_installed:
        missing = []
        if not status.repo_present:
            missing.append("repo")
        if not status.venv_present:
            missing.append("venv")
        if status.entry_script is None:
            missing.append("entry script")
        raise RuntimeError(
            f"Wan2GP isn't fully installed at {install_root}. Missing: "
            f"{', '.join(missing)}. Run the installer first."
        )

    py = venv_python(install_root)
    cmd = [str(py), str(status.entry_script)]
    if extra_args:
        cmd += list(extra_args)
    cwd = status.entry_script.parent

    if capture_output:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    else:
        proc = subprocess.Popen(cmd, cwd=str(cwd))
    return proc


# ---------- internal ----------

def _run(cmd: list[str], progress: ProgressCb) -> None:
    """Run a subprocess and stream output through ``progress``.

    Raises :class:`RuntimeError` with the cmd + exit code on non-zero exit.
    """
    progress(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        progress(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {' '.join(cmd)}"
        )
