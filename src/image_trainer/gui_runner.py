"""Subprocess runner for the GUI.

Owns the `subprocess.Popen` for the currently-running ``trainer`` CLI
invocation. Holding the handle on the instance (rather than inside a
worker thread's local scope) lets the "Stop (graceful)" button actually
deliver a SIGINT to the child.

Stdout is tailed on a background thread into a thread-safe queue; the Tk
main loop drains that queue on a timer so the UI thread never blocks.
"""

from __future__ import annotations

import queue
import signal
import subprocess
import sys
import threading
from typing import Optional


class CLIRunner:
    """Manages the currently-running ``trainer`` subprocess for the GUI.

    A single :class:`CLIRunner` is owned by the main GUI instance and
    reused across actions (prep / caption / train / generate).
    """

    def __init__(self, log_queue: "queue.Queue[str]") -> None:
        self.log_queue = log_queue
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        """True while the pump thread is still alive."""
        return self.thread is not None and self.thread.is_alive()

    def start(self, args: list[str]) -> None:
        """Spawn ``python -m image_trainer.cli <args>`` and start the pump.

        Raises :class:`RuntimeError` if another job is already running.
        """
        if self.is_running():
            raise RuntimeError("another step is still running")
        cmd = [sys.executable, "-m", "image_trainer.cli", *args]
        self.log_queue.put(f"$ {' '.join(cmd)}\n")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _pump() -> None:
            assert self.proc is not None
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log_queue.put(line)
            self.proc.wait()
            self.log_queue.put(f"[exit {self.proc.returncode}]\n")

        self.thread = threading.Thread(target=_pump, daemon=True)
        self.thread.start()

    def stop_graceful(self) -> bool:
        """Signal the running subprocess to checkpoint and exit cleanly."""
        if not self.is_running() or self.proc is None:
            return False
        try:
            if sys.platform.startswith("win"):
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                self.proc.send_signal(signal.SIGINT)
            return True
        except Exception as e:
            self.log_queue.put(f"[stop failed: {e}]\n")
            return False
