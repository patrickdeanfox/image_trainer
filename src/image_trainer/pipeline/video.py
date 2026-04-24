"""Overnight video pipeline: extract → upscale → interpolate → assemble.

This module runs the deterministic phases of the video workflow:

    Phase 4: extract frames from a Wan2GP raw .mp4 with ffmpeg
    Phase 5: upscale each frame 2x with realesrgan-ncnn-vulkan
    Phase 6: interpolate frames with rife-ncnn-vulkan
    Phase 7: assemble final .mp4 with ffmpeg

The video-generation phase itself (Wan2GP) lives outside this module — it
needs a long-running interactive process and the user typically drives it
through Wan2GP's own Gradio UI. We pick up the resulting .mp4 from the
project's video/<timestamp>/ folder and take it from there.

Every phase writes intermediates under ``<project>/video/<run_stamp>/``
so a crash mid-pipeline keeps progress. A run_stamp folder layout::

    video/20260423_220000/
        raw.mp4              # symlinked-in or user-dropped Wan2GP output
        frames/              # phase 4 output  (PNG sequence)
        upscaled/            # phase 5 output  (PNG sequence, 2x)
        interpolated/        # phase 6 output  (PNG sequence, 2x fps)
        final.mp4            # phase 7 output
        run.log              # tee'd phase output for the morning audit
"""

from __future__ import annotations

import datetime as dt
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

from ..config import Project

ProgressCb = Callable[[str, str], None]  # (phase, message) → None


def new_run_dir(project: Project) -> Path:
    """Mint a fresh ``video/<timestamp>/`` under the project."""
    base = project.root / "video"
    base.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def extract_frames(
    raw_mp4: Path,
    frames_dir: Path,
    progress: Optional[ProgressCb] = None,
) -> int:
    """Extract every frame from ``raw_mp4`` into ``frames_dir`` as PNGs.

    Returns the number of frames written. Uses ffmpeg's ``-qscale:v 1`` for
    the highest-quality PNG output (PNG is lossless either way; the flag
    matters only for the JPEG path, kept for habit).
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_mp4),
        "-qscale:v", "1",
        str(frames_dir / "frame_%06d.png"),
    ]
    if progress:
        progress("extract", f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    n = len(list(frames_dir.glob("frame_*.png")))
    if progress:
        progress("extract", f"Extracted {n} frames")
    return n


def upscale_frames(
    frames_dir: Path,
    upscaled_dir: Path,
    *,
    model: str = "realesr-animevideov3",
    scale: int = 2,
    progress: Optional[ProgressCb] = None,
) -> int:
    """Upscale every frame in ``frames_dir`` by ``scale`` via realesrgan-ncnn-vulkan.

    The ``realesr-animevideov3`` model handles AI-generated video well; for
    photoreal output try ``realesrgan-x4plus`` instead.
    """
    upscaled_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "realesrgan-ncnn-vulkan",
        "-i", str(frames_dir),
        "-o", str(upscaled_dir),
        "-n", model,
        "-s", str(scale),
    ]
    if progress:
        progress("upscale", f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    n = len(list(upscaled_dir.glob("*.png")))
    if progress:
        progress("upscale", f"Upscaled {n} frames")
    return n


def interpolate_frames(
    upscaled_dir: Path,
    interpolated_dir: Path,
    *,
    model: str = "rife-v4.6",
    multiplier: int = 2,
    progress: Optional[ProgressCb] = None,
) -> int:
    """Run RIFE to insert interpolated frames between every consecutive pair."""
    interpolated_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rife-ncnn-vulkan",
        "-i", str(upscaled_dir),
        "-o", str(interpolated_dir),
        "-m", model,
        "-n", str(multiplier),
    ]
    if progress:
        progress("interpolate", f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    n = len(list(interpolated_dir.glob("*.png")))
    if progress:
        progress("interpolate", f"Now have {n} frames")
    return n


def assemble_final(
    frames_dir: Path,
    out_path: Path,
    *,
    framerate: int = 32,
    crf: int = 18,
    progress: Optional[ProgressCb] = None,
) -> Path:
    """ffmpeg-encode a numbered PNG sequence to MP4 at ``framerate`` and ``crf``."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(framerate),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    if progress:
        progress("assemble", f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if progress:
        progress("assemble", f"Wrote {out_path}")
    return out_path


def run_post_generation_pipeline(
    project: Project,
    raw_mp4: Path,
    *,
    target_framerate: int = 32,
    rife_multiplier: int = 2,
    upscale_model: str = "realesr-animevideov3",
    upscale_scale: int = 2,
    progress: Optional[ProgressCb] = None,
) -> dict:
    """Run phases 4-7 end-to-end on a Wan2GP raw output.

    Returns a dict::

        {
            "run_dir": Path,
            "frames_dir": Path,
            "upscaled_dir": Path,
            "interpolated_dir": Path,
            "final_mp4": Path,
            "n_frames_raw": int,
            "n_frames_upscaled": int,
            "n_frames_interpolated": int,
        }

    Each phase writes to a dedicated subdir under ``<project>/video/<stamp>/``
    so re-runs don't stomp prior intermediates. If a phase fails, the
    earlier outputs remain on disk for inspection.
    """
    run_dir = new_run_dir(project)
    # Either copy or symlink the raw mp4 in so the run is self-contained.
    raw_in_run = run_dir / "raw.mp4"
    if not raw_in_run.exists():
        try:
            raw_in_run.symlink_to(Path(raw_mp4).resolve())
        except OSError:
            shutil.copy2(raw_mp4, raw_in_run)

    frames_dir = run_dir / "frames"
    upscaled_dir = run_dir / "upscaled"
    interpolated_dir = run_dir / "interpolated"
    final_mp4 = run_dir / "final.mp4"

    n_raw = extract_frames(raw_in_run, frames_dir, progress=progress)
    n_up = upscale_frames(
        frames_dir, upscaled_dir,
        model=upscale_model, scale=upscale_scale,
        progress=progress,
    )
    n_int = interpolate_frames(
        upscaled_dir, interpolated_dir,
        multiplier=rife_multiplier,
        progress=progress,
    )
    assemble_final(
        interpolated_dir, final_mp4,
        framerate=target_framerate,
        progress=progress,
    )

    return {
        "run_dir": run_dir,
        "frames_dir": frames_dir,
        "upscaled_dir": upscaled_dir,
        "interpolated_dir": interpolated_dir,
        "final_mp4": final_mp4,
        "n_frames_raw": n_raw,
        "n_frames_upscaled": n_up,
        "n_frames_interpolated": n_int,
    }
