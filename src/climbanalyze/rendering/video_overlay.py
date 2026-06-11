"""Render a full annotated playback video — skeleton + CoM drawn on every frame.

The viewer plays *this* file instead of the raw clip so the skeleton, center of
mass, and its trail stay attached to the climber throughout playback, not only on
the per-issue still frames.

Frames are encoded to browser-playable H.264 via ffmpeg (raw BGR piped on stdin).
ffmpeg is already a hard dependency of the project, so this adds no new requirement.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional

try:
    import cv2  # noqa: F401  (availability flag only)
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from ..pose.schema import PoseFrame
from ..analysis.center_of_mass import CenterOfMass
from ..analysis.rules.base import Issue
from .annotator import annotate_frame

_TRAIL_LEN = 20  # frames of CoM history drawn behind the dot


def _issue_by_frame(
    issues: list[Issue], timestamps_ms: list[int], n_frames: int
) -> dict[int, Issue]:
    """Map each frame index to the issue active at its timestamp (earlier wins)."""
    active: dict[int, Issue] = {}
    for iss in issues:
        for i in range(n_frames):
            if iss.start_ms <= timestamps_ms[i] <= iss.end_ms and i not in active:
                active[i] = iss
    return active


def write_annotated_video(
    raw_images: list,
    render_frames: list[PoseFrame],
    coms: list[Optional[CenterOfMass]],
    fps: float,
    out_path: str,
    conf_threshold: float,
    issues: Optional[list[Issue]] = None,
) -> Optional[str]:
    """Draw skeleton/CoM on every frame and encode to H.264 at *out_path*.

    Returns *out_path* on success, or None if cv2/ffmpeg are unavailable, there
    are no frames, or encoding fails.
    """
    if not _HAS_CV2 or not raw_images:
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    h, w = raw_images[0].shape[:2]
    out_fps = max(1.0, float(fps))
    issue_map = (
        _issue_by_frame(issues, [f.timestamp_ms for f in render_frames], len(raw_images))
        if issues else {}
    )

    proc = subprocess.Popen(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{out_fps}",
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            # yuv420p needs even dimensions; trunc keeps odd-sized inputs valid.
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        for fi, image in enumerate(raw_images):
            com = coms[fi]
            trail_start = max(0, fi - _TRAIL_LEN)
            com_trail = [
                (coms[i].x, coms[i].y)
                for i in range(trail_start, fi + 1)
                if coms[i] and coms[i].reliable
            ]
            frame = annotate_frame(
                image=image,
                frame=render_frames[fi],
                com=com,
                com_trail=com_trail,
                issue=issue_map.get(fi),
                conf_threshold=conf_threshold,
            )
            proc.stdin.write(frame.tobytes())
    except (BrokenPipeError, OSError):
        proc.kill()
        proc.wait()
        return None
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()

    proc.stderr.read() if proc.stderr else b""
    proc.wait()
    return out_path if proc.returncode == 0 else None
