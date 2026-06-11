"""Detect the active climbing segment within a clip.

Videos usually contain non-climbing footage at the head and tail: the climber
walking toward the wall, chalking up, resting, or stepping off at the top.
Running the rule engine over those frames produces false issues (e.g. an
"unstable center of mass" while the climber is simply walking).

This module finds the [start, end] frame window where the climber is actually
on the wall, so the pipeline can gate analysis to it.

Signal (combined, see brainstorm decision):
  A frame is "climb-active" if EITHER
    - a wrist is above the shoulder line (reaching/gripping holds), OR
    - the center of mass is ascending over a short lookback (pulling up).
  The climb starts at the first sustained run of active frames and ends at the
  last active frame of that run (small inactive gaps are tolerated via hysteresis).
"""

from __future__ import annotations

from typing import Optional

from ..pose.schema import PoseFrame
from .center_of_mass import CenterOfMass

# A keypoint YOLO could not localize is parked at (0, 0); reject those.
_OCCLUDED_EPS = 0.01

# Wrist must clear the shoulder line by this normalized margin to count as "hands up".
_HANDS_UP_MARGIN = 0.0

# CoM must rise (y decreases) by at least this normalized amount over the lookback.
_COM_RISE_THRESHOLD = 0.012
_COM_RISE_LOOKBACK = 3  # frames


def _valid(kp) -> bool:
    return (
        kp is not None
        and kp.reliable
        and not (kp.x < _OCCLUDED_EPS and kp.y < _OCCLUDED_EPS)
    )


def _hands_above_shoulders(frame: PoseFrame) -> bool:
    shoulders = [frame.get("left_shoulder"), frame.get("right_shoulder")]
    ys = [s.y for s in shoulders if _valid(s)]
    if not ys:
        return False
    shoulder_y = sum(ys) / len(ys)
    for wname in ("left_wrist", "right_wrist"):
        w = frame.get(wname)
        if _valid(w) and w.y < shoulder_y - _HANDS_UP_MARGIN:  # y-down: smaller = higher
            return True
    return False


def _com_ascending(coms: list[Optional[CenterOfMass]], i: int) -> bool:
    c = coms[i]
    j = i - _COM_RISE_LOOKBACK
    if j < 0 or c is None or not c.reliable:
        return False
    cp = coms[j]
    if cp is None or not cp.reliable:
        return False
    return (cp.y - c.y) > _COM_RISE_THRESHOLD  # y decreased => moved up


def _frame_active(frame: PoseFrame, coms: list[Optional[CenterOfMass]], i: int) -> bool:
    if not frame.reliable:
        return False
    return _hands_above_shoulders(frame) or _com_ascending(coms, i)


def detect_climb_segment(
    frames: list[PoseFrame],
    coms: list[Optional[CenterOfMass]],
    target_fps: int,
) -> Optional[tuple[int, int]]:
    """Return (start_idx, end_idx) inclusive frame indices of the climb, or None.

    None means no sustained climbing was detected; the caller should decide
    whether to fall back to analyzing the whole clip.
    """
    n = len(frames)
    if n == 0:
        return None

    fps = max(1, target_fps)
    run_frames = max(3, round(0.8 * fps))    # 0.8 s sustained activity to start
    gap_frames = max(run_frames, round(10 * fps))  # 10 s inactivity ends the climb

    active = [_frame_active(frames[i], coms, i) for i in range(n)]

    start: Optional[int] = None
    end: Optional[int] = None
    state = "idle"
    run = 0
    gap = 0

    for i in range(n):
        if active[i]:
            run += 1
            gap = 0
        else:
            gap += 1
            run = 0

        if state == "idle":
            if run >= run_frames:
                start = i - run + 1
                end = i
                state = "climbing"
        else:  # climbing
            if active[i]:
                end = i
            elif gap >= gap_frames:
                break  # first sustained climb segment ends here

    if start is None:
        return None
    return start, end
