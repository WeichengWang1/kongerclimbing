from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..pose.schema import PoseFrame

_Pt = Optional[tuple[float, float]]


@dataclass
class FrameMetrics:
    shoulder_line_angle: Optional[float] = None
    hip_line_angle: Optional[float] = None
    torso_angle: Optional[float] = None
    left_elbow_angle: Optional[float] = None
    right_elbow_angle: Optional[float] = None
    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None

    def to_dict(self) -> dict:
        d: dict = {}
        mapping = {
            "shoulder_line_angle": "shoulderLineAngle",
            "hip_line_angle": "hipLineAngle",
            "torso_angle": "torsoAngle",
            "left_elbow_angle": "leftElbowAngle",
            "right_elbow_angle": "rightElbowAngle",
            "left_knee_angle": "leftKneeAngle",
            "right_knee_angle": "rightKneeAngle",
        }
        for attr, json_key in mapping.items():
            val = getattr(self, attr)
            if val is not None:
                d[json_key] = round(val, 1)
        return d


def _joint_angle(a: _Pt, b: _Pt, c: _Pt) -> Optional[float]:
    """Angle in degrees at vertex b formed by segments b→a and b→c."""
    if a is None or b is None or c is None:
        return None
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    dot = ax * cx + ay * cy
    mag_a = math.sqrt(ax ** 2 + ay ** 2)
    mag_c = math.sqrt(cx ** 2 + cy ** 2)
    if mag_a == 0 or mag_c == 0:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag_a * mag_c)))))


def _line_angle(p1: _Pt, p2: _Pt) -> Optional[float]:
    """Angle of line p1→p2 relative to horizontal, in degrees."""
    if p1 is None or p2 is None:
        return None
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def compute_frame_metrics(frame: PoseFrame, conf_threshold: float = 0.0) -> FrameMetrics:
    def xy(name: str) -> _Pt:
        kp = frame.get(name)
        if not kp or kp.confidence < conf_threshold:
            return None
        # YOLO places occluded keypoints at pixel (0,0); reject those as invalid.
        if kp.x < 0.01 and kp.y < 0.01:
            return None
        return (kp.x, kp.y)

    ls, rs = xy("left_shoulder"), xy("right_shoulder")
    lh, rh = xy("left_hip"), xy("right_hip")
    le, re = xy("left_elbow"), xy("right_elbow")
    lw, rw = xy("left_wrist"), xy("right_wrist")
    lk, rk = xy("left_knee"), xy("right_knee")
    la, ra = xy("left_ankle"), xy("right_ankle")

    sc = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2) if ls and rs else None
    hc = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2) if lh and rh else None

    return FrameMetrics(
        shoulder_line_angle=_line_angle(ls, rs),
        hip_line_angle=_line_angle(lh, rh),
        torso_angle=_line_angle(hc, sc),
        left_elbow_angle=_joint_angle(ls, le, lw),
        right_elbow_angle=_joint_angle(rs, re, rw),
        left_knee_angle=_joint_angle(lh, lk, la),
        right_knee_angle=_joint_angle(rh, rk, ra),
    )
