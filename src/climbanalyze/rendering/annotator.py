from __future__ import annotations

from typing import Optional

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from ..pose.schema import PoseFrame
from ..analysis.center_of_mass import CenterOfMass
from ..analysis.rules.base import Issue

# Skeleton pairs — nose excluded: back-facing climbers almost always have
# unreliable nose detection, and shoulder-nose lines produce visual artifacts.
_SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# Nose shown only when confidence is clearly reliable (face visible, not occluded)
_NOSE_CONF_THRESHOLD = 0.65

_COLOR_SKELETON = (0, 200, 255)    # BGR: yellow-orange
_COLOR_KP = (0, 255, 100)          # BGR: green
_COLOR_COM = (0, 100, 255)         # BGR: orange-red
_COLOR_COM_TRAIL = (255, 180, 0)   # BGR: bright cyan-blue
_COLOR_TEXT = (255, 255, 255)
_COLOR_TEXT_BG = (30, 30, 30)

# Maximum normalized distance between consecutive trail points; larger gaps are skipped.
_TRAIL_MAX_JUMP = 0.08


def annotate_frame(
    image,
    frame: PoseFrame,
    com: Optional[CenterOfMass],
    com_trail: list[tuple[float, float]],
    issue: Optional[Issue],
    conf_threshold: float = 0.0,
):
    """Return a copy of *image* with skeleton, CoM, trail, and issue label drawn on it."""
    if not _HAS_CV2:
        return image

    out = image.copy()
    h, w = out.shape[:2]

    def px(x_norm: float, y_norm: float) -> tuple[int, int]:
        return int(x_norm * w), int(y_norm * h)

    kp_map = {kp.name: kp for kp in frame.keypoints if kp.confidence >= conf_threshold}

    # Skeleton lines (nose excluded)
    for a, b in _SKELETON:
        if a in kp_map and b in kp_map:
            cv2.line(out, px(kp_map[a].x, kp_map[a].y), px(kp_map[b].x, kp_map[b].y),
                     _COLOR_SKELETON, 2, cv2.LINE_AA)

    # Body keypoints (nose excluded from kp_map loop — handled separately below)
    for kp in kp_map.values():
        if kp.name == "nose":
            continue
        cv2.circle(out, px(kp.x, kp.y), 4, _COLOR_KP, -1, cv2.LINE_AA)

    # Nose: only display when confidence clearly indicates face is visible (not occluded).
    # Skipping low-confidence nose avoids the pink artifact dot on the back/chest when the
    # climber faces the wall and YOLO misprojects the nose onto the torso.
    nose = frame.get("nose")
    if nose and nose.confidence >= _NOSE_CONF_THRESHOLD:
        cv2.circle(out, px(nose.x, nose.y), 6, _COLOR_KP, -1, cv2.LINE_AA)

    # CoM trail — skip large jumps so line doesn't teleport across the frame
    filtered_trail = _filter_trail(com_trail, _TRAIL_MAX_JUMP)
    if len(filtered_trail) >= 2:
        for i in range(1, len(filtered_trail)):
            alpha = int(80 + 175 * i / len(filtered_trail))  # fade older points
            color = (_COLOR_COM_TRAIL[0], _COLOR_COM_TRAIL[1], _COLOR_COM_TRAIL[2])
            cv2.line(out, px(*filtered_trail[i - 1]), px(*filtered_trail[i]),
                     color, 2, cv2.LINE_AA)

    # Current CoM
    if com and com.reliable:
        cv2.circle(out, px(com.x, com.y), 8, _COLOR_COM, -1, cv2.LINE_AA)
        cv2.circle(out, px(com.x, com.y), 8, (255, 255, 255), 2, cv2.LINE_AA)

    # Issue label, evidence values, and coaching tip
    if issue:
        label_text = f"{issue.label}  [{issue.severity}]"
        evidence_text = _format_evidence(issue.evidence_metrics)
        rec_text = issue.recommendation
        y = 30
        _draw_text_box(out, label_text, (10, y), scale=0.6)
        y += 30
        if evidence_text:
            _draw_text_box(out, evidence_text, (10, y), scale=0.42, color=(180, 230, 255))
            y += 26
        _draw_text_box(out, rec_text, (10, y), scale=0.42)

    return out


def _filter_trail(
    trail: list[tuple[float, float]], max_jump: float
) -> list[tuple[float, float]]:
    """Remove points that teleport more than *max_jump* from the previous point."""
    if not trail:
        return []
    result = [trail[0]]
    for pt in trail[1:]:
        dx = pt[0] - result[-1][0]
        dy = pt[1] - result[-1][1]
        if (dx * dx + dy * dy) ** 0.5 <= max_jump:
            result.append(pt)
    return result


def _format_evidence(metrics: dict) -> str:
    """Turn evidence metrics dict into a compact ASCII string for OpenCV rendering."""
    # OpenCV putText only supports ASCII — no Unicode (no degree sign, no delta).
    parts = []
    labels = {
        "elbowAngleDeltaDeg": "elbow d",
        "hipDisplacement": "hip disp",
        "kneeAngleDeltaDeg": "knee d",
        "directionChanges": "dir chg",
        "velocityVariance": "vel var",
        "ankleJitter": "jitter",
        "comShiftToSupport": "CoM shift",
        "leftElbowAngleDeg": "L-elbow",
        "rightElbowAngleDeg": "R-elbow",
        "postLockComProgress": "post-lock",
    }
    for key, val in metrics.items():
        label = labels.get(key, key)
        is_angle = "Deg" in key
        if isinstance(val, float):
            if is_angle:
                parts.append(f"{label}: {val:.1f}deg")
            elif abs(val) < 0.01:
                parts.append(f"{label}: {val:.4f}")
            elif abs(val) < 1:
                parts.append(f"{label}: {val:.3f}")
            else:
                parts.append(f"{label}: {val:.2f}")
        else:
            parts.append(f"{label}: {val}")
    return "  |  ".join(parts)


def _draw_text_box(
    img, text: str, origin: tuple[int, int],
    scale: float = 0.5,
    color: tuple = _COLOR_TEXT,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 4), _COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
