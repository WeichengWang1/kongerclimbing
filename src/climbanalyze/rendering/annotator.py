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
# Minimum normalized segment length to draw a skeleton line.
# Segments shorter than this are almost always YOLO misprojections (elbow ≈ shoulder).
_MIN_SEGMENT_LEN = 0.04


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

    # Skeleton lines (nose excluded).
    # Skip segments that are shorter than _MIN_SEGMENT_LEN: these almost always
    # indicate YOLO has projected elbow/wrist onto the same pixel as the shoulder.
    for a, b in _SKELETON:
        if a in kp_map and b in kp_map:
            ka, kb = kp_map[a], kp_map[b]
            dx, dy = ka.x - kb.x, ka.y - kb.y
            if (dx * dx + dy * dy) ** 0.5 < _MIN_SEGMENT_LEN:
                continue
            cv2.line(out, px(ka.x, ka.y), px(kb.x, kb.y), _COLOR_SKELETON, 2, cv2.LINE_AA)

    # Body keypoints — radius and fill scaled by confidence so unreliable points
    # are visually distinct (small hollow ring) from reliable ones (solid dot).
    for kp in kp_map.values():
        if kp.name == "nose":
            continue
        if kp.confidence >= conf_threshold * 1.5:   # clearly reliable: solid filled
            cv2.circle(out, px(kp.x, kp.y), 5, _COLOR_KP, -1, cv2.LINE_AA)
        else:                                         # marginal confidence: small hollow ring
            cv2.circle(out, px(kp.x, kp.y), 3, _COLOR_KP, 1, cv2.LINE_AA)

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

    # Issue label, evidence values, and coaching tip.
    # Font size and stroke scale with resolution so text stays legible on 1080p/4K
    # frames (fixed sizes looked tiny on high-res video). Long lines are wrapped so
    # the larger text never runs off the right edge.
    if issue:
        fs = max(max(h, w) / 1280.0, 1.0)   # 1.0 at 1280px, ~1.5 at 1080p, ~3.0 at 4K
        label_scale = 0.9 * fs
        body_scale = 0.62 * fs
        thickness = max(2, round(1.5 * fs))
        margin = int(14 * fs)
        line_gap = int(8 * fs)
        max_text_w = w - 2 * margin
        y = int(42 * fs)

        y = _draw_paragraph(out, f"{issue.label}  [{issue.severity}]", margin, y,
                            label_scale, thickness, max_text_w, line_gap, _COLOR_TEXT)
        evidence_text = _format_evidence(issue.evidence_metrics)
        if evidence_text:
            y = _draw_paragraph(out, evidence_text, margin, y, body_scale, thickness,
                                max_text_w, line_gap, (180, 230, 255))
        _draw_paragraph(out, issue.recommendation, margin, y, body_scale, thickness,
                        max_text_w, line_gap, _COLOR_TEXT)

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


def _wrap_text(
    text: str, scale: float, thickness: int, max_width: int
) -> list[str]:
    """Greedy word-wrap *text* so each line fits within *max_width* pixels."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = word if not current else f"{current} {word}"
        (tw, _), _ = cv2.getTextSize(trial, font, scale, thickness)
        if tw <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_paragraph(
    img, text: str, x: int, y: int,
    scale: float, thickness: int, max_width: int, line_gap: int,
    color: tuple = _COLOR_TEXT,
) -> int:
    """Draw *text* (word-wrapped) with a dark background box per line.

    Returns the y coordinate just below the drawn block, ready for the next paragraph.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = max(3, thickness + 2)
    for line in _wrap_text(text, scale, thickness, max_width):
        (tw, th), baseline = cv2.getTextSize(line, font, scale, thickness)
        cv2.rectangle(
            img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad),
            _COLOR_TEXT_BG, -1,
        )
        cv2.putText(img, line, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        y += th + baseline + pad + line_gap
    return y
