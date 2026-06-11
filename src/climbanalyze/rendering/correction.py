"""Generate a side-by-side actual/corrected pose diagram for a detected issue.

Rendered entirely with OpenCV (no extra dependencies).

Left panel  — ACTUAL    — detected skeleton in gray
Right panel — CORRECTED — corrected skeleton in green

Active holds are inferred from wrist (hand holds) and ankle (foot holds)
keypoints at the peak frame.  The correction algorithm keeps those anchor
points fixed and adjusts the torso/limbs using a two-bone IK solver.
"""

from __future__ import annotations

import math
import os
from typing import Optional

from ..analysis.center_of_mass import CenterOfMass
from ..analysis.rules.base import Issue
from ..pose.schema import PoseFrame
from .repose import repose_person

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# Canvas dimensions
_W, _H = 900, 480

_BG = (22, 22, 22)
_GRAY_SKEL = (90, 90, 90)
_GRAY_KP = (120, 120, 120)
_GREEN = (60, 220, 80)
_HAND_HOLD = (50, 210, 255)   # cyan-yellow: hand holds
_FOOT_HOLD = (60, 140, 255)   # orange: foot holds
_TEXT_DIM = (80, 80, 80)
_TEXT_MAIN = (200, 200, 200)
_ACCENT = (120, 190, 255)

_SKELETON_PAIRS = [
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

_HOLD_JOINTS = frozenset({"left_wrist", "right_wrist", "left_ankle", "right_ankle"})


# ---------------------------------------------------------------------------
# Keypoint helpers
# ---------------------------------------------------------------------------

def _extract_kps(frame: PoseFrame, conf_threshold: float) -> dict[str, Optional[tuple]]:
    result: dict[str, Optional[tuple]] = {}
    for kp in frame.keypoints:
        valid = (
            kp.confidence >= conf_threshold
            and not (kp.x < 0.01 and kp.y < 0.01)
        )
        result[kp.name] = (kp.x, kp.y) if valid else None
    return result


def _bone_len(kps: dict, a: str, b: str) -> float:
    pa, pb = kps.get(a), kps.get(b)
    if pa is None or pb is None:
        return 0.12
    dx, dy = pa[0] - pb[0], pa[1] - pb[1]
    return max(0.03, (dx * dx + dy * dy) ** 0.5)


# ---------------------------------------------------------------------------
# Two-bone IK solver
# ---------------------------------------------------------------------------

def _ik_2bone(
    base: tuple, tip: tuple, r1: float, r2: float, hint: Optional[tuple]
) -> tuple:
    """Given a fixed base and fixed tip, compute the middle joint.

    r1 = upper segment length, r2 = lower segment length.
    hint = current middle-joint position used to choose elbow side.
    """
    dx, dy = tip[0] - base[0], tip[1] - base[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= 1e-6:
        return base

    max_reach = r1 + r2 - 1e-4
    if dist >= max_reach:
        t = r1 / dist
        return (base[0] + dx * t, base[1] + dy * t)

    min_reach = abs(r1 - r2)
    if dist <= min_reach:
        dist = min_reach + 1e-4

    cos_a = (r1 * r1 + dist * dist - r2 * r2) / (2.0 * r1 * dist)
    cos_a = max(-1.0, min(1.0, cos_a))
    angle_a = math.acos(cos_a)
    base_angle = math.atan2(dy, dx)

    candidates = []
    for sign in (1, -1):
        a = base_angle + sign * angle_a
        mid = (base[0] + r1 * math.cos(a), base[1] + r1 * math.sin(a))
        candidates.append(mid)

    if hint is None:
        return candidates[0]

    # Cross-product sign of (tip-base) × (hint-base) tells which side hint is on
    cross_hint = dx * (hint[1] - base[1]) - dy * (hint[0] - base[0])
    for mid in candidates:
        cross_mid = dx * (mid[1] - base[1]) - dy * (mid[0] - base[0])
        if (cross_hint >= 0) == (cross_mid >= 0):
            return mid
    return candidates[0]


# ---------------------------------------------------------------------------
# Issue-specific corrections
# ---------------------------------------------------------------------------

def _apply_correction(kps: dict, issue_code: str) -> dict:
    """Return a new kps dict with corrected joint positions.

    Hold positions (wrists + ankles) are kept fixed as anchors.
    """
    c = {k: v for k, v in kps.items()}  # shallow copy

    if issue_code == "over_pulling_with_arms":
        # Drive hips up toward holds; shoulders rise only slightly.
        # Hips must stay at least 0.05 below the shoulder on the same side.
        hip_rise = 0.13
        shoulder_rise = 0.03
        wrist_ys = [kps[f"{s}_wrist"][1] for s in ("left", "right") if kps.get(f"{s}_wrist")]
        min_shoulder_y = (min(wrist_ys) + 0.03) if wrist_ys else 0.0
        for j_side in ("left", "right"):
            j = f"{j_side}_hip"
            shoulder = kps.get(f"{j_side}_shoulder")
            if c.get(j):
                x, y = c[j]
                new_y = y - hip_rise
                # Hips can't overtake shoulder — keep at least 0.05 below it
                if shoulder:
                    new_y = max(shoulder[1] + 0.05, new_y)
                c[j] = (x, new_y)
        for j in ("left_shoulder", "right_shoulder"):
            if c.get(j):
                x, y = c[j]
                c[j] = (x, max(min_shoulder_y, y - shoulder_rise))
        # Recompute elbows with IK (wrists = hold anchors, unchanged).
        # Use a synthetic hint pointing elbow downward — after shoulders rise the
        # original elbow hint can flip IK to the wrong side.
        for side in ("left", "right"):
            shoulder = c.get(f"{side}_shoulder")
            wrist = kps.get(f"{side}_wrist")
            if shoulder and wrist:
                r1 = _bone_len(kps, f"{side}_shoulder", f"{side}_elbow")
                r2 = _bone_len(kps, f"{side}_elbow", f"{side}_wrist")
                mid_x = (shoulder[0] + wrist[0]) / 2.0
                mid_y = (shoulder[1] + wrist[1]) / 2.0
                downward_hint = (mid_x, mid_y + 0.08)
                c[f"{side}_elbow"] = _ik_2bone(shoulder, wrist, r1, r2, downward_hint)
        # Recompute knees with IK (ankles = foot-hold anchors, unchanged)
        for side in ("left", "right"):
            hip = c.get(f"{side}_hip")
            ankle = kps.get(f"{side}_ankle")
            hint = kps.get(f"{side}_knee")
            if hip and ankle:
                r1 = _bone_len(kps, f"{side}_hip", f"{side}_knee")
                r2 = _bone_len(kps, f"{side}_knee", f"{side}_ankle")
                c[f"{side}_knee"] = _ik_2bone(hip, ankle, r1, r2, hint)

    elif issue_code == "locked_elbow_too_early":
        # Introduce a ~20% bend by offsetting the elbow perpendicular to the
        # shoulder-wrist axis.  Wrist stays at hold anchor; only elbow moves.
        for side in ("left", "right"):
            shoulder = kps.get(f"{side}_shoulder")
            wrist = kps.get(f"{side}_wrist")  # hold anchor — unchanged
            hint = kps.get(f"{side}_elbow")
            if shoulder and wrist:
                dx, dy = wrist[0] - shoulder[0], wrist[1] - shoulder[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < 1e-6:
                    continue
                # Midpoint along shoulder-wrist line
                mx = (shoulder[0] + wrist[0]) / 2.0
                my = (shoulder[1] + wrist[1]) / 2.0
                # Unit perpendicular (rotated 90° CCW from direction)
                px_dir, py_dir = -dy / dist, dx / dist
                # Match the side the current elbow is on
                if hint:
                    cross = dx * (hint[1] - shoulder[1]) - dy * (hint[0] - shoulder[0])
                    if cross < 0:
                        px_dir, py_dir = -px_dir, -py_dir
                # Offset proportional to arm length for consistent visual bend
                offset = dist * 0.22
                c[f"{side}_elbow"] = (mx + px_dir * offset, my + py_dir * offset)

    elif issue_code == "unstable_center_of_mass":
        # Shift trunk so CoM sits centered over ankle midpoint.
        la, ra = kps.get("left_ankle"), kps.get("right_ankle")
        if la and ra:
            ankle_mid_x = (la[0] + ra[0]) / 2.0
            trunk = [kps[k] for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip") if kps.get(k)]
            if trunk:
                trunk_cx = sum(p[0] for p in trunk) / len(trunk)
                shift = (ankle_mid_x - trunk_cx) * 0.65
                for j in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
                    if c.get(j):
                        x, y = c[j]
                        c[j] = (x + shift, y)
                # Recompute elbows via IK after torso shift (wrists = hold anchors)
                for side in ("left", "right"):
                    shoulder = c.get(f"{side}_shoulder")
                    wrist = kps.get(f"{side}_wrist")
                    hint = kps.get(f"{side}_elbow")
                    if shoulder and wrist:
                        r1 = _bone_len(kps, f"{side}_shoulder", f"{side}_elbow")
                        r2 = _bone_len(kps, f"{side}_elbow", f"{side}_wrist")
                        c[f"{side}_elbow"] = _ik_2bone(shoulder, wrist, r1, r2, hint)

    elif issue_code == "poor_foot_engagement":
        # Align knees over ankles; shift hips accordingly.
        for side in ("left", "right"):
            ankle = kps.get(f"{side}_ankle")
            knee = kps.get(f"{side}_knee")
            hip = kps.get(f"{side}_hip")
            if ankle and knee and hip:
                knee_dx = ankle[0] - knee[0]
                c[f"{side}_knee"] = (knee[0] + knee_dx * 0.65, knee[1])
                c[f"{side}_hip"] = (hip[0] + knee_dx * 0.35, hip[1])

    return c


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _compute_bbox(points: list[tuple], pad: float = 0.12) -> tuple:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    sx = max(max(xs) - min(xs), 0.05)
    sy = max(max(ys) - min(ys), 0.05)
    return (
        min(xs) - sx * pad,
        min(ys) - sy * pad,
        max(xs) + sx * pad,
        max(ys) + sy * pad,
    )


def _make_to_px(bbox: tuple, px0: int, py0: int, px1: int, py1: int):
    """Return a closure mapping normalized (x, y) → panel pixel (px, py)."""
    nx0, ny0, nx1, ny1 = bbox
    pw, ph = px1 - px0, py1 - py0
    span_x = max(nx1 - nx0, 1e-6)
    span_y = max(ny1 - ny0, 1e-6)
    scale = min(pw / span_x, ph / span_y)
    ox = px0 + (pw - span_x * scale) / 2.0
    oy = py0 + (ph - span_y * scale) / 2.0

    def to_px(xy: tuple) -> tuple:
        return (int(ox + (xy[0] - nx0) * scale), int(oy + (xy[1] - ny0) * scale))

    return to_px


def _draw_skeleton(img, kps: dict, to_px, skel_color: tuple, kp_color: tuple):
    for a, b in _SKELETON_PAIRS:
        pa, pb = kps.get(a), kps.get(b)
        if pa and pb:
            cv2.line(img, to_px(pa), to_px(pb), skel_color, 2, cv2.LINE_AA)

    for name, pos in kps.items():
        if pos is None or name == "nose":
            continue
        px = to_px(pos)
        if name in _HOLD_JOINTS:
            color = _HAND_HOLD if "wrist" in name else _FOOT_HOLD
            cv2.circle(img, px, 7, color, -1, cv2.LINE_AA)
            cv2.circle(img, px, 7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(img, px, 5, kp_color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Correction tip text per issue code
# ---------------------------------------------------------------------------

_ROUTE_HOLD = (140, 100, 180)  # muted violet — visible but reads as background


_TIPS = {
    "over_pulling_with_arms":
        "Drive hips UP toward your hands - push with legs, not just pull with arms",
    "unstable_center_of_mass":
        "Move one limb at a time - keep your center of mass balanced over your feet",
    "poor_foot_engagement":
        "Trust your feet - weight your footholds, knee directly above your toes",
    "locked_elbow_too_early":
        "Keep a slight elbow bend through the move - don't lock out mid-reach",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _draw_route_holds(
    img, route_holds: list, to_px, x0: int, y0: int, x1: int, y1: int
) -> None:
    """Draw background markers for all detected route holds within panel bounds."""
    for pos in route_holds:
        px, py = to_px(pos)
        if x0 <= px <= x1 and y0 <= py <= y1:
            cv2.circle(img, (px, py), 6, _ROUTE_HOLD, 1, cv2.LINE_AA)
            cv2.line(img, (px - 4, py), (px + 4, py), _ROUTE_HOLD, 1, cv2.LINE_AA)
            cv2.line(img, (px, py - 4), (px, py + 4), _ROUTE_HOLD, 1, cv2.LINE_AA)


def _bbox_px(points_norm: list[tuple], w: int, h: int, pad: float = 0.25) -> tuple:
    xs = [p[0] * w for p in points_norm]
    ys = [p[1] * h for p in points_norm]
    sx = max(max(xs) - min(xs), 0.05 * w)
    sy = max(max(ys) - min(ys), 0.05 * h)
    x0 = max(0, int(min(xs) - sx * pad))
    y0 = max(0, int(min(ys) - sy * pad))
    x1 = min(w, int(max(xs) + sx * pad))
    y1 = min(h, int(max(ys) + sy * pad))
    return x0, y0, x1, y1


def _place_and_overlay(canvas, src_img, crop, panel, kps, w, h, skel_color, kp_color, route_holds):
    """Resize *src_img[crop]* into *panel*, then overlay skeleton + holds.

    Returns nothing; draws onto *canvas*. Same crop+panel on both sides keeps the
    two climbers at an identical scale ("比例协调").
    """
    x0, y0, x1, y1 = crop
    px0, py0, px1, py1 = panel
    cw, ch = max(1, x1 - x0), max(1, y1 - y0)
    pw, ph = px1 - px0, py1 - py0
    s = min(pw / cw, ph / ch)
    dw, dh = max(1, int(cw * s)), max(1, int(ch * s))
    sub = cv2.resize(src_img[y0:y1, x0:x1], (dw, dh), interpolation=cv2.INTER_AREA)
    ox, oy = px0 + (pw - dw) // 2, py0 + (ph - dh) // 2
    canvas[oy:oy + dh, ox:ox + dw] = sub

    def to_px(nx, ny):
        return (int(ox + (nx * w - x0) * s), int(oy + (ny * h - y0) * s))

    if route_holds:
        for hx, hy in route_holds:
            p = to_px(hx, hy)
            if px0 <= p[0] <= px1 and py0 <= p[1] <= py1:
                cv2.circle(canvas, p, 6, _ROUTE_HOLD, 1, cv2.LINE_AA)

    for a, b in _SKELETON_PAIRS:
        pa, pb = kps.get(a), kps.get(b)
        if pa and pb:
            cv2.line(canvas, to_px(*pa), to_px(*pb), skel_color, 2, cv2.LINE_AA)
    for name, pos in kps.items():
        if pos is None or name == "nose":
            continue
        p = to_px(*pos)
        if name in _HOLD_JOINTS:
            c = _HAND_HOLD if "wrist" in name else _FOOT_HOLD
            cv2.circle(canvas, p, 6, c, -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 6, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, p, 4, kp_color, -1, cv2.LINE_AA)


def _render_real_diagram(image, kps, corrected, issue: Issue, route_holds) -> Optional["np.ndarray"]:
    """Real climber (left) vs re-posed real climber (right), each with skeleton."""
    h, w = image.shape[:2]
    reposed = repose_person(image, kps, corrected)
    if reposed is None:
        return None

    # Shared crop covering both actual and corrected joints → identical scale.
    pts = [v for v in kps.values() if v is not None] + [v for v in corrected.values() if v is not None]
    if route_holds:
        for hx, hy in route_holds:
            if any(((hx - p[0]) ** 2 + (hy - p[1]) ** 2) ** 0.5 <= 0.2 for p in pts):
                pts.append((hx, hy))
    crop = _bbox_px(pts, w, h)

    canvas = np.full((_H, _W, 3), _BG, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    label = issue.label[:44]
    (tw, _), _ = cv2.getTextSize(label, font, 0.58, 1)
    cv2.putText(canvas, label, ((_W - tw) // 2, 22), font, 0.58, _TEXT_MAIN, 1, cv2.LINE_AA)
    cv2.putText(canvas, "ACTUAL", (28, 42), font, 0.48, _TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, "CORRECTED", (462, 42), font, 0.48, _GREEN, 1, cv2.LINE_AA)
    cv2.line(canvas, (0, 52), (_W, 52), (42, 42, 42), 1)

    left_panel = (18, 58, 440, _H - 42)
    right_panel = (458, 58, _W - 18, _H - 42)
    _place_and_overlay(canvas, image, crop, left_panel, kps, w, h,
                       _GRAY_SKEL, (0, 220, 255), route_holds)
    _place_and_overlay(canvas, reposed, crop, right_panel, corrected, w, h,
                       _GREEN, _GREEN, route_holds)

    tip = _TIPS.get(issue.code, "Focus on body positioning and balance")
    (tw, _), _ = cv2.getTextSize(tip, font, 0.42, 1)
    cv2.putText(canvas, tip[:90], ((_W - min(tw, _W - 40)) // 2, _H - 14),
                font, 0.42, _ACCENT, 1, cv2.LINE_AA)
    return canvas


def generate_correction_diagram(
    frame: PoseFrame,
    com: Optional[CenterOfMass],
    issue: Issue,
    output_path: str,
    conf_threshold: float = 0.3,
    route_holds: Optional[list] = None,
    image=None,
) -> bool:
    """Render and save an actual-vs-corrected pose diagram.

    When *image* (the real BGR frame) is provided, both panels show the real
    climber with a skeleton overlay; the corrected panel re-poses the climber's
    real pixels per-bone (proportions preserved). Falls back to a skeleton-only
    diagram when the image is missing or reposing fails.

    Returns True if the image was written, False if skipped (missing deps,
    too few keypoints, or write error).
    """
    if not _HAS_CV2:
        return False

    kps = _extract_kps(frame, conf_threshold)
    valid = [v for v in kps.values() if v is not None]
    if len(valid) < 6:
        return False

    corrected = _apply_correction(kps, issue.code)

    if image is not None:
        canvas = _render_real_diagram(image, kps, corrected, issue, route_holds)
        if canvas is not None:
            try:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                cv2.imwrite(output_path, canvas)
                return True
            except Exception:
                return False
        # else: fall through to the skeleton-only diagram below.

    # Bounding box: actual keypoints + any route hold within 0.25 of the climber.
    # Using only actual keypoints (not corrected) keeps a stable scale so the
    # ACTUAL panel is always properly proportioned and corrections that move
    # joints don't shrink the reference view.
    bbox_pts = valid[:]
    if route_holds:
        for hx, hy in route_holds:
            for pos in valid:
                if ((hx - pos[0]) ** 2 + (hy - pos[1]) ** 2) ** 0.5 <= 0.25:
                    bbox_pts.append((hx, hy))
                    break
    bbox = _compute_bbox(bbox_pts)

    # Canvas
    img = np.full((_H, _W, 3), _BG, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Row 1 (y≈20): issue label centred
    label = issue.label[:44]
    (tw, _), _ = cv2.getTextSize(label, font, 0.58, 1)
    cv2.putText(img, label, ((_W - tw) // 2, 22), font, 0.58, _TEXT_MAIN, 1, cv2.LINE_AA)

    # Row 2 (y≈40): panel labels
    cv2.putText(img, "ACTUAL", (28, 42), font, 0.48, _TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, "CORRECTED", (462, 42), font, 0.48, _GREEN, 1, cv2.LINE_AA)

    # Header rule + panel divider
    cv2.line(img, (0, 52), (_W, 52), (42, 42, 42), 1)
    cv2.line(img, (450, 52), (450, _H - 38), (42, 42, 42), 1)

    # Hold legend (bottom-right)
    cv2.circle(img, (_W - 110, _H - 14), 5, _HAND_HOLD, -1, cv2.LINE_AA)
    cv2.putText(img, "hand hold", (_W - 100, _H - 10), font, 0.38, _TEXT_DIM, 1, cv2.LINE_AA)
    cv2.circle(img, (_W - 110, _H - 28), 5, _FOOT_HOLD, -1, cv2.LINE_AA)
    cv2.putText(img, "foot hold", (_W - 100, _H - 24), font, 0.38, _TEXT_DIM, 1, cv2.LINE_AA)
    # Route holds legend entry (only shown when route_holds were detected)
    if route_holds:
        cv2.circle(img, (_W - 110, _H - 42), 5, _ROUTE_HOLD, 1, cv2.LINE_AA)
        cv2.line(img, (_W - 114, _H - 42), (_W - 106, _H - 42), _ROUTE_HOLD, 1, cv2.LINE_AA)
        cv2.line(img, (_W - 110, _H - 46), (_W - 110, _H - 38), _ROUTE_HOLD, 1, cv2.LINE_AA)
        cv2.putText(img, "route hold", (_W - 100, _H - 38), font, 0.38, _TEXT_DIM, 1, cv2.LINE_AA)

    # Mapping functions — same scale, different destination rectangles
    left_to_px = _make_to_px(bbox, 18, 58, 440, _H - 42)
    right_to_px = _make_to_px(bbox, 458, 58, _W - 18, _H - 42)

    # Route hold background markers (drawn first so skeleton renders on top)
    if route_holds:
        _draw_route_holds(img, route_holds, left_to_px,  18,  58, 440, _H - 42)
        _draw_route_holds(img, route_holds, right_to_px, 458, 58, _W - 18, _H - 42)

    # Draw skeletons
    _draw_skeleton(img, kps, left_to_px, _GRAY_SKEL, _GRAY_KP)
    _draw_skeleton(img, corrected, right_to_px, _GREEN, _GREEN)

    # Coaching tip at bottom
    tip = _TIPS.get(issue.code, "Focus on body positioning and balance")
    (tw, _), _ = cv2.getTextSize(tip, font, 0.42, 1)
    if tw > _W - 40:
        # Word-wrap into two lines
        words = tip.split()
        line1 = ""
        for w in words:
            trial = w if not line1 else f"{line1} {w}"
            (tw2, _), _ = cv2.getTextSize(trial, font, 0.42, 1)
            if tw2 <= _W - 40:
                line1 = trial
            else:
                break
        line2 = tip[len(line1):].strip()
        (tw1, _), _ = cv2.getTextSize(line1, font, 0.42, 1)
        (tw2, _), _ = cv2.getTextSize(line2, font, 0.42, 1)
        cv2.putText(img, line1, ((_W - tw1) // 2, _H - 28), font, 0.42, _ACCENT, 1, cv2.LINE_AA)
        cv2.putText(img, line2, ((_W - tw2) // 2, _H - 12), font, 0.42, _ACCENT, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, tip, ((_W - tw) // 2, _H - 14), font, 0.42, _ACCENT, 1, cv2.LINE_AA)

    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, img)
        return True
    except Exception:
        return False
