"""Detect climbing route holds by colour segmentation.

Infers the route colour from known contact joints (wrists + ankles), then
finds all blobs of the same colour in the full frame.  No external deps —
pure OpenCV.
"""

from __future__ import annotations

from typing import Optional

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# Hue tolerance (OpenCV: 0-179), saturation/value lower bounds for the sampled colour
_HUE_TOL = 20
_SAT_MIN = 45
_VAL_MIN = 45


def detect_route_holds(
    image,
    kps: dict[str, Optional[tuple]],
    *,
    hue_tol: int = _HUE_TOL,
) -> list[tuple[float, float]]:
    """Return normalised (x, y) centres of all holds matching the route colour.

    The route colour is inferred from pixels sampled at the active contact
    joints (wrists and ankles).  Returns [] when cv2 is unavailable, no
    contact joint is detected, or the sampled colour is too dark / grey
    to distinguish from the wall.
    """
    if not _HAS_CV2:
        return []

    h_img, w_img = image.shape[:2]

    # --- Sample BGR pixels at each contact joint (7×7 patch) ---
    sample_bgr: list = []
    for joint in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
        pos = kps.get(joint)
        if pos is None:
            continue
        px = int(pos[0] * w_img)
        py = int(pos[1] * h_img)
        px = max(3, min(w_img - 4, px))
        py = max(3, min(h_img - 4, py))
        patch = image[py - 3: py + 4, px - 3: px + 4]
        sample_bgr.append(patch.reshape(-1, 3))

    if not sample_bgr:
        return []

    samples = np.vstack(sample_bgr).astype(np.uint8)
    hsv_s = cv2.cvtColor(samples.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

    med_h = int(np.median(hsv_s[:, 0]))
    med_s = int(np.median(hsv_s[:, 1]))
    med_v = int(np.median(hsv_s[:, 2]))

    # Reject grey / black — indistinguishable from wall
    if med_s < _SAT_MIN or med_v < _VAL_MIN:
        return []

    # --- Build HSV mask with hue tolerance ---
    lo_s = max(0, med_s - 60)
    lo_v = max(0, med_v - 70)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def _band(h_lo: int, h_hi: int):
        lo = np.array([max(0, h_lo), lo_s, lo_v], dtype=np.uint8)
        hi = np.array([min(179, h_hi), 255, 255], dtype=np.uint8)
        return cv2.inRange(hsv_img, lo, hi)

    h_lo, h_hi = med_h - hue_tol, med_h + hue_tol
    mask = _band(h_lo, h_hi)

    # Wrap-around at 0 / 179
    if h_lo < 0:
        mask = cv2.bitwise_or(mask, _band(180 + h_lo, 179))
    elif h_hi > 179:
        mask = cv2.bitwise_or(mask, _band(0, h_hi - 180))

    # Morphological cleanup: remove noise, fill gaps inside holds
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # --- Extract blob centres ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = w_img * h_img
    min_area = 0.0003 * img_area   # too small → noise
    max_area = 0.030 * img_area    # too large → wall patch, not a hold

    holds: list[tuple[float, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] < 1e-6:
            continue
        cx = M["m10"] / M["m00"] / w_img
        cy = M["m01"] / M["m00"] / h_img
        holds.append((cx, cy))

    return holds
