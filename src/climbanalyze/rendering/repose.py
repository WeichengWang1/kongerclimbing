"""Repose the climber's *real* pixels into a corrected pose — per-bone rigidly.

This is NOT a global image warp (which shears/smears). Each bone (upper arm,
forearm, torso, thigh, shin, head) is moved by its own similarity transform
(rotation + uniform scale + translation) derived from the two joint endpoints.
A similarity transform cannot shear or stretch non-uniformly, so every limb keeps
its real proportions — the body is *re-posed*, not deformed.

Pipeline per frame:
  1. Build a body mask from the bone slabs, inpaint the climber out → clean wall.
  2. For each bone, cut its oriented pixel slab from the original frame and paste
     it onto the clean background via the bone's similarity transform.
  3. Painter's order (torso → legs → arms → head) keeps limbs layered naturally.

Deterministic, offline, cv2-only. Returns None when there aren't enough reliable
keypoints, so callers can fall back to a skeleton-only diagram.
"""

from __future__ import annotations

import math
from typing import Optional

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# (joint_a, joint_b, half-width as a fraction of the reference body width).
# Drawn back-to-front so nearer limbs overpaint farther ones.
_BONES = [
    ("torso",         0.62),   # shoulder-mid → hip-mid (special pseudo-joints)
    ("left_thigh",    0.30),
    ("right_thigh",   0.30),
    ("left_shin",     0.24),
    ("right_shin",    0.24),
    ("left_uarm",     0.26),
    ("right_uarm",    0.26),
    ("left_farm",     0.22),
    ("right_farm",    0.22),
]

_BONE_JOINTS = {
    "left_thigh":  ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shin":   ("left_knee", "left_ankle"),
    "right_shin":  ("right_knee", "right_ankle"),
    "left_uarm":   ("left_shoulder", "left_elbow"),
    "right_uarm":  ("right_shoulder", "right_elbow"),
    "left_farm":   ("left_elbow", "left_wrist"),
    "right_farm":  ("right_elbow", "right_wrist"),
}


def _mid(p, q):
    return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)


def _similarity(a, b, A, B):
    """2x3 similarity matrix mapping a→A and b→B (rotation+uniform scale+translate)."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    dlen = math.hypot(dx, dy)
    if dlen < 1e-6:
        return None
    DX, DY = B[0] - A[0], B[1] - A[1]
    Dlen = math.hypot(DX, DY)
    s = Dlen / dlen
    # rotation = angle(D) - angle(d)
    ang = math.atan2(DY, DX) - math.atan2(dy, dx)
    cos_t, sin_t = s * math.cos(ang), s * math.sin(ang)
    tx = A[0] - (cos_t * ax - sin_t * ay)
    ty = A[1] - (sin_t * ax + cos_t * ay)
    return np.float32([[cos_t, -sin_t, tx], [sin_t, cos_t, ty]])


def _slab_poly(A, B, half_w, ext):
    """Quad covering segment A-B with half-width *half_w*, extended *ext* past ends."""
    dx, dy = B[0] - A[0], B[1] - A[1]
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return None
    ux, uy = dx / L, dy / L           # along bone
    px, py = -uy, ux                  # perpendicular
    a0 = (A[0] - ux * ext, A[1] - uy * ext)
    b0 = (B[0] + ux * ext, B[1] + uy * ext)
    return np.int32([
        [a0[0] + px * half_w, a0[1] + py * half_w],
        [b0[0] + px * half_w, b0[1] + py * half_w],
        [b0[0] - px * half_w, b0[1] - py * half_w],
        [a0[0] - px * half_w, a0[1] - py * half_w],
    ])


def _endpoints(name, src, dst):
    """Return (src_a, src_b, dst_a, dst_b) for a bone, or None if unavailable."""
    if name == "torso":
        need = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if not all(src.get(k) and dst.get(k) for k in need):
            return None
        sa = _mid(src["left_shoulder"], src["right_shoulder"])
        sb = _mid(src["left_hip"], src["right_hip"])
        da = _mid(dst["left_shoulder"], dst["right_shoulder"])
        db = _mid(dst["left_hip"], dst["right_hip"])
        return sa, sb, da, db
    ja, jb = _BONE_JOINTS[name]
    if not (src.get(ja) and src.get(jb) and dst.get(ja) and dst.get(jb)):
        return None
    return src[ja], src[jb], dst[ja], dst[jb]


def repose_person(
    image,
    src_norm: dict[str, Optional[tuple]],
    dst_norm: dict[str, Optional[tuple]],
):
    """Return *image* with the climber re-posed from src_norm to dst_norm.

    Both dicts map keypoint name → normalized (x, y) (or None). Returns a BGR
    image the same size as *image*, or None if cv2 is missing / too few joints.
    """
    if not _HAS_CV2:
        return None
    h, w = image.shape[:2]

    def to_px(d):
        return {k: (v[0] * w, v[1] * h) for k, v in d.items() if v is not None}

    src = to_px(src_norm)
    dst = to_px(dst_norm)
    if not all(src.get(k) for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")):
        return None

    # Reference body width (shoulder span) sets all slab widths.
    ref = math.hypot(
        src["left_shoulder"][0] - src["right_shoulder"][0],
        src["left_shoulder"][1] - src["right_shoulder"][1],
    )
    ref = max(ref, 0.06 * w)

    # --- 1. Inpaint the climber out of the frame for a clean background. ---
    person_mask = np.zeros((h, w), dtype=np.uint8)
    for name, hw_frac in _BONES:
        ep = _endpoints(name, src, dst)
        if ep is None:
            continue
        poly = _slab_poly(ep[0], ep[1], ref * hw_frac, ref * 0.35)
        if poly is not None:
            cv2.fillConvexPoly(person_mask, poly, 255)
    # Head blob around the nose / above shoulder-mid.
    sc = _mid(src["left_shoulder"], src["right_shoulder"])
    head_c = src.get("nose") or (sc[0], sc[1] - ref * 0.6)
    cv2.circle(person_mask, (int(head_c[0]), int(head_c[1])), int(ref * 0.55), 255, -1)
    person_mask = cv2.dilate(person_mask, np.ones((9, 9), np.uint8), iterations=1)
    background = cv2.inpaint(image, person_mask, 4, cv2.INPAINT_TELEA)

    # --- 2. Paste each bone's real pixels onto the background via similarity. ---
    out = background.copy()

    def paste(sa, sb, da, db, half_w):
        M = _similarity(sa, sb, da, db)
        if M is None:
            return
        warped = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        poly = _slab_poly(da, db, half_w, ref * 0.35)
        if poly is None:
            return
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0
        np.copyto(out, warped, where=mask3)

    for name, hw_frac in _BONES:
        ep = _endpoints(name, src, dst)
        if ep is None:
            continue
        paste(ep[0], ep[1], ep[2], ep[3], ref * hw_frac)

    # --- 3. Head: move it by the torso transform. ---
    ep = _endpoints("torso", src, dst)
    if ep is not None:
        M = _similarity(ep[0], ep[1], ep[2], ep[3])
        if M is not None:
            warped = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101)
            hc = (M[0, 0] * head_c[0] + M[0, 1] * head_c[1] + M[0, 2],
                  M[1, 0] * head_c[0] + M[1, 1] * head_c[1] + M[1, 2])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (int(hc[0]), int(hc[1])), int(ref * 0.55), 255, -1)
            mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0
            np.copyto(out, warped, where=mask3)

    return out
