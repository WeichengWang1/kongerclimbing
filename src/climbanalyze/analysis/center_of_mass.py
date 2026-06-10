from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..pose.schema import Keypoint, PoseFrame

_LIMB_NAMES = (
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

_BASE_WEIGHTS = {"shoulders": 0.35, "hips": 0.45, "limbs": 0.20}


@dataclass
class CenterOfMass:
    x: float
    y: float
    confidence: float
    reliable: bool = True

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "confidence": self.confidence}


def compute_center_of_mass(
    frame: PoseFrame,
    conf_threshold: float = 0.0,
) -> Optional[CenterOfMass]:
    def valid(name: str) -> Optional[Keypoint]:
        kp = frame.get(name)
        if not kp or kp.confidence < conf_threshold:
            return None
        if kp.x < 0.01 and kp.y < 0.01:  # YOLO (0,0) artifact for occluded keypoints
            return None
        return kp

    segments: dict[str, tuple[float, float]] = {}
    seg_conf: dict[str, float] = {}

    ls, rs = valid("left_shoulder"), valid("right_shoulder")
    if ls and rs:
        segments["shoulders"] = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
        seg_conf["shoulders"] = (ls.confidence + rs.confidence) / 2

    lh, rh = valid("left_hip"), valid("right_hip")
    if lh and rh:
        segments["hips"] = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)
        seg_conf["hips"] = (lh.confidence + rh.confidence) / 2

    limb_kps = [kp for name in _LIMB_NAMES if (kp := valid(name)) is not None]
    if limb_kps:
        segments["limbs"] = (
            sum(kp.x for kp in limb_kps) / len(limb_kps),
            sum(kp.y for kp in limb_kps) / len(limb_kps),
        )
        seg_conf["limbs"] = sum(kp.confidence for kp in limb_kps) / len(limb_kps)

    if not segments:
        return None

    reliable = "shoulders" in segments and "hips" in segments

    # Renormalize weights to present segments only
    total_base = sum(_BASE_WEIGHTS[k] for k in segments)
    norm_w = {k: _BASE_WEIGHTS[k] / total_base for k in segments}

    x = sum(norm_w[k] * segments[k][0] for k in segments)
    y = sum(norm_w[k] * segments[k][1] for k in segments)
    conf = sum(norm_w[k] * seg_conf[k] for k in segments)

    return CenterOfMass(x=x, y=y, confidence=conf, reliable=reliable)
