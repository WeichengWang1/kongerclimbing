from dataclasses import dataclass, field
from typing import Optional

KEYPOINT_NAMES: list[str] = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

CORE_KEYPOINTS: frozenset[str] = frozenset({
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
})


@dataclass
class Keypoint:
    name: str
    x: float   # normalized 0-1
    y: float   # normalized 0-1
    confidence: float
    reliable: bool = True

    def to_dict(self) -> dict:
        return {"name": self.name, "x": self.x, "y": self.y, "confidence": self.confidence}


@dataclass
class PoseFrame:
    frame_index: int            # sequential index in sampled frames
    original_frame_index: int   # original video frame number
    timestamp_ms: int
    keypoints: list[Keypoint] = field(default_factory=list)
    reliable: bool = True

    def get(self, name: str) -> Optional[Keypoint]:
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None

    def core_confidence(self, threshold: float = 0.0) -> float:
        scores = [
            kp.confidence for kp in self.keypoints
            if kp.name in CORE_KEYPOINTS and kp.confidence >= threshold
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self, metrics_dict: Optional[dict] = None, com_dict: Optional[dict] = None) -> dict:
        d: dict = {
            "frameIndex": self.frame_index,
            "originalFrameIndex": self.original_frame_index,
            "timestampMs": self.timestamp_ms,
            "reliable": self.reliable,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
        }
        if com_dict is not None:
            d["centerOfMass"] = com_dict
        if metrics_dict is not None:
            d["metrics"] = metrics_dict
        return d
