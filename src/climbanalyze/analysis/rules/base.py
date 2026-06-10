from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ...pose.schema import PoseFrame
from ..center_of_mass import CenterOfMass
from ..metrics import FrameMetrics


@dataclass
class Issue:
    id: str
    code: str
    label: str
    start_ms: int
    end_ms: int
    confidence: float
    severity: str   # low | medium | high
    primary_frame_index: int
    evidence_metrics: dict
    message: str
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "code": self.code,
            "label": self.label,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "confidence": round(self.confidence, 4),
            "severity": self.severity,
            "evidence": {
                "primaryFrameIndex": self.primary_frame_index,
                "metrics": self.evidence_metrics,
            },
            "message": self.message,
            "recommendation": self.recommendation,
        }


def severity_from_m(m: float) -> str:
    if m < 0.33:
        return "low"
    if m < 0.66:
        return "medium"
    return "high"


@dataclass
class WindowData:
    frames: list[PoseFrame]
    coms: list[Optional[CenterOfMass]]
    metrics: list[FrameMetrics]
    start_ms: int
    end_ms: int
    primary_frame_index: int


class BaseRule(ABC):
    def __init__(self, thresholds: dict) -> None:
        self.thresholds = thresholds

    @property
    @abstractmethod
    def code(self) -> str: ...

    @property
    @abstractmethod
    def label(self) -> str: ...

    @abstractmethod
    def evaluate(self, window: WindowData, conf_threshold: float) -> Optional[Issue]: ...

    def _reliable_frame_ratio(self, window: WindowData) -> float:
        if not window.frames:
            return 0.0
        return sum(1 for f in window.frames if f.reliable) / len(window.frames)

    def _window_avg_confidence(self, window: WindowData) -> float:
        confs: list[float] = []
        for f in window.frames:
            if f.reliable:
                c = f.core_confidence()
                if c > 0:
                    confs.append(c)
        for com in window.coms:
            if com and com.reliable:
                confs.append(com.confidence)
        return sum(confs) / len(confs) if confs else 0.0
