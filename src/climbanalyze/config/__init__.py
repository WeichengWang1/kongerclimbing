from dataclasses import dataclass, field
import json

_RULE_DEFAULTS: dict[str, dict] = {
    "over_pulling_with_arms": {
        "elbowAngleDeltaDeg": -25,
        "minHipDisplacement": 0.02,
        "minKneeAngleDeltaDeg": 5,
        "fullScaleDeg": 30,
    },
    "unstable_center_of_mass": {
        "maxDirectionChanges": 3,
        "maxVelocityVariance": 0.005,   # raised: normal climbing produces high variance
    },
    "poor_foot_engagement": {
        "maxAnkleJitter": 0.08,         # raised: feet naturally move between holds
        "minComShiftToSupport": 0.015,
    },
    "locked_elbow_too_early": {
        "lockedElbowAngleDeg": 165,
        "minPostLockComProgress": 0.02,
        "minPreLockComMotion": 0.015,   # require active movement before lock to avoid resting false-positives
    },
}


@dataclass
class AnalysisConfig:
    poseModel: str = "yolo11x-pose"
    device: str = "auto"
    targetFps: float = 10.0
    keypointConfidenceThreshold: float = 0.4
    frameConfidenceThreshold: float = 0.5
    smoothingWindow: int = 5
    analysisWindowFrames: int = 12
    maxAnnotatedFrames: int = 20
    language: str = "en"
    ruleThresholds: dict = field(
        default_factory=lambda: {k: dict(v) for k, v in _RULE_DEFAULTS.items()}
    )

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisConfig":
        cfg = cls()
        for key in (
            "poseModel", "device", "targetFps", "keypointConfidenceThreshold",
            "frameConfidenceThreshold", "smoothingWindow", "analysisWindowFrames",
            "maxAnnotatedFrames", "language",
        ):
            if key in data:
                setattr(cfg, key, data[key])
        if "ruleThresholds" in data:
            for code, thresholds in data["ruleThresholds"].items():
                if code in cfg.ruleThresholds:
                    cfg.ruleThresholds[code].update(thresholds)
                else:
                    cfg.ruleThresholds[code] = thresholds
        return cfg

    @classmethod
    def from_file(cls, path: str) -> "AnalysisConfig":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def to_dict(self) -> dict:
        return {
            "poseModel": self.poseModel,
            "device": self.resolve_device(),
            "targetFps": self.targetFps,
            "keypointConfidenceThreshold": self.keypointConfidenceThreshold,
            "frameConfidenceThreshold": self.frameConfidenceThreshold,
            "smoothingWindow": self.smoothingWindow,
            "analysisWindowFrames": self.analysisWindowFrames,
            "maxAnnotatedFrames": self.maxAnnotatedFrames,
            "language": self.language,
            "ruleThresholds": self.ruleThresholds,
        }
