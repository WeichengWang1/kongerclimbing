from __future__ import annotations

import math
from typing import Optional

from .base import BaseRule, Issue, WindowData, severity_from_m
from ...export.suggestions import get_issue_texts


class OverPullingRule(BaseRule):
    code = "over_pulling_with_arms"
    label = "Over-pulling with arms"

    def evaluate(self, window: WindowData, conf_threshold: float) -> Optional[Issue]:
        if self._reliable_frame_ratio(window) < 0.5:
            return None

        t = self.thresholds
        elbow_delta_thresh: float = t["elbowAngleDeltaDeg"]
        min_hip_disp: float = t["minHipDisplacement"]
        min_knee_delta: float = t["minKneeAngleDeltaDeg"]
        full_scale: float = t.get("fullScaleDeg", 30)

        reliable = [
            (m, f) for m, f in zip(window.metrics, window.frames) if f.reliable
        ]
        if not reliable:
            return None

        def avg_elbow(m):
            vals = [v for v in (m.left_elbow_angle, m.right_elbow_angle) if v is not None]
            return sum(vals) / len(vals) if vals else None

        first_m = reliable[0][0]
        last_m = reliable[-1][0]
        first_e = avg_elbow(first_m)
        last_e = avg_elbow(last_m)
        if first_e is None or last_e is None:
            return None

        elbow_delta = last_e - first_e  # negative = arm bending/pulling

        hip_disp = self._compute_hip_displacement(window)

        def avg_knee(m):
            vals = [v for v in (m.left_knee_angle, m.right_knee_angle) if v is not None]
            return sum(vals) / len(vals) if vals else None

        first_k = avg_knee(first_m)
        last_k = avg_knee(last_m)
        knee_delta = (last_k - first_k) if (first_k and last_k) else 0.0

        if not (elbow_delta < elbow_delta_thresh and hip_disp < min_hip_disp and knee_delta < min_knee_delta):
            return None

        overshoot = abs(elbow_delta) - abs(elbow_delta_thresh)
        m = min(1.0, max(0.0, overshoot / full_scale))

        avg_conf = self._window_avg_confidence(window)
        sev = severity_from_m(m)
        issue_conf = avg_conf * m
        msg, rec = get_issue_texts(self.code, sev, issue_conf)

        return Issue(
            id="",
            code=self.code,
            label=self.label,
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            confidence=issue_conf,
            severity=sev,
            primary_frame_index=window.primary_frame_index,
            evidence_metrics={
                "elbowAngleDeltaDeg": round(elbow_delta, 1),
                "hipDisplacement": round(hip_disp, 4),
                "kneeAngleDeltaDeg": round(knee_delta, 1),
            },
            message=msg,
            recommendation=rec,
        )

    def _compute_hip_displacement(self, window: WindowData) -> float:
        positions = []
        for f in window.frames:
            if not f.reliable:
                continue
            lh, rh = f.get("left_hip"), f.get("right_hip")
            if lh and rh:
                positions.append(((lh.x + rh.x) / 2, (lh.y + rh.y) / 2))
        if len(positions) < 2:
            return 0.0
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        return math.sqrt(dx ** 2 + dy ** 2)
