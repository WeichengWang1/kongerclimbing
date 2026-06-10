from __future__ import annotations

import math
from typing import Optional

from .base import BaseRule, Issue, WindowData, severity_from_m
from ...export.suggestions import get_issue_texts


class UnstableCoMRule(BaseRule):
    code = "unstable_center_of_mass"
    label = "Unstable center of mass"

    def evaluate(self, window: WindowData, conf_threshold: float) -> Optional[Issue]:
        if self._reliable_frame_ratio(window) < 0.5:
            return None

        t = self.thresholds
        max_dir_changes: int = t["maxDirectionChanges"]
        max_vel_var: float = t["maxVelocityVariance"]

        # Gather reliable CoM+frame pairs
        pairs = [
            (c, f) for c, f in zip(window.coms, window.frames)
            if c and c.reliable and f.reliable
        ]
        if len(pairs) < 3:
            return None

        # Velocity vectors between consecutive reliable frames
        velocities: list[tuple[float, float]] = []
        for i in range(1, len(pairs)):
            c_prev, f_prev = pairs[i - 1]
            c_curr, f_curr = pairs[i]
            dt = (f_curr.timestamp_ms - f_prev.timestamp_ms) / 1000.0
            if dt <= 0:
                continue
            velocities.append(((c_curr.x - c_prev.x) / dt, (c_curr.y - c_prev.y) / dt))

        if len(velocities) < 2:
            return None

        # Direction changes: consecutive vectors with dot product < 0 (angle > 90°)
        dir_changes = sum(
            1 for i in range(1, len(velocities))
            if velocities[i - 1][0] * velocities[i][0] + velocities[i - 1][1] * velocities[i][1] < 0
        )

        speeds = [math.sqrt(v[0] ** 2 + v[1] ** 2) for v in velocities]
        mean_speed = sum(speeds) / len(speeds)
        variance = sum((s - mean_speed) ** 2 for s in speeds) / len(speeds)

        # Require at least one direction reversal — pure variance without reversals is
        # just normal climbing momentum, not erratic movement.
        if dir_changes == 0:
            return None
        if not (dir_changes >= max_dir_changes or variance >= max_vel_var):
            return None

        m_dir = min(1.0, dir_changes / (max_dir_changes * 2)) if dir_changes >= max_dir_changes else 0.0
        m_var = min(1.0, variance / (max_vel_var * 2)) if variance >= max_vel_var else 0.0
        m = max(m_dir, m_var)

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
                "directionChanges": dir_changes,
                "velocityVariance": round(variance, 6),
            },
            message=msg,
            recommendation=rec,
        )
