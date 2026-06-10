from __future__ import annotations

import math
from typing import Optional

from .base import BaseRule, Issue, WindowData, severity_from_m
from ...export.suggestions import get_issue_texts


class PoorFootEngagementRule(BaseRule):
    code = "poor_foot_engagement"
    label = "Poor foot engagement"

    def evaluate(self, window: WindowData, conf_threshold: float) -> Optional[Issue]:
        if self._reliable_frame_ratio(window) < 0.5:
            return None

        t = self.thresholds
        max_jitter: float = t["maxAnkleJitter"]
        min_com_shift: float = t["minComShiftToSupport"]

        reliable_pairs = [
            (f, c) for f, c in zip(window.frames, window.coms)
            if f.reliable and c and c.reliable
        ]
        if len(reliable_pairs) < 2:
            return None

        # Ankle jitter per side
        ankle_positions: dict[str, list[tuple[float, float]]] = {
            "left_ankle": [], "right_ankle": [],
        }
        for f, _ in reliable_pairs:
            for side in ankle_positions:
                kp = f.get(side)
                if kp and kp.confidence >= conf_threshold:
                    ankle_positions[side].append((kp.x, kp.y))

        jitters: dict[str, float] = {}
        support_side: Optional[str] = None
        max_mean_y = -1.0

        for side, pts in ankle_positions.items():
            if len(pts) < 2:
                continue
            mx = sum(p[0] for p in pts) / len(pts)
            my = sum(p[1] for p in pts) / len(pts)
            std = math.sqrt(sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in pts) / len(pts))
            jitters[side] = std
            # Support foot = lowest (highest y in image coords)
            if my > max_mean_y:
                max_mean_y = my
                support_side = side

        if not jitters:
            return None

        ankle_jitter = max(jitters.values())

        # CoM horizontal shift toward support side
        com_shift = 0.0
        if support_side:
            first_com = reliable_pairs[0][1]
            last_com = reliable_pairs[-1][1]
            # positive = CoM moved toward the support side
            direction = 1.0 if support_side == "left_ankle" else -1.0
            com_shift = direction * (last_com.x - first_com.x)

        if not (ankle_jitter > max_jitter and com_shift < min_com_shift):
            return None

        m = min(1.0, (ankle_jitter - max_jitter) / max_jitter)
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
                "ankleJitter": round(ankle_jitter, 4),
                "comShiftToSupport": round(com_shift, 4),
            },
            message=msg,
            recommendation=rec,
        )
