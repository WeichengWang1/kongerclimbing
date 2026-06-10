from __future__ import annotations

import math
from typing import Optional

from .base import BaseRule, Issue, WindowData, severity_from_m
from ...export.suggestions import get_issue_texts


class LockedElbowRule(BaseRule):
    code = "locked_elbow_too_early"
    label = "Locked elbow too early"

    def evaluate(self, window: WindowData, conf_threshold: float) -> Optional[Issue]:
        if self._reliable_frame_ratio(window) < 0.5:
            return None

        t = self.thresholds
        locked_angle: float = t["lockedElbowAngleDeg"]
        min_progress: float = t["minPostLockComProgress"]

        reliable_triples = [
            (f, m, c) for f, m, c in zip(window.frames, window.metrics, window.coms)
            if f.reliable
        ]
        if not reliable_triples:
            return None

        # Find first frame where any elbow is near-straight
        lock_idx: Optional[int] = None
        lock_metrics = None
        for i, (f, m, c) in enumerate(reliable_triples):
            angles = [a for a in (m.left_elbow_angle, m.right_elbow_angle) if a is not None]
            if any(a >= locked_angle for a in angles):
                lock_idx = i
                lock_metrics = m
                break

        if lock_idx is None or lock_metrics is None:
            return None

        # Require that CoM was actively moving BEFORE the lock.
        # Straight arms while stationary = resting (good technique), not a problem.
        min_pre_motion: float = t.get("minPreLockComMotion", 0.015)
        pre_lock_coms = [
            (c.x, c.y) for _, _, c in reliable_triples[:lock_idx + 1]
            if c and c.reliable
        ]
        if len(pre_lock_coms) >= 2:
            px = pre_lock_coms[-1][0] - pre_lock_coms[0][0]
            py = pre_lock_coms[-1][1] - pre_lock_coms[0][1]
            pre_motion = math.sqrt(px ** 2 + py ** 2)
            if pre_motion < min_pre_motion:
                return None  # climber was stationary before lock — resting, not a problem

        # CoM progress after lock point
        post_lock_coms = [
            (c.x, c.y) for _, _, c in reliable_triples[lock_idx:]
            if c and c.reliable
        ]
        if len(post_lock_coms) < 2:
            return None

        dx = post_lock_coms[-1][0] - post_lock_coms[0][0]
        dy = post_lock_coms[-1][1] - post_lock_coms[0][1]
        com_progress = math.sqrt(dx ** 2 + dy ** 2)

        if com_progress >= min_progress:
            return None

        m_val = min(1.0, 1.0 - com_progress / min_progress)

        # Record which elbow(s) triggered
        evidence: dict = {}
        if lock_metrics.left_elbow_angle and lock_metrics.left_elbow_angle >= locked_angle:
            evidence["leftElbowAngleDeg"] = round(lock_metrics.left_elbow_angle, 1)
        if lock_metrics.right_elbow_angle and lock_metrics.right_elbow_angle >= locked_angle:
            evidence["rightElbowAngleDeg"] = round(lock_metrics.right_elbow_angle, 1)
        evidence["postLockComProgress"] = round(com_progress, 4)

        avg_conf = self._window_avg_confidence(window)
        sev = severity_from_m(m_val)
        issue_conf = avg_conf * m_val
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
            evidence_metrics=evidence,
            message=msg,
            recommendation=rec,
        )
