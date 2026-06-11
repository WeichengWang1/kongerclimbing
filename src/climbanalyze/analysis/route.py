"""Reconstruct the *climbed* route from detected holds + climber contacts.

Detected holds (``analysis.holds``) are an unordered candidate map of every hold
on the wall. The route the climber actually used is derived from the climber: a
hand or foot that stabilizes (low velocity) near a hold is grasping/standing on
it. Snapping those stabilizations to holds yields an ordered contact timeline and,
for any moment, the hold a moving hand is reaching toward.

This makes posture judgment and correction route-aware without manual labeling.
Everything is normalized (0-1, y-down). Degrades to an empty route when holds are
unavailable, in which case all queries return None and callers fall back to
pose-only behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..pose.schema import PoseFrame
from .holds import Hold

_HANDS = ("left_wrist", "right_wrist")
_FEET = ("left_ankle", "right_ankle")
_LIMBS = _HANDS + _FEET

# Defaults (overridable via config["route"]).
_VEL_THRESH = 0.12          # normalized units/sec below which a limb counts as stable
_MIN_STABLE_FRAMES = 3      # consecutive stable frames to register a contact
_MAX_SNAP_DIST = 0.10       # max normalized dist from limb to a hold to snap
_MAX_REACH_DIST = 0.45      # max normalized dist a hand reaches to a target hold
_REACH_CONE = 1.2           # |dx| <= _REACH_CONE * dy_up  → within the "above" cone


@dataclass
class Contact:
    limb: str
    hold_index: int
    start_ms: int
    end_ms: int
    x: float
    y: float

    def to_dict(self) -> dict:
        return {
            "limb": self.limb,
            "holdIndex": self.hold_index,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
        }


@dataclass
class Route:
    holds: list[Hold] = field(default_factory=list)
    contacts: list[Contact] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "holds": [h.to_dict() for h in self.holds],
            "contacts": [c.to_dict() for c in self.contacts],
        }

    def hold_pos(self, i: int) -> Optional[tuple[float, float]]:
        if 0 <= i < len(self.holds):
            return (self.holds[i].x, self.holds[i].y)
        return None

    def nearest_hold(
        self, x: float, y: float, max_dist: float = _MAX_SNAP_DIST
    ) -> Optional[int]:
        best_i, best_d = None, max_dist
        for i, hd in enumerate(self.holds):
            d = math.hypot(hd.x - x, hd.y - y)
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def reach_target(
        self, x: float, y: float,
        max_dist: float = _MAX_REACH_DIST, cone: float = _REACH_CONE,
    ) -> Optional[tuple[float, float]]:
        """The hold a hand at (x, y) is most likely reaching toward.

        Nearest hold that is above the hand (smaller y) within an upward cone and
        within *max_dist*. Returns the hold center, or None.
        """
        best, best_d = None, max_dist
        for hd in self.holds:
            dy_up = y - hd.y          # positive => hold is above the hand
            if dy_up <= 0.01:
                continue
            if abs(hd.x - x) > cone * dy_up:
                continue
            d = math.hypot(hd.x - x, hd.y - y)
            if d < best_d:
                best, best_d = (hd.x, hd.y), d
        return best

    def support_holds(self, frame_ms: int) -> list[tuple[str, tuple[float, float]]]:
        """Holds in contact at *frame_ms* as (limb, (x, y))."""
        out = []
        for c in self.contacts:
            if c.start_ms <= frame_ms <= c.end_ms:
                pos = self.hold_pos(c.hold_index)
                if pos:
                    out.append((c.limb, pos))
        return out


def _xy(frame: PoseFrame, name: str, conf_threshold: float) -> Optional[tuple[float, float]]:
    kp = frame.get(name)
    if not kp or kp.confidence < conf_threshold or not kp.reliable:
        return None
    if kp.x < 0.01 and kp.y < 0.01:   # YOLO occluded-keypoint artifact
        return None
    return (kp.x, kp.y)


def reconstruct_route(
    frames: list[PoseFrame],
    holds: list[Hold],
    conf_threshold: float,
    config: Optional[dict] = None,
) -> Route:
    """Build a Route by snapping low-velocity limb stabilizations to holds."""
    route = Route(holds=list(holds))
    if not holds or len(frames) < 2:
        return route

    cfg = config or {}
    vel_thresh = float(cfg.get("velThresh", _VEL_THRESH))
    min_stable = int(cfg.get("minStableFrames", _MIN_STABLE_FRAMES))
    max_snap = float(cfg.get("maxSnapDist", _MAX_SNAP_DIST))

    for limb in _LIMBS:
        run_start: Optional[int] = None
        run_pts: list[tuple[float, float]] = []

        for i in range(len(frames)):
            p = _xy(frames[i], limb, conf_threshold)
            speed = None
            if p is not None and i > 0:
                q = _xy(frames[i - 1], limb, conf_threshold)
                if q is not None:
                    dt = (frames[i].timestamp_ms - frames[i - 1].timestamp_ms) / 1000.0
                    if dt > 0:
                        speed = math.hypot(p[0] - q[0], p[1] - q[1]) / dt

            stable = p is not None and speed is not None and speed <= vel_thresh
            if stable:
                if run_start is None:
                    run_start = i
                    run_pts = []
                run_pts.append(p)
            else:
                _flush_contact(route, limb, frames, run_start, i - 1, run_pts,
                               min_stable, max_snap)
                run_start, run_pts = None, []

        _flush_contact(route, limb, frames, run_start, len(frames) - 1, run_pts,
                       min_stable, max_snap)

    route.contacts.sort(key=lambda c: c.start_ms)
    return route


def _flush_contact(
    route: Route, limb: str, frames: list[PoseFrame],
    start_i: Optional[int], end_i: int, pts: list[tuple[float, float]],
    min_stable: int, max_snap: float,
) -> None:
    if start_i is None or len(pts) < min_stable:
        return
    ax = sum(p[0] for p in pts) / len(pts)
    ay = sum(p[1] for p in pts) / len(pts)
    hi = route.nearest_hold(ax, ay, max_snap)
    if hi is None:
        return
    route.contacts.append(Contact(
        limb=limb,
        hold_index=hi,
        start_ms=frames[start_i].timestamp_ms,
        end_ms=frames[end_i].timestamp_ms,
        x=ax, y=ay,
    ))
