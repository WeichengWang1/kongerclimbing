from collections import deque

from .schema import Keypoint, PoseFrame, KEYPOINT_NAMES


class MovingAverageSmoother:
    def __init__(self, window: int) -> None:
        self._window = window
        self._buffers: dict[str, deque[tuple[float, float, float]]] = {
            name: deque(maxlen=window) for name in KEYPOINT_NAMES
        }

    def smooth(self, frame: PoseFrame) -> PoseFrame:
        for kp in frame.keypoints:
            if kp.name in self._buffers:
                self._buffers[kp.name].append((kp.x, kp.y, kp.confidence))

        smoothed: list[Keypoint] = []
        for kp in frame.keypoints:
            buf = self._buffers.get(kp.name)
            if not buf:
                smoothed.append(kp)
                continue
            n = len(buf)
            smoothed.append(
                Keypoint(
                    name=kp.name,
                    x=sum(p[0] for p in buf) / n,
                    y=sum(p[1] for p in buf) / n,
                    confidence=sum(p[2] for p in buf) / n,
                    reliable=kp.reliable,
                )
            )

        return PoseFrame(
            frame_index=frame.frame_index,
            original_frame_index=frame.original_frame_index,
            timestamp_ms=frame.timestamp_ms,
            keypoints=smoothed,
            reliable=frame.reliable,
        )

    def reset(self) -> None:
        for buf in self._buffers.values():
            buf.clear()


class RenderSmoother:
    """Single-frame spike filter for rendering only.

    When a keypoint jumps more than *max_jump* (normalized) in one frame,
    it is replaced with the previous accepted position. This rejects the
    occasional high-confidence-but-wrong YOLO detection (e.g. elbow snapping
    to the shoulder when the arm is raised) without introducing the temporal
    lag of a full moving average.

    Only used for annotation. Analysis uses MovingAverageSmoother.
    """

    def __init__(self, max_jump: float = 0.12) -> None:
        self._max_jump = max_jump
        self._prev: dict[str, tuple[float, float, float]] = {}  # name → (x, y, conf)

    def smooth(self, frame: PoseFrame) -> PoseFrame:
        filtered: list[Keypoint] = []
        for kp in frame.keypoints:
            prev = self._prev.get(kp.name)
            if prev is not None:
                dx, dy = kp.x - prev[0], kp.y - prev[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > self._max_jump:
                    # Spike: keep previous accepted position
                    filtered.append(
                        Keypoint(
                            name=kp.name,
                            x=prev[0],
                            y=prev[1],
                            confidence=prev[2],
                            reliable=kp.reliable,
                        )
                    )
                    continue
            self._prev[kp.name] = (kp.x, kp.y, kp.confidence)
            filtered.append(kp)

        return PoseFrame(
            frame_index=frame.frame_index,
            original_frame_index=frame.original_frame_index,
            timestamp_ms=frame.timestamp_ms,
            keypoints=filtered,
            reliable=frame.reliable,
        )

    def reset(self) -> None:
        self._prev.clear()
