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
