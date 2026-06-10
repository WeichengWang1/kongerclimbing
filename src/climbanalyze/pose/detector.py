from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Optional

from .schema import Keypoint, PoseFrame
from .adapter import COCO17_TO_SCHEMA

# COCO-17 indices for core keypoints used in primary-climber selection
_COCO_CORE_INDICES = [5, 6, 11, 12, 13, 14]


class PoseDetector(ABC):
    @abstractmethod
    def detect(
        self,
        image,
        frame_index: int,
        original_frame_index: int,
        timestamp_ms: int,
    ) -> PoseFrame: ...

    @abstractmethod
    def close(self) -> None: ...


class YoloPoseDetector(PoseDetector):
    def __init__(self, model_name: str, device: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            print(
                "ultralytics is not installed.\n"
                "Create a Python 3.12 venv and run: pip install -r requirements.txt",
                file=sys.stderr,
            )
            sys.exit(3)
        self._model = YOLO(model_name)
        self._device = device
        self._prev_center: Optional[tuple[float, float]] = None

    def detect(
        self,
        image,
        frame_index: int,
        original_frame_index: int,
        timestamp_ms: int,
    ) -> PoseFrame:
        results = self._model(image, device=self._device, verbose=False)

        empty = PoseFrame(
            frame_index=frame_index,
            original_frame_index=original_frame_index,
            timestamp_ms=timestamp_ms,
            keypoints=[],
            reliable=False,
        )

        if not results:
            return empty

        result = results[0]
        if result.keypoints is None or len(result.keypoints.data) == 0:
            return empty

        h, w = image.shape[:2]
        person_idx = _select_primary_climber(result, self._prev_center)

        # Update previous center for continuity heuristic
        if result.boxes is not None and person_idx < len(result.boxes):
            box = result.boxes.xyxyn[person_idx].cpu().tolist()
            self._prev_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

        kps_data = result.keypoints.data[person_idx].cpu().numpy()  # (17, 3): x_px, y_px, conf

        keypoints = [
            Keypoint(
                name=name,
                x=float(kps_data[coco_idx][0]) / w,
                y=float(kps_data[coco_idx][1]) / h,
                confidence=float(kps_data[coco_idx][2]),
            )
            for name, coco_idx in COCO17_TO_SCHEMA.items()
        ]

        return PoseFrame(
            frame_index=frame_index,
            original_frame_index=original_frame_index,
            timestamp_ms=timestamp_ms,
            keypoints=keypoints,
            reliable=True,
        )

    def close(self) -> None:
        pass


def _select_primary_climber(
    result,
    prev_center: Optional[tuple[float, float]],
) -> int:
    if result.boxes is None or len(result.boxes) == 0:
        return 0

    n = len(result.boxes)
    if n == 1:
        return 0

    boxes = result.boxes.xyxyn.cpu().numpy()  # (N, 4) normalized
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    max_area = max(areas)

    candidates = [i for i, a in enumerate(areas) if max_area - a < 0.05 * max_area]
    if len(candidates) == 1:
        return candidates[0]

    # Tiebreak 1: core keypoint average confidence
    if result.keypoints is not None and len(result.keypoints.data) > max(candidates):
        kps = result.keypoints.data.cpu().numpy()

        def avg_core(idx: int) -> float:
            confs = [kps[idx][ci][2] for ci in _COCO_CORE_INDICES if kps[idx][ci][2] > 0]
            return sum(confs) / len(confs) if confs else 0.0

        ranked = sorted(candidates, key=avg_core, reverse=True)
        if avg_core(ranked[0]) - avg_core(ranked[1]) > 0.05:
            return ranked[0]
        candidates = ranked

    # Tiebreak 2: proximity to previous frame center
    if prev_center is not None:
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        return min(
            candidates,
            key=lambda i: (centers[i][0] - prev_center[0]) ** 2
            + (centers[i][1] - prev_center[1]) ** 2,
        )

    return candidates[0]
