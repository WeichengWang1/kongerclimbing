from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterator

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


@dataclass
class VideoMeta:
    video_path: str
    duration_ms: int
    fps: float
    width: int
    height: int
    total_frames: int


@dataclass
class FrameData:
    image: object   # cv2.Mat — untyped to avoid hard import at module level
    original_frame_index: int
    timestamp_ms: int


def load_video_meta(path: str) -> VideoMeta:
    if not os.path.exists(path):
        print(f"Video not found: {path}", file=sys.stderr)
        sys.exit(2)

    if cv2 is None:
        print("opencv-python is not installed. Run: pip install opencv-python", file=sys.stderr)
        sys.exit(3)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Cannot open video (file may be corrupt or ffmpeg missing): {path}", file=sys.stderr)
        cap.release()
        sys.exit(3)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ms = round(total_frames / fps * 1000) if fps > 0 else 0
    cap.release()

    supported_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    if os.path.splitext(path)[1].lower() not in supported_exts:
        print(f"Unsupported video format: {path}", file=sys.stderr)
        sys.exit(2)

    return VideoMeta(
        video_path=path,
        duration_ms=duration_ms,
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
    )


def extract_frames(meta: VideoMeta, target_fps: float) -> Iterator[FrameData]:
    if cv2 is None:
        print("opencv-python is not installed.", file=sys.stderr)
        sys.exit(3)

    step = max(1, round(meta.fps / target_fps))
    cap = cv2.VideoCapture(meta.video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {meta.video_path}", file=sys.stderr)
        sys.exit(3)

    orig_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if orig_idx % step == 0:
                yield FrameData(
                    image=frame,
                    original_frame_index=orig_idx,
                    timestamp_ms=round(orig_idx / meta.fps * 1000),
                )
            orig_idx += 1
    finally:
        cap.release()
