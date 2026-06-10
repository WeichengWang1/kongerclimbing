from __future__ import annotations

import json
import os
from typing import Optional

from ..analysis.rules.base import Issue
from ..analysis.center_of_mass import CenterOfMass
from ..analysis.metrics import FrameMetrics
from ..config import AnalysisConfig
from ..pose.schema import PoseFrame
from ..video.ingestion import VideoMeta


def build_result(
    meta: VideoMeta,
    config: AnalysisConfig,
    frames: list[PoseFrame],
    coms: list[Optional[CenterOfMass]],
    all_metrics: list[FrameMetrics],
    issues: list[Issue],
    annotated_paths: list[tuple[str, str]],   # [(issue_id, relative_path), ...]
    warnings: list[str],
) -> dict:
    reliable_count = sum(1 for f in frames if f.reliable)
    total = len(frames)

    if total == 0:
        status = "no_person_detected"
    elif reliable_count == 0:
        status = "no_person_detected"
    elif reliable_count / total < 0.5:
        status = "low_quality"
    else:
        status = "ok"

    frames_out = []
    for f, com, m in zip(frames, coms, all_metrics):
        com_dict = com.to_dict() if com else None
        frames_out.append(f.to_dict(metrics_dict=m.to_dict(), com_dict=com_dict))

    return {
        "version": "1.0",
        "coordinateSpace": "normalized",
        "status": status,
        "warnings": warnings,
        "source": {
            "videoPath": meta.video_path,
            "durationMs": meta.duration_ms,
            "fps": meta.fps,
            "width": meta.width,
            "height": meta.height,
        },
        "config": config.to_dict(),
        "frames": frames_out,
        "issues": [iss.to_dict() for iss in issues],
        "artifacts": {
            "annotatedFrames": [
                {"issueId": issue_id, "path": path}
                for issue_id, path in annotated_paths
            ]
        },
    }


def write_result(result: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "analysis.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path
