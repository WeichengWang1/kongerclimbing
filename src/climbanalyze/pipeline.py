from __future__ import annotations

import os
import sys
from typing import Optional

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from .config import AnalysisConfig
from .video.ingestion import VideoMeta, load_video_meta, extract_frames
from .pose.detector import YoloPoseDetector
from .pose.smoother import MovingAverageSmoother
from .pose.schema import PoseFrame
from .analysis.center_of_mass import CenterOfMass, compute_center_of_mass
from .analysis.metrics import FrameMetrics, compute_frame_metrics, fill_com_velocity
from .analysis.rules import ALL_RULES, Issue, WindowData
from .rendering.annotator import annotate_frame
from .export.json_export import build_result, write_result


def run(video_path: str, config: AnalysisConfig, output_dir: str = "outputs") -> dict:
    warnings: list[str] = []

    # Resolve device; warn on fallback
    resolved_device = config.resolve_device()
    if config.device == "auto" and resolved_device == "cpu":
        warnings.append("MPS not available; falling back to CPU.")
    elif config.device == "mps" and resolved_device != "mps":
        warnings.append("Requested device 'mps' not available; falling back to CPU.")
        resolved_device = "cpu"

    meta = load_video_meta(video_path)

    detector = YoloPoseDetector(config.poseModel, resolved_device)
    smoother = MovingAverageSmoother(config.smoothingWindow)

    frames: list[PoseFrame] = []
    coms: list[Optional[CenterOfMass]] = []
    all_metrics: list[FrameMetrics] = []
    raw_images: list = []  # kept for annotation export

    # --- Frame-by-frame processing ---
    for seq_idx, fd in enumerate(extract_frames(meta, config.targetFps)):
        pose = detector.detect(
            fd.image, seq_idx, fd.original_frame_index, fd.timestamp_ms
        )

        # Mark keypoints below threshold as unreliable
        for kp in pose.keypoints:
            if kp.confidence < config.keypointConfidenceThreshold:
                kp.reliable = False

        # Frame-level reliability
        core_conf = pose.core_confidence(config.keypointConfidenceThreshold)
        if not pose.keypoints or core_conf < config.frameConfidenceThreshold:
            pose.reliable = False

        pose = smoother.smooth(pose)
        com = compute_center_of_mass(pose, config.keypointConfidenceThreshold)
        metrics = compute_frame_metrics(pose, config.keypointConfidenceThreshold)

        frames.append(pose)
        coms.append(com)
        all_metrics.append(metrics)
        raw_images.append(fd.image)

    detector.close()

    # Back-fill per-frame CoM velocity and direction change (requires full frame list)
    fill_com_velocity(
        all_metrics, coms,
        [f.timestamp_ms for f in frames],
    )

    # --- Exit early if no persons detected ---
    if not frames or all(not f.reliable for f in frames):
        result = build_result(
            meta, config, frames, coms, all_metrics, [], [], warnings
        )
        result["status"] = "no_person_detected"
        write_result(result, output_dir)
        sys.exit(4)

    # Warn if quality is low
    reliable_ratio = sum(1 for f in frames if f.reliable) / len(frames)
    if reliable_ratio < 0.5:
        warnings.append(
            f"Only {reliable_ratio:.0%} of frames are reliable. "
            "Analysis results may be less accurate."
        )

    # --- Sliding-window rule evaluation ---
    _WARMUP_MS = 1500  # skip issues in the first 1.5 s (climber getting onto wall)

    issues: list[Issue] = []
    issue_counter = 0
    window_size = config.analysisWindowFrames

    rules = [
        rule_cls(config.ruleThresholds.get(rule_cls.code, {}))
        for rule_cls in ALL_RULES
    ]

    for start in range(len(frames) - window_size + 1):
        end = start + window_size
        w_frames = frames[start:end]
        w_coms = coms[start:end]
        w_metrics = all_metrics[start:end]

        primary_idx = next(
            (start + i for i, f in enumerate(w_frames) if f.reliable),
            start,
        )

        window = WindowData(
            frames=w_frames,
            coms=w_coms,
            metrics=w_metrics,
            start_ms=w_frames[0].timestamp_ms,
            end_ms=w_frames[-1].timestamp_ms,
            primary_frame_index=primary_idx,
        )

        for rule in rules:
            issue = rule.evaluate(window, config.keypointConfidenceThreshold)
            if issue is None:
                continue
            # Skip warmup window
            if issue.start_ms < _WARMUP_MS:
                continue
            # Drop issues whose confidence is too low to be actionable
            if issue.confidence < 0.05:
                continue
            # Dedup: skip if same code overlaps OR is within cooldown of previous emission
            if _is_suppressed(issue, issues):
                continue
            issue_counter += 1
            issue.id = f"issue_{issue_counter:04d}"
            issues.append(issue)

    # --- Annotation export ---
    annotated_dir = os.path.join(output_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)
    annotated_paths: list[tuple[str, str]] = []

    _TRAIL_LEN = 20  # number of consecutive frames to look back for the trail

    # Export up to maxAnnotatedFrames, selecting by issue primary frames
    frames_to_annotate = _select_annotation_frames(issues, config.maxAnnotatedFrames, len(frames))

    issue_by_frame: dict[int, Issue] = {}
    for iss in issues:
        fi = iss.primary_frame_index
        if fi not in issue_by_frame:
            issue_by_frame[fi] = iss

    for fi in frames_to_annotate:
        if fi >= len(frames) or not _HAS_CV2:
            break
        com = coms[fi]

        # Build trail from the last _TRAIL_LEN consecutive frames ending at fi.
        # Using consecutive history avoids the jump artifacts that occurred when
        # trail was accumulated across non-consecutive annotated frames.
        trail_start = max(0, fi - _TRAIL_LEN)
        com_trail = [
            (coms[i].x, coms[i].y)
            for i in range(trail_start, fi + 1)
            if coms[i] and coms[i].reliable
        ]

        annotated = annotate_frame(
            image=raw_images[fi],
            frame=frames[fi],
            com=com,
            com_trail=com_trail,
            issue=issue_by_frame.get(fi),
            conf_threshold=config.keypointConfidenceThreshold,
        )

        issue = issue_by_frame.get(fi)
        fname = f"{issue.id}.jpg" if issue else f"frame_{fi:04d}.jpg"
        fpath = os.path.join(annotated_dir, fname)
        cv2.imwrite(fpath, annotated)

        rel_path = os.path.join("outputs", "annotated", fname)
        if issue:
            annotated_paths.append((issue.id, rel_path))

    result = build_result(
        meta, config, frames, coms, all_metrics, issues, annotated_paths, warnings
    )
    write_result(result, output_dir)
    return result


_ISSUE_COOLDOWN_MS = 8_000  # minimum gap between same-code issues


def _is_suppressed(candidate: Issue, existing: list[Issue]) -> bool:
    for iss in existing:
        if iss.code != candidate.code:
            continue
        # Suppress if windows overlap
        if candidate.start_ms <= iss.end_ms and candidate.end_ms >= iss.start_ms:
            return True
        # Suppress if within cooldown of the previous emission
        if candidate.start_ms - iss.end_ms < _ISSUE_COOLDOWN_MS:
            return True
    return False


def _select_annotation_frames(
    issues: list[Issue], max_frames: int, total_frames: int
) -> list[int]:
    frames: list[int] = []
    seen: set[int] = set()
    for iss in issues:
        fi = iss.primary_frame_index
        if fi not in seen and fi < total_frames:
            frames.append(fi)
            seen.add(fi)
        if len(frames) >= max_frames:
            break
    return frames
