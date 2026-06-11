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
from .analysis.activity import detect_climb_segment
from .analysis.rules import ALL_RULES, Issue, WindowData
from .rendering.annotator import annotate_frame
from .rendering.correction import generate_correction_diagram
from .rendering.holds import detect_route_holds
from .rendering.video_overlay import write_annotated_video
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

    frames: list[PoseFrame] = []           # smoothed — used for analysis
    render_frames: list[PoseFrame] = []    # raw detection — used for annotation only
    coms: list[Optional[CenterOfMass]] = []
    all_metrics: list[FrameMetrics] = []
    raw_images: list = []

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

        # Save the raw (unsmoothed) pose for rendering.
        # Smoothed coordinates lag behind the actual position when the climber
        # moves; raw coords match the pixel content of this exact image.
        render_frames.append(pose)

        smoothed = smoother.smooth(pose)
        com = compute_center_of_mass(smoothed, config.keypointConfidenceThreshold)
        metrics = compute_frame_metrics(smoothed, config.keypointConfidenceThreshold)

        frames.append(smoothed)
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

    # --- Restrict analysis to the active climbing segment ---
    # The clip usually opens with the climber walking toward the wall and ends
    # with them stepping off; analyzing those frames produces false issues.
    segment = detect_climb_segment(frames, coms, config.targetFps)
    if segment is None:
        # No sustained climbing detected — analyze the whole clip but flag it.
        seg_start, seg_end = 0, len(frames) - 1
        warnings.append(
            "Climbing segment not detected; analyzing the entire clip. "
            "Issues before/after the actual climb may be inaccurate."
        )
    else:
        seg_start, seg_end = segment

    # --- Sliding-window rule evaluation (within the climb segment only) ---
    issues: list[Issue] = []
    issue_counter = 0
    window_size = config.analysisWindowFrames

    rules = [
        rule_cls(config.ruleThresholds.get(rule_cls.code, {}))
        for rule_cls in ALL_RULES
    ]

    for start in range(len(frames) - window_size + 1):
        end = start + window_size
        # Only evaluate windows fully inside the detected climbing segment.
        if start < seg_start or (end - 1) > seg_end:
            continue
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
    annotated_paths: list[dict] = []

    _TRAIL_LEN = 20  # number of consecutive frames to look back for the trail

    issue_export_count = 0

    for iss in issues:
        if issue_export_count >= config.maxAnnotatedFrames or not _HAS_CV2:
            break

        peak_fi = iss.primary_frame_index

        # Export annotated peak frame (each issue gets its own unique filename)
        if peak_fi < len(frames):
            com = coms[peak_fi]
            trail_start = max(0, peak_fi - _TRAIL_LEN)
            com_trail = [
                (coms[i].x, coms[i].y)
                for i in range(trail_start, peak_fi + 1)
                if coms[i] and coms[i].reliable
            ]
            annotated = annotate_frame(
                image=raw_images[peak_fi],
                frame=render_frames[peak_fi],
                com=com,
                com_trail=com_trail,
                issue=iss,
                conf_threshold=config.keypointConfidenceThreshold,
            )
            fname = f"{iss.id}_peak.jpg"
            fpath = os.path.join(annotated_dir, fname)
            cv2.imwrite(fpath, annotated)
            rel_path = os.path.join("outputs", "annotated", fname)
            annotated_paths.append({"issueId": iss.id, "frameRole": "peak", "path": rel_path})

        # Detect all route holds by colour at the peak frame
        route_holds = detect_route_holds(
            raw_images[peak_fi],
            {kp.name: (kp.x, kp.y) if kp.confidence >= config.keypointConfidenceThreshold else None
             for kp in frames[peak_fi].keypoints},
        ) if _HAS_CV2 else []

        # Correction diagram: actual vs corrected pose side-by-side
        corr_fname = f"{iss.id}_correction.jpg"
        corr_path = os.path.join(annotated_dir, corr_fname)
        if generate_correction_diagram(
            frame=frames[peak_fi],
            com=coms[peak_fi],
            issue=iss,
            output_path=corr_path,
            conf_threshold=config.keypointConfidenceThreshold,
            route_holds=route_holds,
        ):
            annotated_paths.append({
                "issueId": iss.id,
                "frameRole": "correction",
                "path": os.path.join("outputs", "annotated", corr_fname),
            })

        issue_export_count += 1

    # --- Full annotated playback video (skeleton/CoM on every frame) ---
    # The viewer plays this so the skeleton stays attached during playback.
    annotated_video_rel: Optional[str] = None
    if _HAS_CV2 and raw_images:
        video_abs = os.path.join(output_dir, "annotated.mp4")
        if write_annotated_video(
            raw_images=raw_images,
            render_frames=render_frames,
            coms=coms,
            fps=config.targetFps,
            out_path=video_abs,
            conf_threshold=config.keypointConfidenceThreshold,
            issues=issues,
        ):
            annotated_video_rel = os.path.join("outputs", "annotated.mp4")
        else:
            warnings.append("Annotated video encoding failed; viewer will use the raw clip.")

    result = build_result(
        meta, config, frames, coms, all_metrics, issues, annotated_paths, warnings,
        annotated_video=annotated_video_rel,
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


