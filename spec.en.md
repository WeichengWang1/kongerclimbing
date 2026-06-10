# Climbing AI Analysis System Development Specification

## 1. Specification Goal

This document is based on `mr.md` and guides subsequent engineering development. The first development goal is to deliver the V1.0 minimum usable loop: input a local climbing video and output pose keypoints, center-of-mass trajectory, movement issue logs, and annotated keyframe images.

## 2. V1.0 Scope

### 2.1 Must Implement

- Read local video files.
- Sample frames according to configuration.
- Detect pose keypoints for each frame; if multiple people are detected, select one primary climber by rule and analyze only that single-person trajectory.
- Filter low-confidence keypoints and low-quality frames.
- Apply basic smoothing to keypoint trajectories.
- Compute frame-level 2D center-of-mass coordinates.
- Identify the first batch of movement issues by rules; V1.0 includes only the four hold-independent issue types listed in §3.5.
- Output structured JSON analysis results.
- Export annotated key issue frames with English annotations.

Note: The detector itself may output multiple people. V1.0 selects one primary climber per frame (see §3.2), and downstream modules such as smoothing, center-of-mass estimation, and movement analysis only process this single-person trajectory.

### 2.2 Not Implemented Yet

- Multi-person scene analysis.
- 3D pose estimation.
- Automatic hold detection.
- Web interactive player.
- Full enhanced video export.
- Correct-pose animation rendering.

### 2.3 Technology Stack and Runtime Environment

- Language: Python. **Runtime is locked to a Python 3.11 or 3.12 virtual environment**. The local system Python is 3.14, and torch / ultralytics currently do not provide 3.14 wheels, so the system interpreter must not be used directly.
- Pose detection: Ultralytics YOLO11-pose, which outputs COCO-17 keypoints.
- Inference backend: PyTorch. On Apple Silicon, such as M4 Max, use the `mps` device and fall back to `cpu` when unavailable.
- Video decoding: OpenCV (`opencv-python`), dependent on system `ffmpeg` (missing locally; install with `brew install ffmpeg`).
- Delivery form: core Python library API plus a thin CLI entrypoint: `python -m climbanalyze <video> [--config config.json]`.
- Keypoint adaptation: detector output is mapped through an adapter layer into the 13-point schema in §3.2. COCO-17 is a superset, with eyes/ears discarded. Future detectors can be replaced by implementing the same adapter interface.

### 2.4 Coordinate and Unit Conventions

- All keypoints, center of mass, and displacement metrics use **normalized coordinates**: `x_norm = x_px / width`, `y_norm = y_px / height`, range 0-1. The origin is at the top-left, and y increases downward.
- JSON coordinates are all normalized values. `source.width` and `source.height` keep the original pixel dimensions for reconstruction when needed.
- Angle metrics use degrees.
- Velocity metrics use normalized distance per second.
- Top-level field `coordinateSpace` is fixed to `"normalized"` to identify the coordinate system for consumers.

## 3. System Modules

### 3.1 Video Ingestion

Responsibilities:

- Validate video path and format.
- Read video metadata: duration, resolution, frame rate.
- Sample frames by target FPS or frame interval.

Input:

- Local video path.
- Analysis configuration.

Output:

- Frame image.
- Frame index.
- Timestamp.
- Video metadata.

Frame index and timestamp conventions:

- After sampling, frames are re-indexed. `frameIndex` is the continuous index of the sampled sequence (0, 1, 2, ...), not the original video frame number.
- `timestampMs` is based on the original video timeline: `round(originalFrameNo / sourceFps * 1000)`, so it can map back to the original video for seeking.
- Sampling strategy: sample the original `sourceFps` at equal intervals according to `targetFps` (`step = round(sourceFps / targetFps)`, minimum 1).
- Also keep `originalFrameIndex`, the original video frame number, for precise location and debugging.

### 3.2 Pose Detection

Responsibilities:

- Detect human keypoints for each frame.
- Output keypoint coordinates (normalized) and confidence.
- Select one primary climber from multi-person detections; downstream modules analyze only this single-person trajectory.

Primary climber selection rules (V1.0, deterministic):

1. Primary key: choose the person with the largest normalized bbox area.
2. Tie-breaker when area difference is < 5%: choose the person with higher average confidence on core keypoints.
3. If still tied: choose the bbox center closest to the previous frame's primary climber bbox center to preserve trajectory continuity; on the first frame, choose the lower index.

V1.0 does not perform cross-frame multi-object tracking (ReID); per-frame independent selection is sufficient.

Minimum output keypoints:

- nose
- left_shoulder
- right_shoulder
- left_elbow
- right_elbow
- left_wrist
- right_wrist
- left_hip
- right_hip
- left_knee
- right_knee
- left_ankle
- right_ankle

### 3.3 Pose Filtering & Smoothing

Responsibilities:

- Filter abnormal frames based on keypoint confidence.
- Smooth keypoints across consecutive frames.
- Mark frames that cannot be used for movement judgment.

Default rules:

- A keypoint is marked unreliable when its confidence is below `keypointConfidenceThreshold`.
- A frame is excluded from movement judgment when the average confidence of core keypoints is below `frameConfidenceThreshold`.
- V1.0 may use a moving average or exponential smoothing for trajectory smoothing.

### 3.4 Center of Mass Estimation

Responsibilities:

- Estimate frame-level center of mass from 2D keypoints.
- Output center-of-mass coordinates, confidence, and trajectory.

V1.0 simplified calculation, using three non-overlapping segments to avoid double weighting:

- `shoulders_center` = midpoint of left_shoulder and right_shoulder.
- `hips_center` = midpoint of left_hip and right_hip.
- `limbs_center` = average of valid elbows, wrists, knees, and ankles.
- `center_of_mass = shoulders_center * 0.35 + hips_center * 0.45 + limbs_center * 0.20`.

Weight rationale: torso and hips account for most body mass, so shoulders + hips total 0.80, with hips slightly higher; limbs total 0.20.

Missing data handling:

- If a segment cannot be computed, such as `limbs_center` having no valid points, exclude that segment and **renormalize** the remaining segment weights before computing the weighted sum. Do not fill with 0 or fabricated values.
- Center-of-mass `confidence` is the weighted average confidence of participating keypoints.
- If either `shoulders_center` or `hips_center` is missing, mark the frame's center of mass as unreliable and exclude it from movement judgment.

### 3.5 Movement Analysis

Responsibilities:

- Compute derived movement metrics.
- Identify movement issues by rules.
- Output explainable English issue descriptions and suggestions.

V1.0 derived metrics:

- shoulder_line_angle
- hip_line_angle
- torso_angle
- left_elbow_angle
- right_elbow_angle
- left_knee_angle
- right_knee_angle
- center_of_mass_velocity
- center_of_mass_direction_changes
- hip_displacement
- wrist_reach_distance

V1.0 issue types, all independent of holds/target direction:

| code | label | Trigger basis |
| --- | --- | --- |
| over_pulling_with_arms | Over-pulling with arms | Elbow angle decreases rapidly while hip/knee displacement is insufficient |
| unstable_center_of_mass | Unstable center of mass | Center of mass changes direction multiple times in a short window or velocity fluctuates excessively |
| poor_foot_engagement | Poor foot engagement | Knee/ankle stability is poor and center of mass does not shift toward the support side |
| locked_elbow_too_early | Locked elbow too early | Arm becomes nearly straight or locked too early, limiting subsequent movement |

V1.1 deferred issue types, which depend on target direction and require hold hints or a reach-direction proxy:

| code | label | Trigger basis |
| --- | --- | --- |
| late_hip_shift | Late hip shift | Hips do not move toward the target direction before the reach |
| inefficient_reach | Inefficient reach | Reach distance increases while body position does not improve accordingly |

Severity grading in V1.0 uses the normalized rule violation magnitude `m`, defined per rule in §6.3:

- `m < 0.33` -> `low`
- `0.33 <= m < 0.66` -> `medium`
- `m >= 0.66` -> `high`

Issue `confidence` is the product of:

- Average keypoint/center-of-mass confidence over frames participating in the judgment window.
- Normalized rule violation score `m`; the more it exceeds the threshold, the higher the score, capped at 1.0.

Unreliable frames derived from values below `keypointConfidenceThreshold` are excluded. If reliable frames account for < 50% of the window, the issue must not be emitted.

### 3.6 Annotation Rendering

Responsibilities:

- Overlay skeleton, center of mass, trajectory segment, and English annotation on keyframe images.
- Export annotated images.

V1.0 image content:

- Original frame.
- Keypoints and skeleton lines.
- Current center-of-mass point.
- Recent center-of-mass trajectory.
- Issue label and one English suggestion.

### 3.7 Result Export

Responsibilities:

- Output analysis JSON.
- Output annotated keyframe images.
- Keep output paths and filenames stable for future UI integration.

## 4. Configuration Specification

Use a single analysis configuration object:

```json
{
  "poseModel": "yolo11x-pose",
  "device": "auto",
  "targetFps": 10,
  "keypointConfidenceThreshold": 0.4,
  "frameConfidenceThreshold": 0.5,
  "smoothingWindow": 5,
  "analysisWindowFrames": 12,
  "maxAnnotatedFrames": 20,
  "language": "en",
  "ruleThresholds": {
    "over_pulling_with_arms": {
      "elbowAngleDeltaDeg": -25,
      "minHipDisplacement": 0.02,
      "minKneeAngleDeltaDeg": 5
    },
    "unstable_center_of_mass": {
      "maxDirectionChanges": 3,
      "maxVelocityVariance": 0.0008
    },
    "poor_foot_engagement": {
      "maxAnkleJitter": 0.03,
      "minComShiftToSupport": 0.015
    },
    "locked_elbow_too_early": {
      "lockedElbowAngleDeg": 165,
      "minPostLockComProgress": 0.02
    }
  }
}
```

Constraints:

- `language` is fixed to `en` in V1.0.
- `device` accepts `auto` | `mps` | `cpu`; `auto` prefers `mps` on Apple Silicon and otherwise uses `cpu`.
- Displacement and velocity thresholds use normalized coordinate units (see §2.4), and angles use degrees.
- Thresholds must be configurable and must not be hardcoded in rule implementations. Rules read from `ruleThresholds[code]`.
- When configuration is missing, use defaults and record the complete final configuration, including `ruleThresholds`, in the output JSON.

## 5. Output JSON Schema

The V1.0 output file should be named `analysis.json`.

```json
{
  "version": "1.0",
  "coordinateSpace": "normalized",
  "status": "ok",
  "warnings": [],
  "source": {
    "videoPath": "samples/climb.mp4",
    "durationMs": 12000,
    "fps": 30,
    "width": 1920,
    "height": 1080
  },
  "config": {
    "poseModel": "yolo11x-pose",
    "device": "mps",
    "targetFps": 10,
    "keypointConfidenceThreshold": 0.4,
    "frameConfidenceThreshold": 0.5,
    "smoothingWindow": 5,
    "analysisWindowFrames": 12,
    "maxAnnotatedFrames": 20,
    "language": "en",
    "ruleThresholds": { "...": "see §4; write the full content in output" }
  },
  "frames": [
    {
      "frameIndex": 0,
      "originalFrameIndex": 0,
      "timestampMs": 0,
      "reliable": true,
      "keypoints": [
        {
          "name": "left_shoulder",
          "x": 0.319,
          "y": 0.295,
          "confidence": 0.91
        }
      ],
      "centerOfMass": {
        "x": 0.365,
        "y": 0.482,
        "confidence": 0.84
      },
      "metrics": {
        "torsoAngle": 12.3,
        "leftElbowAngle": 95.4,
        "rightElbowAngle": 142.8,
        "leftKneeAngle": 118.1,
        "rightKneeAngle": 104.5
      }
    }
  ],
  "issues": [
    {
      "id": "issue_0001",
      "code": "over_pulling_with_arms",
      "label": "Over-pulling with arms",
      "startMs": 3400,
      "endMs": 4100,
      "confidence": 0.76,
      "severity": "medium",
      "evidence": {
        "primaryFrameIndex": 34,
        "metrics": {
          "elbowAngleDeltaDeg": -38.0,
          "hipDisplacement": 0.006,
          "kneeAngleDeltaDeg": 1.2
        }
      },
      "message": "You are pulling hard with your arms while your hips and legs stay still.",
      "recommendation": "Drive from your legs and push your hips up before pulling with your arms."
    }
  ],
  "artifacts": {
    "annotatedFrames": [
      {
        "issueId": "issue_0001",
        "path": "outputs/annotated/issue_0001.jpg"
      }
    ]
  }
}
```

Field notes:

- `coordinateSpace`: fixed to `"normalized"` (see §2.4).
- `status`: `ok` | `no_person_detected` | `low_quality`. If most frames are unreliable, still output JSON with a warning. Fatal errors are covered in §11 and do not produce this file.
- `warnings`: array of non-fatal warnings, such as low light, low reliable-frame ratio, or unavailable hold information.
- Coordinate fields in `keypoints` and `centerOfMass` are all normalized 0-1 values.

## 6. Rule Implementation Specification

### 6.1 General Rule Requirements

- Every rule must return `confidence`.
- Every rule must return English `message` and `recommendation`.
- Rules must not output strong conclusions from low-reliability frames.
- The same issue type in the same time window should be merged to avoid repeated noise.
- Rule thresholds should be read from configuration or a rule configuration table.

### 6.2 Example Rule Pseudocode

```text
if elbow_angle_delta < elbowAngleDeltaDeg          # rapid elbow angle decrease during pulling
and hip_displacement < minHipDisplacement
and knee_angle_delta < minKneeAngleDeltaDeg:
  emit over_pulling_with_arms
```

```text
if center_of_mass_direction_changes >= maxDirectionChanges
or center_of_mass_velocity_variance >= maxVelocityVariance:
  emit unstable_center_of_mass
```

```text
if ankle_jitter > maxAnkleJitter                   # high ankle jitter = unstable foot placement
and com_shift_toward_support < minComShiftToSupport:
  emit poor_foot_engagement
```

```text
if elbow_angle > lockedElbowAngleDeg               # nearly straight or locked too early
and com_progress_after_lock < minPostLockComProgress:
  emit locked_elbow_too_early
```

Pseudocode for `late_hip_shift` and `inefficient_reach` is deferred to V1.1 because it depends on target direction.

### 6.3 Derived Metrics and Default Thresholds

Window definition: window length = `analysisWindowFrames`, sliding step = 1 frame.

- `center_of_mass_velocity`: center-of-mass displacement between adjacent reliable frames / frame interval in seconds, in normalized units per second.
- `center_of_mass_direction_changes`: number of center-of-mass velocity vector direction reversals inside the window; angle with previous vector > 90 degrees counts as one reversal.
- `center_of_mass_velocity_variance`: variance of speed magnitudes inside the window.
- `elbow_angle_delta` / `knee_angle_delta`: angle difference from the first to last frame of the window, in degrees.
- `hip_displacement`: displacement magnitude of the hip midpoint inside the window, normalized.
- `ankle_jitter`: standard deviation of ankle point positions inside the window, normalized.
- `com_shift_toward_support`: horizontal center-of-mass movement toward the support side, where the support side is the lower foot, normalized.
- `com_progress_after_lock`: net center-of-mass displacement after elbow lock, normalized.

Default thresholds are centralized in §4 `ruleThresholds` and must not be hardcoded inside rule code. Rule violation magnitude `m`, used for severity and confidence, is `clamp(amount over threshold / calibrated full-scale amount, 0, 1)`. Each rule defines its own full-scale amount; for example, over-pulling may treat `elbow_angle_delta` being 30 degrees lower than the threshold as full scale.

## 7. English Copy Specification

English output should follow this format:

- Label: short label with initial capitalization, for example `Late hip shift`.
- Message: one sentence that points out the issue, for example `Your hips moved after the reach started.`.
- Recommendation: one actionable suggestion, for example `Shift your center of mass over the support foot before extending your arm.`.

Copy requirements:

- Avoid absolute diagnoses such as `You are wrong`.
- Prefer coaching language such as `Try...`, `Shift...`, or `Keep...`.
- Use conservative wording under low confidence, such as `Possible...`.

## 8. File and Directory Conventions

Recommended future implementation structure:

```text
src/climbanalyze/
  __init__.py
  __main__.py        # CLI entrypoint: python -m climbanalyze <video>
  pipeline.py        # End-to-end orchestration of all modules
  video/             # Video reading and frame sampling
  pose/              # Pose detection, COCO-17-to-schema adapter, smoothing
  analysis/          # Center of mass, derived metrics, movement rules
  rendering/         # Image/video overlays
  export/            # JSON and file output
  config/            # Default configuration and rule thresholds
samples/
outputs/
  analysis.json
  annotated/
```

Notes:

- `__main__.py` parses CLI arguments, loads configuration, and calls `pipeline`.
- `pipeline.py` orchestrates modules and handles errors (see §11).
- `video/` handles video reading and frame sampling.
- `pose/` handles pose detection, keypoint schema, detector adapter layer, and smoothing.
- `analysis/` handles center of mass, metrics, and movement rules.
- `rendering/` handles image/video overlays.
- `export/` handles JSON and file output.
- `config/` handles default configuration and rule thresholds.

## 9. Acceptance Criteria

V1.0 is complete when all of the following are true:

1. Given a local climbing video, the program completes the analysis flow successfully.
2. It outputs `analysis.json` containing `source`, `config`, `frames`, `issues`, and `artifacts`.
3. Every reliable frame contains keypoint and center-of-mass data.
4. Every issue contains English `label`, `message`, and `recommendation`.
5. It can identify at least 3 hold-independent movement issue types: `over_pulling_with_arms`, `unstable_center_of_mass`, `locked_elbow_too_early`, or `poor_foot_engagement`.
6. It exports at least 1 annotated keyframe image.
7. Low-confidence frames do not generate deterministic movement issues.
8. Coordinate output uses normalized values, `coordinateSpace = normalized`, and `source` includes original `width` and `height`.
9. Error inputs return the corresponding exit codes in §11 without crashing.

## 10. Future Version Extension Points

### V1.1

- `late_hip_shift` and `inefficient_reach` rules, implemented after introducing hold hints or a reach-direction proxy.
- Suggestion template library.
- Reference pose skeleton images.
- Richer keyframe export.

### V1.2

- Frontend player.
- Issue timeline.
- Auto pause and jump.
- Analysis result visualization panel.

### V2.0

- Enhanced video export.
- Correct-pose animation or skeleton rendering.
- Route and hold visualization.
- More complete training report.

## 11. Error Handling and Exit Codes

Fatal errors: print an English error message to stderr, do not produce `analysis.json`, and return a non-zero exit code.

| Exit code | Scenario | Handling |
| --- | --- | --- |
| 0 | Completed normally | Output full JSON; `status` may be `ok` or `low_quality` |
| 2 | Video path does not exist or format is unsupported | Print stderr error and exit |
| 3 | Video is corrupted or cannot be decoded, including missing ffmpeg | Print stderr error and exit |
| 4 | No person detected in the whole video | Output minimal JSON, set `status = no_person_detected`, exit with code 4 |
| 5 | Invalid config file, such as JSON parse failure or field type error | Print stderr error and exit |

Non-fatal degradation with exit code 0 and `warnings`:

- Reliable-frame ratio < 50% -> `status = low_quality`; still output JSON but do not force issue conclusions.
- Some frames have no person or low confidence -> mark those frames as `reliable=false` and skip their movement judgment.
- Requested `device` is `mps` but unavailable -> fall back to `cpu` and add a warning.

Idempotency and output: repeated runs on the same input overwrite `outputs/`; annotated image names are stable and predictable as `issue_<id>.jpg`.
