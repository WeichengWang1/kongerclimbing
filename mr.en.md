# Climbing AI Analysis System Requirements and Roadmap

## 1. Project Positioning

This project targets climbing training review scenarios. It analyzes climbing videos through human pose detection, center-of-mass estimation, movement quality assessment, and visual rendering to help climbers understand movement issues and receive actionable improvement suggestions.

The system's core value is not simply drawing a human skeleton. It turns pose data into climbing-specific training feedback: whether the center of mass is in place, whether the legs are driving effectively, whether the upper body is over-pulling, whether the hips are close to the wall, and whether the movement is stable and efficient.

## 2. Target Users and Use Cases

### 2.1 Target Users

- Climbing enthusiasts who want to improve movement efficiency through video review.
- Climbing coaches who want to quickly identify student movement issues and generate standardized feedback.
- Training organizations that want to build reusable movement evaluation standards and training cases.

### 2.2 Core Use Case

1. The user uploads a climbing video.
2. The system detects body keypoints, torso direction, limb trajectories, and center-of-mass movement.
3. The system locates key movement segments and potential error timestamps.
4. The system overlays English annotations, pose skeletons, center-of-mass trajectories, and movement suggestions on images or videos.
5. The user reviews the report by timestamp and uses reference suggestions to understand and correct issues.

## 3. Inputs and Outputs

### 3.1 Inputs

- Climbing video file.
- Future extensible inputs:
  - Single-frame image.
  - Manual route/hold annotation.
  - User height, wingspan, dominant hand, training goal, and other profile information.

### 3.2 Outputs

- Pose analysis results: keypoints, joint angles, torso posture, center-of-mass coordinates, and trajectory.
- Movement quality log: abnormal movement timestamps, issue types, confidence, and explanation.
- Visual files:
  - Images with skeleton, center of mass, and English annotations.
  - Videos with pose data, center-of-mass trajectory, ghosting effects, and English annotations.
- Review suggestions: English movement issue descriptions and improvement recommendations.

## 4. Core Requirements

### 4.1 Dynamic Center-of-Mass Analysis

The system must estimate the climber's dynamic center of mass from human keypoints and display the center-of-mass point and trajectory in the video.

Key capabilities:

- Compute a weighted center of mass from torso, hips, shoulders, and limb keypoints.
- Identify trends in the center of mass relative to support points, hips, and torso.
- Quantify whether the center of mass has moved into position, for example:
  - Whether the center of mass is close to the target support foot.
  - Whether preload is completed before movement.
  - Whether the hips are close to the wall or route direction.
  - Whether the center-of-mass trajectory has obvious oscillation, pauses, or uncontrolled swings.

### 4.2 Movement Quality Assessment

The system must track limb and torso motion states and identify common climbing movement issues.

Initial priority issue types:

- Over-pulling with arms: excessive upper-body pulling with insufficient leg drive.
- Unstable center of mass: excessive center-of-mass sway or inefficient path.
- Poor foot engagement: insufficient use of foot holds or unstable foot placement.
- Locked elbow too early: locking the elbow too early, limiting further movement.
- Late hip shift (V1.1): hips move too late and the center of mass is not positioned in advance.
- Inefficient reach (V1.1): body position is insufficient before reaching, causing an ineffective reach.

Note: `Late hip shift` and `Inefficient reach` depend on the target support point or target hold direction. V1.0 does not detect holds and cannot reliably determine the target direction, so these two issue types are deferred to V1.1, when hold hints or a reach-direction proxy will be introduced. V1.0 only implements the first four issue types that do not depend on hold information.

Assessment results must include:

- Timestamp or time range.
- Issue type.
- Confidence.
- English explanation.
- English correction suggestion.

### 4.3 Visual Enhancement and Rendering

The system must overlay explainable training information on images and videos.

Image processing capabilities:

- Highlight body pose, torso direction, limb connections, and center-of-mass point.
- Add body ghosting to show movement paths.
- Overlay English movement annotations and analysis suggestions.

Video processing capabilities:

- Overlay skeleton, center of mass, center-of-mass trajectory, and issue prompts frame by frame.
- Display English assessment text during key error segments.
- Support exporting processed video files.
- Later versions should support rendering reference pose skeletons or animations back into the original video.

### 4.4 Review Interaction

The system must support reviewing movement issues by timestamp.

Core interactions:

- The video player displays the original or enhanced video.
- The issue timestamp list supports click-to-jump.
- The player can automatically pause at issue timestamps.
- The UI synchronizes issue descriptions, correction suggestions, and reference poses.

### 4.5 Language and Content Rules

- Product-generated image annotations, video annotations, movement issue descriptions, and analysis suggestions must be in English.
- Development documents and internal implementation notes may be written in Chinese.
- English feedback should be concise and actionable, avoiding overly long paragraphs.

## 5. Non-Functional Requirements

### 5.1 Accuracy

- Pose keypoint detection must provide confidence scores.
- Low-confidence frames must be marked as unreliable and must not produce deterministic movement conclusions.
- Movement evaluation standards must be configurable to avoid forcing all climbing styles into a single template.
- All coordinate and displacement metrics use normalized coordinates relative to frame size (`x / width`, `y / height`, range 0-1), so thresholds are not bound to a specific resolution and can be reused across videos.
- Accuracy acceptance is primarily qualitative: on the sample video set, humans review whether keypoints align well and issue judgments are reasonable, rather than enforcing a hard percentage metric. Low-confidence frames are excluded from conclusions as a baseline accuracy safeguard.

### 5.2 Explainability

- Every movement issue must be traceable to specific timestamps and data evidence.
- The visualization layer must display the key pose information that triggered the judgment, such as center of mass, hips, shoulder line, knee angle, or elbow angle.

### 5.3 Extensibility

- Pose detection, movement assessment, suggestion generation, and rendering/export should be modular.
- Future versions should be able to replace the pose detection model or introduce different climbing movement standard libraries.

### 5.4 Performance

- V1 prioritizes offline analysis quality.
- Future interactive interfaces must support quickly locating analysis results by video timestamp.

### 5.5 Data and Privacy

- Videos may contain identifiable faces and other personal information.
- V1.0 processes everything locally and offline, without uploading videos or analysis results to external services.
- Output files such as JSON and annotated images are written only to the local `outputs/` directory by default.
- Any future cloud or sharing capability requires a separate data storage and anonymization strategy.

## 6. Key Concepts and Evaluation Standards

### 6.1 Pose Data

Minimum keypoint set:

- Head, neck, or shoulder center.
- Left/right shoulder, left/right elbow, left/right wrist.
- Left/right hip, left/right knee, left/right ankle.

Recommended derived metrics:

- Shoulder line angle.
- Hip line angle.
- Torso tilt angle.
- Elbow joint angle.
- Knee joint angle.
- Horizontal/vertical hip displacement.
- Center-of-mass coordinates and trajectory.

### 6.2 Center-of-Mass Estimation

V1 may use a simplified weighted human body segment model:

- Torso and hips have the highest weight.
- Thighs, calves, upper arms, and forearms participate according to human segment proportions.
- When foot holds, hand holds, or wall-depth information is unavailable, use a 2D video-plane estimate first.

### 6.3 Movement Quality Judgment

V1 uses rules and thresholds as the primary approach because they are easier to debug and explain. Machine learning models may be introduced later.

V1.0 example rules that do not depend on hold information:

- If elbow angles decrease rapidly during the pulling phase while hip and knee displacement is insufficient, mark Over-pulling with arms.
- If the center-of-mass trajectory reverses direction multiple times in a short window, mark Unstable center of mass.
- If ankle/knee stability is poor and the center of mass does not shift toward the support foot, mark Poor foot engagement.
- If the arm becomes nearly straight or locked too early and subsequent movement is limited, mark Locked elbow too early.

V1.1 example rules that depend on target direction and require hold hints or a reach-direction proxy:

- If the hips do not move toward the target direction before the reach, mark Late hip shift.
- If reach distance increases but body position does not improve accordingly, mark Inefficient reach.

## 7. Roadmap

### V0.0: Research and Standard Definition

Goal: establish an explainable and iterative foundation for climbing movement evaluation.

Core tasks:

- Research available pose detection solutions.
- Define keypoint format, center-of-mass calculation, and movement issue taxonomy.
- Build an initial AI evaluation standard library for correct climbing posture.
- Prepare a small sample video set and manual review examples.

Acceptance criteria:

- Output keypoint schema.
- Output center-of-mass calculation notes.
- Output the first batch of movement issue rules.
- Complete manual review records for at least 3 sample videos.

### V1.0: Basic Pose Detection and Log Output

Goal: detect torso and limbs in video, track trajectories, and output initial movement issue logs.

Core tasks:

- Video input, frame sampling, and frame-level processing.
- Human keypoint detection and confidence filtering.
- Keypoint smoothing and trajectory generation.
- Center-of-mass estimation.
- Output abnormal pose timestamps and specific issue analysis.

Acceptance criteria:

- Can process local video files.
- Can output structured JSON analysis results.
- Can output issue timestamp lists.
- Can identify at least 3 hold-independent movement issue types, such as `over_pulling_with_arms`, `unstable_center_of_mass`, `poor_foot_engagement`, or `locked_elbow_too_early`.
- English issue descriptions and suggestions are ready for frontend display.

### V1.1: Intelligent Correction Suggestions

Goal: generate actionable English correction suggestions and reference comparisons based on V1.0 analysis results.

Core tasks:

- Build mappings from movement issues to correction suggestions.
- Generate annotated keyframe images.
- Generate correct-posture reference diagrams or skeleton images.

Acceptance criteria:

- Each issue type has at least one English suggestion template.
- Each key issue timestamp can export an annotated image.

### V1.2: Interactive Review Interface

Goal: develop the video player and issue review interface.

Core tasks:

- Play original or enhanced video.
- Display issue timeline.
- Click an issue to jump to the corresponding timestamp.
- Configure whether the player automatically pauses at issue timestamps.
- Synchronize English annotations, suggestions, and reference images.

Acceptance criteria:

- The user can complete one flow from video upload to issue review.
- Issue list, video timeline, and annotation information stay synchronized.

### V2.0: Pose Rendering and Advanced Visualization

Goal: integrate correct-pose references and movement rendering into the original video.

Core tasks:

- Render correct-pose skeletons or animations.
- Display body ghosting and center-of-mass paths.
- Highlight routes, key holds, and movement directions.
- Export enhanced videos.

Acceptance criteria:

- Output enhanced videos containing skeleton, center of mass, trajectory, and English annotations.
- Support focused rendering for key error segments.

## 8. Priorities

| Priority | Feature | Notes |
| --- | --- | --- |
| P0 | Video input and pose detection | Reliable keypoints are required for all later analysis |
| P0 | Center-of-mass estimation and trajectory | Core product differentiation |
| P0 | Movement issue log | Supports review and future UI |
| P1 | English annotations and suggestions | Supports explainable output |
| P1 | Annotated image export | Quickly validates analysis quality |
| P2 | Interactive player | Improves review experience |
| P2 | Enhanced video export | Improves complete product experience |
| P3 | Correct-pose animation rendering | Depends on mature analysis standards |

## 9. Risks and Open Validation Questions

- Single-view 2D video cannot accurately reflect wall depth or how close the body is to the wall.
- Without hand hold, foot hold, and route detection, some movement judgments can only be approximate.
- Different climbing styles, route types, and individual body conditions affect what counts as correct movement.
- Occlusion, multiple people in frame, low light, and fast movement reduce pose detection accuracy.
- English suggestions must remain professional but not overly absolute, especially under low-confidence conditions.

## 10. Recommended First Development Loop

Prioritize the V1.0 minimum usable loop:

1. Input a local climbing video.
2. Sample frames and detect human keypoints.
3. Compute frame-level center of mass.
4. Generate keypoint, center-of-mass, and issue-log JSON.
5. Export several annotated keyframe images.

After this loop is complete, proceed to V1.1 suggestion generation, including target-direction-dependent rules such as `Late hip shift` and `Inefficient reach`, and then V1.2 interactive interface development.

## 11. Glossary

| Term | Chinese | Meaning |
| --- | --- | --- |
| Center of Mass (CoM) | 重心 | Overall mass center estimated from weighted body segments; V1.0 estimates it in the 2D video plane |
| Support foot | 支撑脚 | The currently primary weight-bearing foot; the center of mass should shift toward this side |
| Preload | 预加载 | Moving the center of mass into position before initiating force generation |
| Hip-to-wall | 贴壁 | Keeping the hips close to the wall to reduce arm-load demand |
| Keypoint | 关键点 | A 2D coordinate point for a human joint/body part with confidence |
| Normalized coordinate | 归一化坐标 | A 0-1 coordinate relative to frame width/height, comparable across resolutions |
| Reach/target direction | 目标方向 | Direction toward the next target hold; unknown in V1.0 because holds are not detected |
| Unreliable frame | 不可靠帧 | A frame whose core keypoint confidence is too low and is excluded from movement judgment |
