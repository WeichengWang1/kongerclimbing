# kongerclimbing

![KongerClimbing AI bouldering coach hero image](assets/img.png)

AI bouldering analysis for beginners training toward V5 routes.

Analyze a local climbing video to detect pose keypoints, estimate center of mass, and identify movement issues. Outputs a structured JSON report and annotated key frames.

## Requirements

- macOS (Apple Silicon recommended — MPS acceleration used automatically)
- Python 3.12 (system Python 3.14 is not supported by torch/ultralytics)
- Homebrew

## Setup

```bash
# 1. Install Python 3.12 and ffmpeg
brew install python@3.12 ffmpeg

# 2. Create a virtual environment
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e .
```

## Usage

```bash
source .venv/bin/activate

# Basic analysis
python -m climbanalyze samples/climb.mp4

# Custom output directory
python -m climbanalyze samples/climb.mp4 --output-dir my_outputs

# Custom config
python -m climbanalyze samples/climb.mp4 --config my_config.json
```

Output is written to `outputs/` by default:

```
outputs/
  analysis.json       # full structured report
  annotated/
    issue_0001.jpg    # annotated frame for each detected issue
    issue_0002.jpg
    ...
```

## Configuration

Create a JSON file to override any default settings:

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
  "ruleThresholds": {
    "over_pulling_with_arms": {
      "elbowAngleDeltaDeg": -25,
      "minHipDisplacement": 0.02
    }
  }
}
```

`device` accepts `auto` (default), `mps`, or `cpu`. On Apple Silicon, `auto` selects MPS automatically.

## Output JSON structure

```
analysis.json
├── version, status, coordinateSpace
├── source          — video path, fps, resolution, duration
├── config          — full config used for this run
├── frames[]        — per-frame keypoints, center of mass, metrics
├── issues[]        — detected movement issues with timestamps and evidence
└── artifacts       — paths to annotated frame images
```

All coordinates are normalized (0–1 relative to frame size).

## Reading the annotated frames

Each exported image overlays analysis information on the original video frame:

```
┌─────────────────────────────────────────────────┐
│ Over-pulling with arms  [medium]                │  ← issue label + severity
│ Drive from your legs and push your hips up...   │  ← one-line coaching tip
│                                                 │
│          ● ──── ●          ← green dots: body keypoints (shoulders,
│         /|      |\           elbows, wrists, hips, knees, ankles)
│        / |      | \        ← orange-yellow lines: skeleton connections
│       ●  ●      ●  ●
│          |      |
│       ◎←─┘      │         ← orange-red circle with white ring: current
│          |      |            center of mass position
│       ●  ●      ●  ●
│                            ← blue line: center of mass trail
│       ╌╌╌╌╌╌╌╌╌            (last ~2 seconds of movement history)
└─────────────────────────────────────────────────┘
```

**Blue trail line** — traces where the center of mass has been over the last ~2 seconds leading up to this frame. A smooth upward arc is good. Erratic reversals, sharp zigzags, or a very short trail (little movement) indicate inefficient body control.

**Orange-red CoM dot** — the estimated center of mass at this exact frame, calculated from a weighted average of shoulders (35%), hips (45%), and limbs (20%).

**Issue label and severity** — the movement problem detected in the window leading up to this frame (`low` / `medium` / `high`), based on how far the measured values exceed the rule thresholds.

**Coaching tip** — a one-sentence actionable suggestion. Intended as a starting point for self-review, not a definitive diagnosis.

## Detected issue types (V1.0)

| Code | Description |
|------|-------------|
| `over_pulling_with_arms` | Pulling hard with arms while hips and legs stay still |
| `unstable_center_of_mass` | Center of mass moving erratically between holds |
| `poor_foot_engagement` | Feet unstable on holds, weight not shifting over support foot |
| `locked_elbow_too_early` | Arm nearly straight too early during an active move |

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Analysis completed successfully |
| 2 | Video not found or unsupported format |
| 3 | Video unreadable, corrupted, or ffmpeg missing |
| 4 | No person detected in video |
| 5 | Config file invalid |
