# climbanalyze — Dev Notes

## Environment Setup

System Python is 3.14 — torch/ultralytics have no 3.14 wheels. Use a 3.12 venv:

```bash
# Install Python 3.12 via pyenv or brew
brew install pyenv
pyenv install 3.12
pyenv local 3.12

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
# Also requires: brew install ffmpeg
```

## Run

```bash
python -m climbanalyze samples/climb.mp4
python -m climbanalyze samples/climb.mp4 --config my_config.json --output-dir outputs
```

## Architecture

```
src/climbanalyze/
  config/       — AnalysisConfig dataclass + defaults (§4)
  video/        — VideoMeta, frame extraction via OpenCV
  pose/         — 13-point schema, COCO-17 adapter, YOLO11 detector, smoother
  analysis/     — CoM (§3.4), FrameMetrics (§6.3), 4 sliding-window rules (§3.5)
  rendering/    — OpenCV skeleton/CoM annotator
  export/       — JSON builder matching §5 schema
  pipeline.py   — end-to-end orchestration, exit codes (§11)
  __main__.py   — CLI argparse entry point
```

## Key Conventions (from spec)

- All coordinates normalized: `x = x_px / width`, `y = y_px / height`, origin top-left, y-down.
- JSON output: camelCase field names (`frameIndex`, `timestampMs`, `centerOfMass`).
- Keypoint names and issue codes: snake_case (`left_shoulder`, `over_pulling_with_arms`).
- Rule thresholds always read from `config.ruleThresholds[code]` — never hardcoded.
- Frame reliability: core keypoint avg confidence < `frameConfidenceThreshold` → `reliable=False`.
- Issue windows: skip if reliable frame ratio < 50%; dedup overlapping same-code windows.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | OK |
| 2 | Video not found / unsupported format |
| 3 | Video unreadable / ffmpeg missing / missing deps |
| 4 | No person detected |
| 5 | Config JSON invalid |

## V1.1 Placeholders

`late_hip_shift` and `inefficient_reach` rules are not implemented — they require
a known reach direction (岩点 info). Add them in `analysis/rules/` once V1.1 design is settled.
