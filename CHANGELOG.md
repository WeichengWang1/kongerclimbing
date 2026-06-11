# Changelog

## V1.1

- **建议模板库**：每类问题按严重程度（low / medium / high）给出差异化的英文提示，`conf < 0.30` 时自动在消息前加 `Possible:` 标注不确定性。
- **多帧导出**：每个问题导出 3 张关键帧——`_before`（问题发生前约 1 秒）、`_peak`（最具代表性的帧，带标注）、`_after`（窗口末帧，展示动作结果）。JSON `artifacts` 新增 `frameRole` 字段。

## V1.0

- **姿态检测**：基于 YOLO11x-pose 的 17 点骨架检测，适配为 13 点 schema；MPS 加速，帧间移动平均平滑（窗口=5）。
- **重心估算**：3 段加权（肩 35%、髋 45%、肢体 20%），过滤 YOLO (0,0) 遮挡假点。
- **动作分析**：4 条滑动窗口规则：
  - `over_pulling_with_arms` — 手臂用力、腿部静止
  - `unstable_center_of_mass` — 重心频繁反向
  - `poor_foot_engagement` — 踝点抖动、重心未移向支撑脚
  - `locked_elbow_too_early` — 运动途中手臂过早伸直
- **抑制逻辑**：前 1.5 秒 warmup 跳过；同类问题 8 秒冷却；置信度 < 0.05 丢弃。
- **标注导出**：关键帧叠加骨架、重心点、CoM 轨迹线、问题标签和建议文字；骨架使用原始帧坐标（非平滑），避免偏移。
- **JSON 输出**：标准化 `analysis.json`，含帧数据、问题列表、配置快照和标注图路径。
- **CLI**：`python -m climbanalyze <video> [--config] [--output-dir]`，退出码 0/2/3/4/5。
