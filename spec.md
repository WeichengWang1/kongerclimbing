# 攀岩 AI 分析系统开发规格说明

## 1. 规格目标

本文档基于 `mr.md`，用于指导后续工程开发。首个开发目标是交付 V1.0 最小可用闭环：输入本地攀岩视频，输出姿态关键点、重心轨迹、动作问题日志和关键帧标注图。

## 2. V1.0 范围

### 2.1 必须实现

- 读取本地视频文件。
- 按配置抽帧。
- 对每帧进行单人姿态关键点检测。
- 过滤低置信度关键点和低质量帧。
- 对关键点轨迹进行基础平滑。
- 计算帧级 2D 重心坐标。
- 基于规则识别首批动作问题。
- 输出结构化 JSON 分析结果。
- 导出关键问题帧标注图片，批注内容使用英文。

### 2.2 暂不实现

- 多人场景分析。
- 3D 姿态估计。
- 岩点自动识别。
- Web 交互播放器。
- 增强视频完整导出。
- 正确姿态动画渲染。

## 3. 系统模块

### 3.1 Video Ingestion

职责：

- 校验视频路径和格式。
- 读取视频元信息：时长、分辨率、帧率。
- 按目标 FPS 或帧间隔抽帧。

输入：

- 本地视频路径。
- 分析配置。

输出：

- 帧图像。
- 帧索引。
- 时间戳。
- 视频元信息。

### 3.2 Pose Detection

职责：

- 对每帧检测人体关键点。
- 输出关键点坐标和置信度。
- 在多人检测结果中选择主要攀爬者；V1.0 默认选择面积最大或平均置信度最高的人体。

输出关键点至少包含：

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

职责：

- 基于关键点置信度过滤异常帧。
- 对连续帧关键点进行平滑。
- 标记不可用于动作判断的帧。

默认规则：

- 单关键点置信度低于 `keypointConfidenceThreshold` 时标记为 unreliable。
- 核心关键点平均置信度低于 `frameConfidenceThreshold` 时，该帧不参与动作判断。
- 平滑算法 V1.0 可使用移动平均或指数平滑。

### 3.4 Center of Mass Estimation

职责：

- 基于 2D 关键点估算帧级重心。
- 输出重心坐标、置信度和轨迹。

V1.0 简化计算：

- torso_center = shoulders midpoint 与 hips midpoint 的平均。
- hips_center = left_hip 与 right_hip 的中点。
- limbs_center = elbows、wrists、knees、ankles 的有效关键点平均。
- center_of_mass = torso_center * 0.50 + hips_center * 0.30 + limbs_center * 0.20。

如果 limbs_center 无法可靠计算，则按有效权重归一化，不得输出虚假的确定值。

### 3.5 Movement Analysis

职责：

- 计算派生运动指标。
- 基于规则识别动作问题。
- 输出可解释的英文问题描述和建议。

V1.0 派生指标：

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

V1.0 问题类型：

| code | label | 触发依据 |
| --- | --- | --- |
| over_pulling_with_arms | Over-pulling with arms | 肘角快速减小，髋部/膝部位移不足 |
| late_hip_shift | Late hip shift | 伸手前髋部未向目标方向移动 |
| unstable_center_of_mass | Unstable center of mass | 短窗口内重心方向多次反转或速度波动过大 |
| poor_foot_engagement | Poor foot engagement | 膝/踝稳定性差，重心未压向支撑侧 |
| locked_elbow_too_early | Locked elbow too early | 手臂过早接近伸直或锁定，后续移动受限 |
| inefficient_reach | Inefficient reach | 伸手距离增加但身体位置未同步改善 |

### 3.6 Annotation Rendering

职责：

- 在关键帧图片上叠加骨架、重心、轨迹片段和英文批注。
- 导出标注图片。

V1.0 图片内容：

- 原始帧。
- 关键点和骨架线。
- 当前重心点。
- 最近一段重心轨迹。
- 问题标签和一句英文建议。

### 3.7 Result Export

职责：

- 输出分析 JSON。
- 输出关键帧标注图片。
- 保持输出路径和文件命名稳定，便于后续 UI 接入。

## 4. 配置规格

建议使用单个分析配置对象：

```json
{
  "targetFps": 10,
  "keypointConfidenceThreshold": 0.4,
  "frameConfidenceThreshold": 0.5,
  "smoothingWindow": 5,
  "analysisWindowFrames": 12,
  "maxAnnotatedFrames": 20,
  "language": "en"
}
```

约束：

- `language` 在 V1.0 固定为 `en`。
- 阈值必须可配置，不能硬编码到规则实现中。
- 配置缺失时使用默认值，并在输出 JSON 中记录最终使用的配置。

## 5. 输出 JSON Schema

V1.0 输出文件建议命名为 `analysis.json`。

```json
{
  "version": "1.0",
  "source": {
    "videoPath": "samples/climb.mp4",
    "durationMs": 12000,
    "fps": 30,
    "width": 1920,
    "height": 1080
  },
  "config": {
    "targetFps": 10,
    "keypointConfidenceThreshold": 0.4,
    "frameConfidenceThreshold": 0.5,
    "smoothingWindow": 5,
    "analysisWindowFrames": 12,
    "maxAnnotatedFrames": 20,
    "language": "en"
  },
  "frames": [
    {
      "frameIndex": 0,
      "timestampMs": 0,
      "reliable": true,
      "keypoints": [
        {
          "name": "left_shoulder",
          "x": 612.4,
          "y": 318.7,
          "confidence": 0.91
        }
      ],
      "centerOfMass": {
        "x": 700.1,
        "y": 520.6,
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
      "code": "late_hip_shift",
      "label": "Late hip shift",
      "startMs": 3400,
      "endMs": 4100,
      "confidence": 0.76,
      "severity": "medium",
      "evidence": {
        "primaryFrameIndex": 34,
        "metrics": {
          "hipDisplacement": 8.2,
          "wristReachDistance": 64.5
        }
      },
      "message": "Move your hips toward the next hold before reaching.",
      "recommendation": "Shift your center of mass over the support foot first, then initiate the reach."
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

## 6. 规则实现规格

### 6.1 通用规则要求

- 每条规则必须返回 `confidence`。
- 每条规则必须返回英文 `message` 和 `recommendation`。
- 规则不得基于低可靠帧输出强结论。
- 同一时间窗口内同类问题需要合并，避免重复刷屏。
- 规则阈值应从配置或规则配置表读取。

### 6.2 示例规则伪代码

```text
if elbow_angle_delta < -threshold
and hip_displacement < min_hip_displacement
and knee_angle_delta < min_knee_delta:
  emit over_pulling_with_arms
```

```text
if wrist_reach_distance increases
and hip_displacement_toward_reach_direction < min_pre_shift:
  emit late_hip_shift
```

```text
if center_of_mass_direction_changes >= max_direction_changes
or center_of_mass_velocity_variance >= max_velocity_variance:
  emit unstable_center_of_mass
```

## 7. 英文文案规格

英文输出应符合以下格式：

- Label：短标签，首字母大写，例如 `Late hip shift`。
- Message：一句话指出问题，例如 `Your hips moved after the reach started.`。
- Recommendation：一句可执行建议，例如 `Shift your center of mass over the support foot before extending your arm.`。

文案要求：

- 避免绝对化诊断，例如 `You are wrong`。
- 优先使用训练建议语气，例如 `Try...`、`Shift...`、`Keep...`。
- 低置信度时使用保守措辞，例如 `Possible...`。

## 8. 文件与目录约定

建议后续实现采用以下目录结构：

```text
src/
  video/
  pose/
  analysis/
  rendering/
  export/
  config/
samples/
outputs/
  analysis.json
  annotated/
```

说明：

- `video/` 负责视频读取和抽帧。
- `pose/` 负责姿态检测、关键点 schema 和平滑。
- `analysis/` 负责重心、指标和动作规则。
- `rendering/` 负责图片/视频叠加。
- `export/` 负责 JSON 和文件输出。
- `config/` 负责默认配置和规则阈值。

## 9. 验收标准

V1.0 完成时必须满足：

1. 给定一个本地攀岩视频，程序可完成分析流程并正常结束。
2. 输出 `analysis.json`，包含 source、config、frames、issues 和 artifacts。
3. 每个可靠帧包含关键点和重心数据。
4. 每个 issue 包含英文 label、message 和 recommendation。
5. 至少能识别 3 类动作问题：`over_pulling_with_arms`、`late_hip_shift`、`unstable_center_of_mass`。
6. 至少导出 1 张关键帧标注图。
7. 低置信度帧不会生成确定性动作问题。

## 10. 后续版本扩展点

### V1.1

- 建议模板库。
- 参考姿态骨架图。
- 更丰富的关键帧导出。

### V1.2

- 前端播放器。
- 问题时间轴。
- 自动暂停和跳转。
- 分析结果可视化面板。

### V2.0

- 增强视频导出。
- 正确姿态动画或骨架渲染。
- 路线和岩点可视化。
- 更完整的训练报告。
