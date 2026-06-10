# 攀岩 AI 分析系统开发规格说明

## 1. 规格目标

本文档基于 `mr.md`，用于指导后续工程开发。首个开发目标是交付 V1.0 最小可用闭环：输入本地攀岩视频，输出姿态关键点、重心轨迹、动作问题日志和关键帧标注图。

## 2. V1.0 范围

### 2.1 必须实现

- 读取本地视频文件。
- 按配置抽帧。
- 对每帧进行姿态关键点检测；若检测到多人，按规则选定唯一主攀者后只分析该单人轨迹。
- 过滤低置信度关键点和低质量帧。
- 对关键点轨迹进行基础平滑。
- 计算帧级 2D 重心坐标。
- 基于规则识别首批动作问题（V1.0 仅含不依赖岩点的四类，见 §3.5）。
- 输出结构化 JSON 分析结果。
- 导出关键问题帧标注图片，批注内容使用英文。

说明：检测器本身可输出多人；V1.0 在每帧选取一个主攀者（见 §3.2），下游模块（平滑、重心、动作分析）只处理这一条单人轨迹。

### 2.2 暂不实现

- 多人场景分析。
- 3D 姿态估计。
- 岩点自动识别。
- Web 交互播放器。
- 增强视频完整导出。
- 正确姿态动画渲染。

### 2.3 技术栈与运行环境

- 语言：Python。**运行时锁定 3.11 或 3.12 虚拟环境**。本机系统 Python 为 3.14，torch / ultralytics 暂无 3.14 轮子，不可直接用系统解释器。
- 姿态检测：Ultralytics YOLO11-pose（输出 COCO-17 关键点）。
- 推理后端：PyTorch；Apple Silicon（如 M4 Max）使用 `mps` 设备，缺失时回退 `cpu`。
- 视频解码：OpenCV（`opencv-python`），依赖系统 `ffmpeg`（本机缺失，需 `brew install ffmpeg`）。
- 交付形态：核心为 Python 库 API，另提供瘦 CLI 入口 `python -m climbanalyze <video> [--config config.json]`。
- 关键点适配：检测器输出经适配层映射到 §3.2 的 13 点 schema（COCO-17 为其超集，丢弃眼/耳）。后续可替换检测器，只需实现同一适配接口。

### 2.4 坐标与单位约定

- 所有关键点、重心、位移类指标使用**归一化坐标**：`x_norm = x_px / width`，`y_norm = y_px / height`，取值 0~1。原点在左上角，y 向下。
- JSON 中坐标均为归一化值；`source.width` / `source.height` 保留原始像素尺寸，供需要时还原。
- 角度类指标单位为度（degree）。
- 速度类指标单位为 归一化距离 / 秒。
- 顶层字段 `coordinateSpace` 固定为 `"normalized"`，标明坐标系，便于消费方解析。

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

帧索引与时间戳约定：

- 抽帧后重新编号，`frameIndex` 为**抽帧序列的连续索引**（0,1,2,…），非原视频帧号。
- `timestampMs` 基于原视频时间轴：`round(originalFrameNo / sourceFps * 1000)`，保证可映射回原视频做跳转。
- 抽帧策略：按 `targetFps` 对原 `sourceFps` 做等间隔采样（`step = round(sourceFps / targetFps)`，至少 1）。
- 同时保留 `originalFrameIndex`（原视频帧号），便于精确定位与调试。

### 3.2 Pose Detection

职责：

- 对每帧检测人体关键点。
- 输出关键点坐标（归一化）和置信度。
- 在多人检测结果中选定唯一主攀者，下游只分析该单人轨迹。

主攀者选定规则（V1.0，确定性）：

1. 主键：选 bbox 归一化面积最大的人体。
2. 平手（面积差 < 5%）：取核心关键点平均置信度更高者。
3. 仍平手：取上一帧主攀者 bbox 中心最近者（保持轨迹连续）；首帧则取索引更小者。

V1.0 不做跨帧多目标跟踪（ReID），逐帧独立选定即可。

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

V1.0 简化计算（三个分段互不重叠，避免重复计权）：

- shoulders_center = left_shoulder 与 right_shoulder 的中点。
- hips_center = left_hip 与 right_hip 的中点。
- limbs_center = elbows、wrists、knees、ankles 中有效关键点的平均。
- center_of_mass = shoulders_center * 0.35 + hips_center * 0.45 + limbs_center * 0.20。

权重依据：躯干+髋部质量最大，故肩/髋合计 0.80 且髋略高；四肢合计 0.20。

缺失处理：

- 若某分段关键点不足以计算（如 limbs_center 无有效点），剔除该分段，对剩余分段权重**重新归一化**后再加权，不得用 0 或臆造值填充。
- 重心 `confidence` = 参与计算的关键点置信度按其权重的加权平均。
- 若 shoulders_center 与 hips_center 任一缺失，该帧重心标记为不可靠，不参与动作判断。

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

V1.0 问题类型（均不依赖岩点/目标方向）：

| code | label | 触发依据 |
| --- | --- | --- |
| over_pulling_with_arms | Over-pulling with arms | 肘角快速减小，髋部/膝部位移不足 |
| unstable_center_of_mass | Unstable center of mass | 短窗口内重心方向多次反转或速度波动过大 |
| poor_foot_engagement | Poor foot engagement | 膝/踝稳定性差，重心未压向支撑侧 |
| locked_elbow_too_early | Locked elbow too early | 手臂过早接近伸直或锁定，后续移动受限 |

V1.1 推迟问题类型（依赖目标方向，需岩点提示或伸手方向代理，V1.0 不实现）：

| code | label | 触发依据 |
| --- | --- | --- |
| late_hip_shift | Late hip shift | 伸手前髋部未向目标方向移动 |
| inefficient_reach | Inefficient reach | 伸手距离增加但身体位置未同步改善 |

severity 分级（V1.0 按规则违反程度的归一化幅度 `m`，各规则定义见 §6.3）：

- `m < 0.33` → `low`
- `0.33 ≤ m < 0.66` → `medium`
- `m ≥ 0.66` → `high`

issue `confidence` 取以下两者乘积：

- 触发窗口内参与判断帧的平均关键点/重心置信度。
- 规则违反幅度的归一化分数 `m`（越超阈值越高，封顶 1.0）。

低于 `keypointConfidenceThreshold` 衍生的不可靠帧不计入，且若窗口内可靠帧占比 < 50%，不得发出该 issue。

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

约束：

- `language` 在 V1.0 固定为 `en`。
- `device` 取 `auto` | `mps` | `cpu`；`auto` 在 Apple Silicon 优先 `mps`，否则 `cpu`。
- 位移/速度阈值均为归一化坐标单位（见 §2.4），角度为度。
- 阈值必须可配置，不能硬编码到规则实现中；规则从 `ruleThresholds[code]` 读取。
- 配置缺失时使用默认值，并在输出 JSON 中记录最终使用的完整配置（含 `ruleThresholds`）。

## 5. 输出 JSON Schema

V1.0 输出文件建议命名为 `analysis.json`。

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
    "ruleThresholds": { "...": "见 §4，输出时写入完整内容" }
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

字段说明：

- `coordinateSpace`：固定 `"normalized"`（见 §2.4）。
- `status`：`ok` | `no_person_detected` | `low_quality`（多数帧不可靠仍出 JSON 但加 warning）。致命错误见 §11，不产出该文件。
- `warnings`：非致命提示数组，如低光照、可靠帧比例偏低、岩点不可用。
- 坐标字段（keypoints、centerOfMass）均为 0~1 归一化值。

## 6. 规则实现规格

### 6.1 通用规则要求

- 每条规则必须返回 `confidence`。
- 每条规则必须返回英文 `message` 和 `recommendation`。
- 规则不得基于低可靠帧输出强结论。
- 同一时间窗口内同类问题需要合并，避免重复刷屏。
- 规则阈值应从配置或规则配置表读取。

### 6.2 示例规则伪代码

```text
if elbow_angle_delta < elbowAngleDeltaDeg          # 肘角快速减小（上拉）
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
if ankle_jitter > maxAnkleJitter                   # 踝点抖动大 = 踩点不稳
and com_shift_toward_support < minComShiftToSupport:
  emit poor_foot_engagement
```

```text
if elbow_angle > lockedElbowAngleDeg               # 过早接近伸直/锁定
and com_progress_after_lock < minPostLockComProgress:
  emit locked_elbow_too_early
```

`late_hip_shift` 与 `inefficient_reach` 的伪代码推迟到 V1.1（依赖目标方向）。

### 6.3 派生量与默认阈值

窗口定义（窗口长度 = `analysisWindowFrames`，滑动步长 1 帧）：

- `center_of_mass_velocity`：相邻可靠帧重心位移 / 帧间隔秒数（归一化单位/秒）。
- `center_of_mass_direction_changes`：窗口内重心速度向量方向反转次数（与前一向量夹角 > 90° 记一次）。
- `center_of_mass_velocity_variance`：窗口内速度大小的方差。
- `elbow_angle_delta` / `knee_angle_delta`：窗口首尾角度差（度）。
- `hip_displacement`：窗口内髋部中点位移幅度（归一化）。
- `ankle_jitter`：窗口内踝点位置标准差（归一化）。
- `com_shift_toward_support`：重心朝支撑侧（位置更低的脚）水平移动量（归一化）。
- `com_progress_after_lock`：锁肘后重心净位移（归一化）。

默认阈值集中见 §4 `ruleThresholds`，不得在规则代码内硬编码。规则违反幅度 `m`（用于 severity 与 confidence）= `clamp(超阈程度 / 标定满刻度, 0, 1)`，每条规则定义自己的满刻度（例如 over_pulling 以 `elbow_angle_delta` 比阈值再低 30° 为满刻度）。

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
src/climbanalyze/
  __init__.py
  __main__.py        # CLI 入口：python -m climbanalyze <video>
  pipeline.py        # 串联各模块的端到端流程
  video/             # 视频读取、抽帧
  pose/              # 姿态检测、COCO-17→schema 适配、平滑
  analysis/          # 重心、派生指标、动作规则
  rendering/         # 图片/视频叠加
  export/            # JSON 和文件输出
  config/            # 默认配置和规则阈值
samples/
outputs/
  analysis.json
  annotated/
```

说明：

- `__main__.py` 解析 CLI 参数，加载配置，调用 `pipeline`。
- `pipeline.py` 负责模块编排与错误处理（见 §11）。
- `video/` 负责视频读取和抽帧。
- `pose/` 负责姿态检测、关键点 schema、检测器适配层和平滑。
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
5. 至少能识别 3 类不依赖岩点的动作问题：`over_pulling_with_arms`、`unstable_center_of_mass`、`locked_elbow_too_early`（或 `poor_foot_engagement`）。
6. 至少导出 1 张关键帧标注图。
7. 低置信度帧不会生成确定性动作问题。
8. 坐标输出为归一化值，`coordinateSpace` = `normalized`，`source` 含原始 width/height。
9. 各类错误输入按 §11 返回对应退出码，不崩溃。

## 10. 后续版本扩展点

### V1.1

- `late_hip_shift`、`inefficient_reach` 规则（引入岩点提示或伸手方向代理后实现）。
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

## 11. 错误处理与退出码

致命错误：打印英文错误信息到 stderr，不产出 `analysis.json`，返回非 0 退出码。

| 退出码 | 场景 | 处理 |
| --- | --- | --- |
| 0 | 正常完成 | 产出完整 JSON（`status` 可能为 `ok` 或 `low_quality`） |
| 2 | 视频路径不存在或格式不支持 | stderr 报错，退出 |
| 3 | 视频损坏/无法解码（含 ffmpeg 缺失） | stderr 报错，退出 |
| 4 | 全片未检测到人体 | 产出最小 JSON，`status = no_person_detected`，退出码 4 |
| 5 | 配置文件非法（JSON 解析失败或字段类型错误） | stderr 报错，退出 |

非致命降级（退出码 0，写入 `warnings`）：

- 可靠帧占比 < 50% → `status = low_quality`，仍产出 JSON 但不强行发问题结论。
- 部分帧无人体或低置信 → 标记该帧 `reliable=false`，跳过其动作判断。
- `device` 请求 `mps` 但不可用 → 回退 `cpu`，加 warning。

幂等与输出：同一输入重复运行覆盖 `outputs/`；标注图命名 `issue_<id>.jpg` 稳定可预测。
