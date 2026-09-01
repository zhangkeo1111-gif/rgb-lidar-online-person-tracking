# 原始物理标定基线影响审计

## 结论

将 Scene01 在线推理从历史 `du=+48 px` 改为原始
`K+D+T, du=dv=0` 后，三项 coverage 均明显下降。这证明历史 `+48`
此前参与了画面 Gate、点归属和 component 选择，并间接影响了 XYZ 与身份状态；
它不能再被描述为“仅显示偏移”。

该结果**不证明 `du=0` 是最终正确标定，也不证明 `+48` 是物理正确值**。
它只量化了移除经验像素偏移对当前算法链的影响。

## 可比范围

两次运行均使用：

- Scene01 同一 ROS bag 和同一开头区间；
- 200 个 RGB 消息、65 个 LiDAR 消息；
- 390 个检测；
- TRAIN_FIT-only；
- 无实时节流、headless、无录像；
- 持久身份开启；
- 同一冻结 YOLO、ReID、静态背景与 occupancy 资产。

旧对照产物：`outputs/projection_auto_validation_final/metrics.json`。
该运行在审核失败后仍把 `du=+48 px` 用于推理。

新基线产物：`outputs/physical_baseline_impact/metrics.json`。
该运行的物理推理为 `du=dv=0`，审核候选不写回，legacy 显示偏移关闭，
并且没有生成视频。

产物顶层状态为 `PURE_ONLINE_30HZ_FAIL`，原因仅是本次故意运行 200/65
短前缀而未满足完整 TRAIN_FIT 数量门禁（`declared_scope_count_pass=false`）；
其他运行门禁均通过。这个状态不代表投影审核候选被采用，也不改变下表的同口径比较。

## 指标定义

```text
C_component = components_available / detections
C_XYZ       = measurements / detections
C_identity  = person_persistent_identity_rows / person_rows
```

`C_identity` 是当前 200 帧 RGB 输出中人物行的固定身份覆盖率，不是身份准确率。

## 结果

| 指标 | 旧 `+48` 推理 | 新 `du=0` 基线 | 变化 |
| --- | ---: | ---: | ---: |
| `C_component` | 390/390 = **100.0%** | 325/390 = **83.33%** | **−16.67 pp** |
| `C_XYZ` | 390/390 = **100.0%** | 325/390 = **83.33%** | **−16.67 pp** |
| `C_identity` | 990/1000 = **99.0%** | 783/1000 = **78.3%** | **−20.7 pp** |
| 人物 XYZ 行覆盖 | 990/1000 = **99.0%** | 795/1000 = **79.5%** | **−19.5 pp** |
| 机器人 XY 行覆盖 | 198/200 = **99.0%** | 198/200 = **99.0%** | **0.0 pp** |

## 新残差审核结果

新运行仍保持物理推理 `du=0`。旁路审核给出诊断候选 `du=+36 px`，但门禁失败：

- 时间分块较稳定；
- 深度分区不一致：约 `+39.30 / +0.45 px`；
- 图像区域不一致：约 `+18.90 / +61.23 px`；
- 静态非语义边缘诊断约 `+24 px`；
- 状态为 `AUDIT_CANDIDATE_FAIL_REPORT_ONLY`；
- `candidate_applied_to_inference = false`。

这些不一致再次说明人物 component→bbox 残差不能直接当作传感器标定对应关系。

## 当前工程状态

```text
Physical inference: raw K+D+T, du=dv=0
Calibration status: baseline only, not proven final
Audit candidate writeback: disabled
Legacy +48 display: disabled by default; display/debug only
```

回归验证：`35 passed`。
