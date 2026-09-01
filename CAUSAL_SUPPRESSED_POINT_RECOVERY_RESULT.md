# Causal Suppressed-Point Recovery 结果

## 工程状态

当前生产在线链已接入唯一通过 VALIDATION 门禁的 `HISTORY_ONLY` 恢复策略，默认启用，可用 `--no-suppressed-recovery` 一键关闭。`K+D+T, du=dv=0`、原框、一点一框归属、冻结 temporal/occupancy prior 及正常组件结果均未改变；恢复只在 PERSON 正常组件缺失时运行。

恢复候选严格限制为：当前 PERSON 框没有正常组件；点通过 temporal suppression、仅被 occupancy suppression 删除；仍使用原始框和一点一框归属；中心投影必须落在原框内；3D 尺寸、点数、组件分数和组件唯一性通过；相邻人物框竞争时拒绝。`HISTORY_ONLY` 复用现有 OnlineTracker 运动门，不增加更宽 gate。

在线输出为每个 LiDAR 测量记录 `measurement_source=NORMAL_COMPONENT` 或 `RECOVERED_SUPPRESSED_COMPONENT`，汇总同时记录恢复 attempts/accepted/rejected，便于关闭开关做同帧回归。

## 同口径结果

所有预测先按时间顺序产生，官方人物 cuboid GT 在预测结束后才加载，只用于离线 3D Hungarian（1.5 m）评价。

| Scope / policy | Box component coverage | Precision | Recall | F1 | XY RMSE | XY P95 | 净 TP / FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN_FIT baseline | 76.99% | 0.823 | 0.627 | 0.712 | 0.362 m | 0.877 m | — |
| TRAIN_FIT HISTORY_ONLY | 77.70% | 0.825 | 0.634 | 0.717 | 0.367 m | 0.878 m | +34 / +0 |
| VALIDATION baseline | 76.79% | 0.882 | 0.666 | 0.759 | 0.359 m | 0.898 m | — |
| VALIDATION HISTORY_ONLY | **79.16%** | **0.885** | **0.690** | **0.775** | 0.370 m | **0.892 m** | **+68 / +0** |

VALIDATION 上，保守恢复带来 component coverage `+2.37 pp`、Recall `+2.33 pp`、F1 `+0.016`；没有新增 FP、没有重复占用 baseline 已匹配 GT、没有观察到 recovered track 的 GT identity transition。全体匹配的 XY RMSE 增加 0.0117 m，P95 反而下降 0.0060 m。

## 风险审核

68 个 VALIDATION 恢复匹配全部集中在 P4，主要窗口为 `1448–1457`、`1462–1475`、`1558–1560`、`1562–1563`、`1565–1603`（中间有少量未恢复帧）。恢复样本自身的 XY RMSE / median / P95 / max 为 `0.617 / 0.599 / 0.807 / 1.033 m`。它们确实在 1.5 m 门内补回 FN，但自身误差明显高于整体均值，因此不能把 coverage 改善描述为高精度恢复。

邻人竞争保护在 crossing/occlusion 窗口 `1360–1385`、`1403`、`1707–1719` 主动拒绝候选；接触图显示这些帧的人物框明显靠近或重叠，拒绝行为符合设计目标。

无历史的 `HISTORY_OR_3FRAME_SEED` 在 VALIDATION 虽将 box coverage 提至 91.79%，却新增 294 个 FP、9 次重复占用已有 GT、23 次邻人 identity transition；F1 从 0.759 降至 0.750，XY RMSE 从 0.359 m 升至 0.471 m。因此“连续稳定三帧”不能排除静态结构/邻人组件，该策略已判定失败，不具备上线资格。

另发现 legacy occupancy 分支把离地高度写成 `point_z - signed_ground_distance`；对当前近水平地面，这几乎等于地面绝对 Z，因此会把 occupancy 附近的人体中部点也当成低点。直接改为 signed ground distance 后，VALIDATION component coverage 从 76.79% 升到 95.44%，但新增 244 FP，XY RMSE 从 0.359 m 恶化到 0.555 m，因此该语义修正只保留为消融证据，未写入在线推理。基于修正高度候选的历史强门禁接受 0 项；三帧 seed 新增 287 FP，同样未上线。

## 在线 A/B 验证

在完全相同的前 4,500 RGB / 1,500 LiDAR 帧、CUDA、无视频、无 projection audit 条件下：

| 指标 | 关闭恢复 | 开启恢复 | 差值 |
| --- | ---: | ---: | ---: |
| LiDAR measurements | 6,388 | 6,475 | **+87** |
| Person measurements | 5,608 | 5,695 | **+87** |
| Person XYZ-valid RGB rows | 16,837 | 17,101 | **+264** |
| Person persistent-identity rows | 14,279 | 14,522 | **+243** |
| Anonymous person rows | 7,868 | 7,625 | **-243** |
| LiDAR total mean / P95 | 32.13 / 53.88 ms | 33.28 / 55.03 ms | **+1.15 / +1.15 ms** |

两次运行都处理完整指定前缀且没有视频帧或输入帧丢失。wall-clock 输出率受同机调度波动影响，未拿来归因模块开销；分阶段 LiDAR 计时显示恢复平均增加约 1.15 ms/帧。

## 结论

`HISTORY_ONLY` 已作为小范围、可关闭的在线修复接入；直接高度修正、严格形状门禁和无历史 seed 均保持禁用。当前结果能证明该策略在 Scene01 VALIDATION 和在线前缀中提高可用测量数，不能证明它已解决全部 occupancy 误杀，也不能外推到 Scene28 或新场景。

验证产物位于 `outputs/suppressed_point_recovery_audit/validation/`，包括逐候选 CSV、统一指标 JSON/CSV、匹配明细和三类视觉接触图。
