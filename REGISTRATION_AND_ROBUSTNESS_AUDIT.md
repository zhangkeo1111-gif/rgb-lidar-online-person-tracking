# Scene01 Registration 与 Clustering 鲁棒性审计

## 当前决定

```text
物理推理：raw K + D + T, du=dv=0
标定候选：REPORT ONLY
legacy +48：DISPLAY/DEBUG ONLY，默认关闭
软门控：不写入运行时
静态背景策略：不修改
```

本轮没有恢复 `+48`，没有改 K/D/T、XYZ、Tracking、point ownership、静态先验或运行配置。

## 1. 静态结构 calibration audit

审核覆盖完整 Scene01 TRAIN_FIT 0–978 LiDAR 帧，按时间划分 discovery、validation 和 holdout。静态体素只在 discovery 建立；validation/holdout 更新数为 0。人物框、人物 GT、Tracking 和 Identity 均未使用。

自动静态边缘搜索得到：

- discovery offset：`(+115, -34) px`，接近搜索边界；
- 主点等价诊断：`(+115.04, -34.00) px`；
- SE(3) translation 的 Z 分量达到优化上界 `+0.15 m`；
- SE(3) holdout radial P95：`42.42 px`，失败；
- 物理标定 promotion：**FAIL**；
- `candidate_applied_to_inference=false`。

虽然主点等价候选在条件化 holdout 伪对应上得到 `7.94 px` P95，但这些 RGB 边缘是在 discovery offset 附近自动选择的，并没有证明与 LiDAR 边缘属于同一个物理点。该结果还与此前约 `+24/+36/+48 px` 的证据明显冲突。因此它是边缘匹配多解，不是新的标定解。

Contact sheet 的实际视觉复核同样没有显示 `(+115,-34)` 比 raw H0 更符合相同静态结构；视觉证据只用于否决，不参与候选拟合。

结论：在没有独立静态 3D–2D 点身份的情况下，当前数据仍不能区分 `cx/cy`、图像原点 convention 和小 SE(3) 偏差。

## 2. `du=0` 软 point ownership 审计

在同一 65 LiDAR / 390 detections 前缀上测试了对称水平 uncertainty band；扩展量按 bbox 宽度归一化，不包含方向性 pixel shift，每个点最多仍只有一个 owner。

| 水平扩展 | Component coverage | Guarded coverage | 原框内中心 | 歧义率 |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 83.33% | 83.33% | 100.00% | 2.15% |
| 5% | 83.33% | 83.33% | 100.00% | 2.77% |
| 10% | 83.33% | 83.08% | 99.69% | 5.23% |
| 15% | 83.33% | 77.18% | 92.62% | 7.69% |
| 20% | 83.33% | 65.13% | 78.15% | 10.46% |
| 25% | 83.33% | 63.08% | 76.00% | 14.46% |

安全推荐仍为 **0%**。小范围扩展没有恢复任何 component；较大扩展只增加了歧义和框外中心。

## 3. 真实丢失位置

65 个 baseline miss 全部属于 PERSON：

- 原始 `du=0` 框内在静态抑制前均有足够点；
- 每个 miss 在抑制前拥有点数中位数为 **89**；
- 静态抑制和高度过滤后中位数仅 **1**；
- `65/65` 全部跌到三点 component 阈值以下。

因此这段前缀的 83.33% coverage 不是因为框内没有投影点，也不是 0–25% 的 ownership band 太窄；缺失发生在冻结背景/occupancy suppression。

## 4. 静态抑制 ablation

| 策略 | Raw coverage | Guarded coverage | 说明 |
| --- | ---: | ---: | --- |
| Temporal + occupancy（当前） | 83.33% | 83.33% | 当前基线 |
| Temporal only | 96.92% | 91.79% | 关闭 occupancy 后恢复较多，但出现非可信尺寸组件 |
| Occupancy only | 87.95% | 83.33% | 关闭 temporal 仅提高 raw coverage，guarded 无提升 |
| None | 100.00% | 77.69% | 100% coverage 伴随明显质量下降 |

这说明 occupancy suppression 是当前前缀损失的主要来源，但直接关闭它并不安全；完全关闭 suppression 虽然恢复 100%，可信组件反而少于当前基线。

## 5. 下一步门禁

1. 标定侧：只有获得具有物理点身份的静态 3D–2D 对应关系后，才重新比较 principal point 与固定 SE(3)。自动最近边缘不再具备 promotion 权限。
2. 聚类侧：停止扩大视锥。下一候选应是**仅对被 temporal/occupancy 判为静态、但在 RGB PERSON 框内持续形成一致三维组件的点做因果恢复**，而不是整体关闭先验。
3. 该候选必须在独立时间段验证 XYZ 误差、邻人污染、静态物体误检和 P95；不能只看 coverage。

## 产物

- `outputs/static_registration_audit/SCENE01_STATIC_REGISTRATION_AUDIT.md`
- `outputs/static_registration_audit/hypothesis_comparison.json`
- `outputs/static_registration_audit/STATIC_REGISTRATION_HOLDOUT_CONTACT_SHEET.jpg`
- `outputs/soft_ownership_audit/SOFT_OWNERSHIP_AUDIT.md`
- `outputs/soft_ownership_audit/variant_metrics.csv`
- `outputs/soft_ownership_audit/suppression_ablation.csv`
