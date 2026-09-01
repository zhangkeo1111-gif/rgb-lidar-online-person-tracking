# Coordinate System and Projection Status

## 2026-08-28 当前生产更新

Scene01 已恢复经验像素配准，但不再沿用旧的 `(+48,0)`。TRAIN_FIT 二维
搜索并在冻结后通过 VALIDATION 门禁的值为 `(+64,+36)`，现用于 point
ownership、component selection、3D–2D association 与默认圆柱显示。
`project_physical()` 的 `du=dv=0` 仍作为原始物理标定审计基线；该经验值
不是物理外参/内参更新，也不得迁移到 Scene28 或其他场景。下文关于“当前
零偏移推理”的内容记录的是恢复经验配准之前的审计阶段。

## 1. 当前结论

当前问题应拆成两层，不能统称为“三维坐标错了”：

1. **三维坐标链**：目前没有发现矩阵顺序、左右手性或轴方向颠倒的直接证据。人物/机器人跟踪状态、地面平面和聚类都位于 `annotated_data_product` 坐标系。
2. **三维投影到 RGB 后的二维残差**：已确认存在，但不同证据给出的残差不一致。它可能同时包含标定误差、图像原点误差、目标语义差异、遮挡和错误 cluster，不能直接等同于相机外参错误。

因此当前没有批准修改三维 XYZ、K/D 或外参。在线物理推理已恢复为
`K+D+T, du=dv=0`，但该状态只能称为**原始物理标定基线**，不能描述为已证明正确的最终标定。历史 `+48 px` 仅保留为默认关闭的显示/调试选项。

## 2. 坐标约定

统一采用：

```text
p_target = T_target_from_source @ p_source
```

| 坐标系 | 用途 | 主要数据 |
| --- | --- | --- |
| `rslidar` | 原始 LiDAR 传感器坐标系 | `/rslidar_points` 原始点云 |
| `annotated_data_product` | 聚类、地面平面、Tracking/Identity 的公共三维坐标系 | 人物 XYZ、机器人 XY、轨迹、圆柱三维底座 |
| `camera_color` | 相机机体坐标系 | frame graph 中间坐标系 |
| `camera_color_optical_frame` | OpenCV 投影坐标系 | `+X` 向右、`+Y` 向下、`+Z` 为相机前方深度 |
| RGB 像素坐标 | 1280×720 原始图像 | `u` 向右、`v` 向下 |

`Z_cam` 指三维点变换到 `camera_color_optical_frame` 后的第三个分量。只有 `Z_cam > 0` 才能合法投影；它是相机前向深度，不是人物世界坐标 Z，也不是由 RGB 框估计出来的深度。

## 3. 当前三条独立链

```text
Physical inference
p_rslidar -> p_annotated -> p_optical -> K+D, du=dv=0
-> image/frustum gate -> point ownership -> component selection
-> 3D–2D association -> XYZ / Tracking

Projection residual audit
raw physical projection -> residual diagnostics -> chronological holdout
-> REPORT ONLY; candidate never enters inference

Legacy visualization
physical projection -> optional +48 px -> DEBUG / DISPLAY ONLY
```

代码中的组合关系为：

```text
T_optical_from_annotated
= inverse(T_color_from_optical)
@ inverse(T_rslidar_from_color)
@ inverse(T_annotated_from_rslidar)
```

在线点云先进入 `annotated_data_product`，再使用 `T_optical_from_annotated` 投影。自动审核中的原始点云直接使用：

```text
T_optical_from_rslidar
= T_optical_from_annotated @ T_annotated_from_rslidar
```

两条写法数学等价。当前代码不再存在“公共 `du`”：所有推理步骤显式走
`project_physical()` 的零偏移结果；只有中心标记与圆柱渲染可走
`project_display()`。审核候选和 legacy 显示偏移都不能改变 XYZ。

## 4. Scene01 当前证据

下面是历史 `+48` 推理版本产生的诊断证据，保留用于解释拆链原因。诊断只使用最前面的 60 个因果 LiDAR 更新，前 42 帧拟合、后 18 帧留出验证，不使用运行时 GT、未来帧或逐帧修正。

| 检查 | 结果 | 判断 |
| --- | ---: | --- |
| 时间分块所需 `du` | `57.45 / 57.78 / 58.00 / 59.65 px` | 稳定，PASS |
| 深度分区所需 `du` | `57.04 / 59.02 px` | 稳定，PASS |
| 图像区域所需 `du` | `57.04 / 85.95 px` | 差异 28.91 px，FAIL |
| 留出段中位绝对残差 | `+48: 11.35 px`；候选 `+58: 2.86 px` | 候选改善 74.8%，PASS |
| 非语义静态边缘旁证 | 约 `+24 px` | 与人物残差不一致，仅作旁证 |

旧版本当时的状态：

```text
FALLBACK_FROZEN_GATE_FAIL (historical artifact)
inference du = +48 px (removed from current inference)
```

这里的人物残差是“可见 LiDAR component 中心 → RGB bbox 水平中心”。可见 component 中心不等于人体 cuboid 几何中心，也不一定等于 bbox 中心。遮挡、截断、姿态和错误 cluster 都会进入这个残差，所以它是任务对齐证据，不是纯标定对应点。

## 5. Scene28 当前证据

Scene28 的正式根因审核使用的是静态结构证据，与 Scene01 的人物 component→bbox 残差口径不同，数值不能直接横向比较。

- RGB 与 CameraInfo 均为原始 1280×720，未发现 resize、crop、ROI、binning 或时间戳错配。
- 矩阵旋转为右手系，链重组误差为 0，发现集静态点的正光学深度比例为 1.0。
- 原始 K+D 投影的静态结构水平残差约 `+88 px`，跨主要深度、区域和时间基本稳定。
- 旧诊断值 `+68 px` 会留下约 `+20 px` 中位水平残差，因此不是 Scene28 的解。
- 自由主点诊断可以吸收残差，但在没有独立物理来源时，它与后投影像素平移不可区分。
- 固定 SE(3) 外参候选未通过尾部误差门禁，因此没有批准修改物理模型，也没有运行后续聚类/Tracking。

Scene28 当前 Gate 仍为 `FAIL`。

## 6. 已排除、未排除与不能混淆的内容

### 已有证据不支持

- 简单的 `x/y/z` 轴交换或左右手坐标系错误；
- RGB 被 resize/crop 后仍错误使用原始 K；
- 把 raw RGB 错当成 rectified RGB；
- CameraInfo 与 RGB 时间戳系统性错位；
- 单纯依靠一个跨数据集固定 `du` 解决所有场景。

### 仍未证明

- 官方教程外参与实际采集时的相机–LiDAR 刚体关系是否完全一致；
- 是否存在未记录的图像原点/主点约定；
- Scene01 右侧区域的偏差来自外参旋转，还是遮挡、截断或错误 cluster；
- Scene01 的 `+48` 是否对应任何真实物理参数。

### 必须区分

- 修改 `du` 或主点 `cx`：改变二维投影，不改变三维 XYZ；
- 修改外参：会改变所有点的相机坐标和投影，属于物理模型变化；
- 修改 `Person XYZ`/轨迹：改变跟踪结果，不能用于掩盖投影错误；
- 把圆柱移到 bbox 中心：只能改善显示，不能证明三维定位正确。

## 7. 跨数据集规则

同一套物理传感器原则上应共享一套经过独立验证的 K/D/外参，而不是每个数据集人工调像素值。

当前 `+48` 是 Scene01 历史经验显示值，**不应作为其他数据集的默认值**。其他数据集应按以下规则运行：

1. 加载该数据集记录的 CameraInfo 和正确 frame graph；
2. 以原始物理标定 `du=0` 作为基线；
3. 自动残差审核只生成报告，不写回推理；
4. 只有独立静态 3D–2D 对应关系在时间、深度、图像区域和 chronological holdout 上均通过，才允许另行审批 K/外参更新；
5. 审批前保持 `du=0` 原始基线，并明确标记其尚未证明为最终标定；
6. 禁止继承 Scene01 `+48`、逐帧调整、分身份调整或使用 TEST/GT 选择参数。

## 8. 当前代码与证据位置

- 投影和坐标矩阵：`src/online_v2/support.py`
- 只报告的投影残差审核：`src/online_v2/registration.py`
- 在线公共投影调用：`src/online_v2/pipeline.py`
- 30 Hz 实时运行与审计：`src/online_v2/runtime.py`
- 审核/显示隔离配置：`configs/projection_residual_audit.json`
- Scene01 历史 `+48` 推理对照：`outputs/projection_auto_validation_final/metrics.json`
- 当前运行审核输出：`outputs/<run>/projection_residual_audit.json`
- 历史 `+48` 来源审核：`PROJECTION_OFFSET_AUDIT.md`
- Scene28 根因审核：`D:\navwareset_scene01_clean\outputs\navwareset_scene28\projection_rootcause\SCENE28_PROJECTION_ROOT_CAUSE_AUDIT.md`
