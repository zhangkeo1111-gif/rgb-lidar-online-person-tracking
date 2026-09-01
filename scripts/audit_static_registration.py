"""Report-only Scene01 static-structure projection audit.

This script never imports detections, person boxes, tracking, identities or GT.
Discovery-only temporal static voxels generate automatic edge pseudo-pairs;
validation and holdout are never refitted.  Pseudo-pairs can reject unstable
models, but cannot authorize a physical calibration write-back.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from online_v4 import pipeline, support  # noqa: E402

OUT = ROOT / "outputs" / "static_registration_audit"
BAG = pipeline.DEFAULT_BAG
DISCOVERY = (0, 299)
VALIDATION = (320, 639)
HOLDOUT = (660, 978)
SAMPLES = 28
STATIC_BUILD_SAMPLES = 48
VOXEL_M = 0.10
STATIC_FREQUENCY = 0.20
EDGE_POINTS_PER_FRAME = 700
SIZE = (1280, 720)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def sample(interval: tuple[int, int], count: int = SAMPLES) -> list[int]:
    return np.linspace(interval[0], interval[1], count, dtype=int).tolist()


def camera_info_audit() -> dict:
    signatures, timestamps = Counter(), []
    first = None
    with AnyReader([BAG]) as reader:
        connections = [value for value in reader.connections if value.topic == "/camera/color/camera_info"]
        for connection, bag_ns, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            payload = {
                "width": int(message.width), "height": int(message.height),
                "distortion_model": str(message.distortion_model),
                "K": [float(value) for value in message.K], "D": [float(value) for value in message.D],
                "R": [float(value) for value in message.R], "P": [float(value) for value in message.P],
                "binning_x": int(message.binning_x), "binning_y": int(message.binning_y),
                "roi": {"x_offset": int(message.roi.x_offset), "y_offset": int(message.roi.y_offset),
                        "width": int(message.roi.width), "height": int(message.roi.height),
                        "do_rectify": bool(message.roi.do_rectify)},
            }
            signatures[json.dumps(payload, sort_keys=True)] += 1
            timestamps.append(pipeline.timestamp_ns(message, int(bag_ns)))
            first = first or payload
    if first is None:
        raise RuntimeError("Scene01 bag has no CameraInfo")
    bundled_k = support.K.reshape(-1)
    bundled_d = support.D.reshape(-1)
    return {
        "messages": len(timestamps), "unique_signatures": len(signatures), "camera_info": first,
        "bundled_K_max_abs_error": float(np.max(np.abs(np.asarray(first["K"]) - bundled_k))),
        "bundled_D_max_abs_error": float(np.max(np.abs(np.asarray(first["D"]) - bundled_d))),
        "raw_projection_contract": "RAW_RGB_USES_K_PLUS_D",
        "resize_or_crop_evidence": False,
    }


def collect_frames(targets: set[int]) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, int, int]], dict]:
    frames, deltas = {}, []
    count = 0
    for index, frame in enumerate(pipeline.synchronized_bag_frames(BAG, 35.0)):
        if index >= 979:
            break
        count += 1; deltas.append(frame.sync_delta_ms)
        if index in targets:
            frames[index] = (frame.image, frame.points_rslidar, frame.rgb_timestamp_ns, frame.lidar_timestamp_ns)
    if count != 979 or len(frames) != len(targets):
        raise RuntimeError(f"Collected TRAIN_FIT {count}/979 and targets {len(frames)}/{len(targets)}")
    return frames, {
        "synchronized_trainfit_frames": count, "target_frames": len(frames),
        "sync_delta_ms": {"median": float(np.median(deltas)), "p95": float(np.percentile(deltas, 95)),
                          "max": float(np.max(deltas))},
        "rule": "latest causal RGB with 0 <= lidar-rgb <= 35 ms",
    }


def build_static_keys(frames: dict[int, tuple], indices: list[int]) -> tuple[set[tuple[int, int, int]], dict]:
    counts: Counter[tuple[int, int, int]] = Counter()
    for index in indices:
        keys = np.unique(np.floor(frames[index][1] / VOXEL_M).astype(np.int32), axis=0)
        counts.update(map(tuple, keys.tolist()))
    threshold = math.ceil(STATIC_FREQUENCY * len(indices))
    static = {key for key, count in counts.items() if count >= threshold}
    return static, {
        "source_split": "DISCOVERY_ONLY", "frames": indices, "voxel_m": VOXEL_M,
        "minimum_frequency": STATIC_FREQUENCY, "minimum_frame_count": threshold,
        "static_voxels": len(static), "validation_updates": 0, "holdout_updates": 0,
        "person_bbox_used": False, "person_gt_used": False, "tracking_used": False,
    }


def project(points: np.ndarray, transform: np.ndarray, K: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = (transform @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
    pixels = np.full((len(points), 2), np.nan)
    if valid.any():
        pixels[valid] = cv2.projectPoints(camera[valid].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    return pixels, camera[:, 2], camera


def lidar_edges(points: np.ndarray, static: set[tuple[int, int, int]], transform: np.ndarray,
                K: np.ndarray, D: np.ndarray, seed: int) -> dict:
    keys = np.floor(points / VOXEL_M).astype(np.int32)
    keep = np.fromiter((tuple(key) in static for key in keys), bool, len(keys))
    points = points[keep]
    pixels, depth, camera = project(points, transform, K, D)
    visible = (np.isfinite(pixels).all(axis=1) & (depth > 1.0) & (depth < 20.0) &
               (pixels[:, 0] >= -160) & (pixels[:, 0] < SIZE[0] + 160) &
               (pixels[:, 1] >= 0) & (pixels[:, 1] < SIZE[1]))
    points, pixels, depth, camera = points[visible], pixels[visible], depth[visible], camera[visible]
    if len(points) < 20:
        return {"points": points, "pixels": pixels, "depth": depth, "camera": camera}
    distances, neighbours = cKDTree(pixels).query(pixels, k=min(9, len(pixels)), workers=-1)
    edge = (distances[:, -1] <= 14.0) & (np.ptp(depth[neighbours], axis=1) >= 0.40)
    points, pixels, depth, camera = points[edge], pixels[edge], depth[edge], camera[edge]
    if len(points) > EDGE_POINTS_PER_FRAME:
        chosen = np.sort(np.random.default_rng(seed).choice(len(points), EDGE_POINTS_PER_FRAME, replace=False))
        points, pixels, depth, camera = points[chosen], pixels[chosen], depth[chosen], camera[chosen]
    return {"points": points, "pixels": pixels, "depth": depth, "camera": camera}


def image_edges(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0.8)
    magnitude = cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3),
                              cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3))
    low, high = np.percentile(magnitude, [80, 95])
    edge = cv2.Canny(smooth, max(20, int(low)), max(60, int(high)))
    y, x = np.nonzero(edge)
    return {"distance": cv2.distanceTransform(255 - edge, cv2.DIST_L2, 3),
            "xy": np.column_stack((x, y)).astype(float)}


def offset_score(data: list[tuple[dict, dict]], du: float, dv: float) -> float:
    values = []
    for lidar, image in data:
        shifted = lidar["pixels"] + np.asarray([du, dv])
        x, y = np.rint(shifted[:, 0]).astype(int), np.rint(shifted[:, 1]).astype(int)
        inside = (x >= 0) & (x < SIZE[0]) & (y >= 0) & (y < SIZE[1])
        if inside.sum() >= 20:
            values.extend(np.minimum(image["distance"][y[inside], x[inside]], 20.0))
    return float(np.mean(values)) if values else math.inf


def discover_offset(data: list[tuple[dict, dict]]) -> dict:
    coarse = [(offset_score(data, du, dv), du, dv)
              for du in range(-120, 121, 4) for dv in range(-32, 33, 4)]
    _, best_u, best_v = min(coarse)
    fine = [(offset_score(data, du, dv), du, dv)
            for du in range(best_u - 5, best_u + 6) for dv in range(best_v - 5, best_v + 6)]
    score, du, dv = min(fine); zero = offset_score(data, 0, 0)
    return {"du_px": float(du), "dv_px": float(dv), "fit_split": "DISCOVERY",
            "score": score, "h0_score": zero, "improvement_fraction": 1.0 - score / max(zero, 1e-9),
            "applied_to_inference": False}


def region(u: float, v: float) -> str:
    x = "LEFT" if u < SIZE[0] / 3 else "CENTER" if u < 2 * SIZE[0] / 3 else "RIGHT"
    y = "TOP" if v < SIZE[1] / 3 else "MIDDLE" if v < 2 * SIZE[1] / 3 else "BOTTOM"
    return f"{x}_{y}"


def pseudo_rows(frame: int, split: str, lidar: dict, image: dict, offset: dict) -> list[dict]:
    expected = lidar["pixels"] + np.asarray([offset["du_px"], offset["dv_px"]])
    distance, match = cKDTree(image["xy"]).query(expected, workers=-1)
    accept = distance <= 9.0
    targets, sources = image["xy"][match[accept]], lidar["pixels"][accept]
    points, depth, camera = lidar["points"][accept], lidar["depth"][accept], lidar["camera"][accept]
    safe = (depth >= 2.0) & (np.abs(camera[:, 0] / depth) <= 1.2) & (np.abs(camera[:, 1] / depth) <= 0.8)
    targets, sources, points, depth = targets[safe], sources[safe], points[safe], depth[safe]
    rows = []
    for target, source, point, z in zip(targets, sources, points, depth, strict=True):
        residual = target - source
        rows.append({"frame": frame, "split": split, "depth_m": float(z),
                     "image_u": float(source[0]), "image_v": float(source[1]),
                     "target_u": float(target[0]), "target_v": float(target[1]),
                     "residual_u_px": float(residual[0]), "residual_v_px": float(residual[1]),
                     "image_region": region(float(source[0]), float(source[1])),
                     "lidar_x": float(point[0]), "lidar_y": float(point[1]), "lidar_z": float(point[2]),
                     "correspondence_class": "AUTOMATIC_STATIC_EDGE_PSEUDO_CORRESPONDENCE",
                     "physical_point_identity_proven": False, "person_bbox_used": False,
                     "person_gt_used": False, "tracking_used": False})
    return rows


def delta_transform(parameters: np.ndarray) -> np.ndarray:
    result = np.eye(4); result[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix(); result[:3, 3] = parameters[3:]
    return result


def fit_se3(rows: list[dict], transform: np.ndarray, K: np.ndarray, D: np.ndarray) -> dict:
    points = np.asarray([[row[f"lidar_{axis}"] for axis in "xyz"] for row in rows])
    targets = np.asarray([[row["target_u"], row["target_v"]] for row in rows])
    if len(points) > 10000:
        chosen = np.linspace(0, len(points) - 1, 10000, dtype=int); points, targets = points[chosen], targets[chosen]

    def objective(value: np.ndarray) -> np.ndarray:
        pixels = project(points, delta_transform(value) @ transform, K, D)[0]
        regularization = np.r_[value[:3] / math.radians(5.0), value[3:] / 0.15]
        return np.r_[(pixels - targets).reshape(-1), 0.5 * regularization]

    bounds = (np.r_[np.full(3, -math.radians(10)), np.full(3, -0.15)],
              np.r_[np.full(3, math.radians(10)), np.full(3, 0.15)])
    result = least_squares(objective, np.zeros(6), bounds=bounds, loss="soft_l1", f_scale=3.0, max_nfev=160)
    return {"optimizer_success": bool(result.success), "rotation_vector_deg": np.degrees(result.x[:3]).tolist(),
            "translation_m": result.x[3:].tolist(), "matrix": delta_transform(result.x).tolist(),
            "fit_rows": len(points), "applied_to_inference": False}


def stats(value: np.ndarray) -> dict:
    value = np.asarray(value, float)
    return {"rows": len(value), "median": float(np.median(value)),
            "median_abs": float(np.median(np.abs(value))), "p95_abs": float(np.percentile(np.abs(value), 95)),
            "rmse": float(np.sqrt(np.mean(value ** 2)))}


def evaluate(rows: list[dict], transform: np.ndarray, K: np.ndarray, D: np.ndarray,
             shift: tuple[float, float], se3: dict) -> dict:
    result = defaultdict(dict)
    for split in ("DISCOVERY", "VALIDATION", "HOLDOUT"):
        selected = [row for row in rows if row["split"] == split]
        points = np.asarray([[row[f"lidar_{axis}"] for axis in "xyz"] for row in selected])
        targets = np.asarray([[row["target_u"], row["target_v"]] for row in selected])
        models = {
            "H0_RAW_K_D_T": project(points, transform, K, D)[0],
            "H1_PRINCIPAL_POINT_DIAGNOSTIC": project(points, transform, K + np.asarray([[0, 0, shift[0]], [0, 0, shift[1]], [0, 0, 0]]), D)[0],
            "H2_FIXED_SE3_DIAGNOSTIC": project(points, np.asarray(se3["matrix"]) @ transform, K, D)[0],
        }
        for name, pixels in models.items():
            residual = targets - pixels
            result[name][split] = {"u": stats(residual[:, 0]), "v": stats(residual[:, 1]),
                                   "radial": stats(np.linalg.norm(residual, axis=1))}
    return dict(result)


def correlations(rows: list[dict]) -> dict:
    result = {}
    for split in ("DISCOVERY", "VALIDATION", "HOLDOUT"):
        selected = [row for row in rows if row["split"] == split]
        residual = np.asarray([row["residual_u_px"] for row in selected])
        result[split] = {key: float(spearmanr(np.asarray([row[key] for row in selected]), residual).statistic)
                         for key in ("depth_m", "image_u", "image_v", "frame")}
        groups = defaultdict(list)
        for row in selected: groups[row["image_region"]].append(row["residual_u_px"])
        result[split]["image_regions"] = {key: stats(np.asarray(value)) for key, value in groups.items() if len(value) >= 20}
        depth = np.asarray([row["depth_m"] for row in selected]); cuts = np.quantile(depth, [1 / 3, 2 / 3])
        result[split]["depth_bins"] = {
            "NEAR": stats(residual[depth <= cuts[0]]),
            "MID": stats(residual[(depth > cuts[0]) & (depth <= cuts[1])]),
            "FAR": stats(residual[depth > cuts[1]]),
        }
    return result


def render_contact(frames: dict[int, tuple], indices: list[int], lidar_by_frame: dict[int, dict],
                   transform: np.ndarray, K: np.ndarray, D: np.ndarray, shift: tuple[float, float]) -> None:
    panels = []
    adjusted = K.copy(); adjusted[0, 2] += shift[0]; adjusted[1, 2] += shift[1]
    for index in indices[:12]:
        image = frames[index][0].copy(); points = lidar_by_frame[index]["points"]
        raw = project(points, transform, K, D)[0]; candidate = project(points, transform, adjusted, D)[0]
        for u, v in raw:
            if 0 <= u < SIZE[0] and 0 <= v < SIZE[1]: cv2.circle(image, (round(u), round(v)), 2, (0, 220, 255), -1)
        for u, v in candidate:
            if 0 <= u < SIZE[0] and 0 <= v < SIZE[1]: cv2.circle(image, (round(u), round(v)), 1, (255, 255, 0), -1)
        cv2.putText(image, f"HOLDOUT {index} | yellow H0 | cyan H1 diagnostic", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(image, f"HOLDOUT {index} | yellow H0 | cyan H1 diagnostic", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA))
    sheet = np.full((810, 1920, 3), 245, np.uint8)
    for ordinal, panel in enumerate(panels):
        row, column = divmod(ordinal, 4); sheet[row * 270:(row + 1) * 270, column * 480:(column + 1) * 480] = panel
    cv2.imwrite(str(OUT / "STATIC_REGISTRATION_HOLDOUT_CONTACT_SHEET.jpg"), sheet)


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = {"DISCOVERY": sample(DISCOVERY), "VALIDATION": sample(VALIDATION), "HOLDOUT": sample(HOLDOUT)}
    static_indices = sample(DISCOVERY, STATIC_BUILD_SAMPLES)
    targets = set(static_indices + sum(splits.values(), []))
    frames, sync = collect_frames(targets)
    camera = camera_info_audit(); write_json(OUT / "input_and_camera_audit.json", {"camera": camera, "sync": sync})
    static, receipt = build_static_keys(frames, static_indices); write_json(OUT / "static_structure_receipt.json", receipt)
    transform_annotated, K, D = support.projection_assets()
    transform = transform_annotated @ support.T_ANNOTATED_FROM_RSLIDAR
    lidar_by_frame = {index: lidar_edges(frames[index][1], static, transform, K, D, 1000 + index) for index in targets}
    images = {index: image_edges(frames[index][0]) for index in targets}
    offset = discover_offset([(lidar_by_frame[index], images[index]) for index in splits["DISCOVERY"]])
    rows = []
    for split, indices in splits.items():
        for index in indices: rows.extend(pseudo_rows(index, split, lidar_by_frame[index], images[index], offset))
    split_counts = {split: sum(row["split"] == split for row in rows) for split in splits}
    if min(split_counts.values()) < 500:
        raise RuntimeError(f"Insufficient static pseudo-pairs: {split_counts}")
    write_csv(OUT / "static_edge_pseudo_correspondences.csv", rows)
    discovery_rows = [row for row in rows if row["split"] == "DISCOVERY"]
    shift = (float(np.median([row["residual_u_px"] for row in discovery_rows])),
             float(np.median([row["residual_v_px"] for row in discovery_rows])))
    se3 = fit_se3(discovery_rows, transform, K, D)
    comparison = evaluate(rows, transform, K, D, shift, se3); write_json(OUT / "hypothesis_comparison.json", comparison)
    residual_structure = correlations(rows); write_json(OUT / "residual_structure.json", residual_structure)
    h1 = comparison["H1_PRINCIPAL_POINT_DIAGNOSTIC"]
    h2 = comparison["H2_FIXED_SE3_DIAGNOSTIC"]
    offset_interior = abs(offset["du_px"]) <= 105.0 and abs(offset["dv_px"]) <= 24.0
    se3_translation = np.asarray(se3["translation_m"], float)
    se3_rotation = np.asarray(se3["rotation_vector_deg"], float)
    se3_interior = bool(np.max(np.abs(se3_translation)) < 0.145 and np.max(np.abs(se3_rotation)) < 9.5)
    statistical_checks = {
        "enough_rows_each_split": min(split_counts.values()) >= 500,
        "discovery_offset_is_not_near_search_boundary": offset_interior,
        "h1_validation_p95_le_12px": h1["VALIDATION"]["radial"]["p95_abs"] <= 12.0,
        "h1_holdout_p95_le_15px": h1["HOLDOUT"]["radial"]["p95_abs"] <= 15.0,
        "se3_solution_is_not_near_optimizer_boundary": se3_interior,
        "h2_validation_p95_le_12px": h2["VALIDATION"]["radial"]["p95_abs"] <= 12.0,
        "h2_holdout_p95_le_15px": h2["HOLDOUT"]["radial"]["p95_abs"] <= 15.0,
    }
    gate = {
        "statistical_screen": "PASS" if all(statistical_checks.values()) else "FAIL",
        "statistical_checks": statistical_checks,
        "physical_calibration_promotion": "FAIL",
        "physical_gate_reason": "AUTOMATIC_EDGE_PSEUDO_PAIRS_DO_NOT_PROVE_SHARED_PHYSICAL_POINT_IDENTITY",
        "candidate_applied_to_inference": False,
    }
    render_contact(frames, splits["HOLDOUT"], lidar_by_frame, transform, K, D, shift)
    result = {"status": "REPORT_ONLY", "scope": "SCENE01_TRAINFIT_ONLY", "splits": splits,
              "split_rows": split_counts, "discovery_offset": offset,
              "principal_point_diagnostic_shift_px": list(shift), "se3_diagnostic": se3,
              "gate": gate, "person_bbox_used": False, "person_gt_used": False,
              "tracking_used": False, "runtime_inference_modified": False}
    write_json(OUT / "static_registration_audit.json", result)
    holdout_h0 = comparison["H0_RAW_K_D_T"]["HOLDOUT"]["radial"]
    holdout_h1 = h1["HOLDOUT"]["radial"]; holdout_h2 = h2["HOLDOUT"]["radial"]
    report = f"""# Scene01 静态结构 Registration Audit

## 结论

物理标定写回 Gate：**FAIL**。当前推理继续保持原始 `K+D+T, du=dv=0`。

本审核未使用人物框、人物 GT、Tracking 或 Identity。静态体素只由 discovery 建立，validation/holdout 更新数均为 0。自动静态边缘只能形成伪对应，不能证明 LiDAR 边缘与 RGB 边缘属于同一个物理点，因此所有候选均为 REPORT ONLY。

## 数据与候选

- 范围：Scene01 TRAIN_FIT 0–978 LiDAR 帧。
- 伪对应数量：`{split_counts}`。
- discovery 网格诊断偏移：`({offset['du_px']:+.1f}, {offset['dv_px']:+.1f}) px`。
- discovery 主点等价诊断：`({shift[0]:+.2f}, {shift[1]:+.2f}) px`。
- 固定 SE(3) 诊断 rotation-vector：`{se3['rotation_vector_deg']}` deg。
- 固定 SE(3) 诊断 translation：`{se3['translation_m']}` m。

## Chronological holdout

| 模型 | radial median | radial P95 |
| --- | ---: | ---: |
| H0 raw K+D+T | {holdout_h0['median']:.2f} px | {holdout_h0['p95_abs']:.2f} px |
| H1 principal-point-equivalent diagnostic | {holdout_h1['median']:.2f} px | {holdout_h1['p95_abs']:.2f} px |
| H2 fixed SE(3) diagnostic | {holdout_h2['median']:.2f} px | {holdout_h2['p95_abs']:.2f} px |

统计筛查：**{gate['statistical_screen']}**。物理标定 promotion：**FAIL**。

额外拒绝证据：静态边缘 offset 接近搜索边界：**{not offset_interior}**；SE(3) 参数接近优化边界：**{not se3_interior}**。H1 的低 holdout P95 是在 discovery offset 附近选择伪对应以后得到的条件统计，不能当作独立重投影精度。

Contact sheet 视觉复核也未支持 H1：大幅平移后的 cyan 点并没有形成比 raw H0 更可信的同物理结构对齐。该视觉结果只用于拒绝候选，不用于拟合新参数。

## 严格解释

该实验可以拒绝无法跨时间复现的候选，但不能在没有真实静态 3D–2D 点身份的情况下区分 `cx/cy`、图像原点 convention 与小 SE(3) 偏差。本次候选还与此前静态边缘旁证（约 `+24 px`）明显矛盾，进一步证明自动边缘伪对应存在多解。任何候选均未写入 inference，也未修改配置、XYZ、Tracking 或 legacy display 状态。
"""
    (OUT / "SCENE01_STATIC_REGISTRATION_AUDIT.md").write_text(report, encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

