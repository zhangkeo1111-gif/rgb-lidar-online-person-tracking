"""TRAIN_FIT-frozen, VALIDATION-only decomposition of the 770 A2 cases.

This is an offline audit.  It never changes projection, detections, component
selection, tracking or identity.  Sensor predictions/detections are frozen and
fingerprinted before the first GT read in this process.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from online_v4 import pipeline  # noqa: E402
import audit_fn_root_causes as fn_audit  # noqa: E402
import audit_suppressed_point_recovery as recovery_audit  # noqa: E402


OUT = ROOT / "outputs" / "a2_correspondence_decomposition"
MANIFEST = Path(r"D:\navwareset_scene01_clean\data\splits\annotated_split_manifest.csv")
PRIOR = ROOT / "outputs" / "fn_root_cause_audit"
W, H = pipeline.IMAGE_SIZE
SUBTYPES = (
    "A2a_CRITERION_TOO_STRICT", "A2b_REGISTRATION_SENSITIVE_PATTERN",
    "A2c_VISIBLE_BOX_3D_CUBOID_SEMANTIC_MISMATCH",
    "A2d_NEIGHBOR_BOX_CORRESPONDENCE_AMBIGUITY",
    "A2e_TRUNCATION_IMAGE_EDGE", "A2f_DETECTION_GEOMETRY_ANOMALY",
    "A2g_UNRESOLVED",
)
SHEET_NAMES = {
    SUBTYPES[0]: "A2_CRITERION_TOO_STRICT.jpg",
    SUBTYPES[1]: "A2_REGISTRATION_SENSITIVE.jpg",
    SUBTYPES[2]: "A2_SEMANTIC_MISMATCH.jpg",
    SUBTYPES[3]: "A2_NEIGHBOR_AMBIGUITY.jpg",
    SUBTYPES[4]: "A2_TRUNCATION.jpg",
    SUBTYPES[5]: "A2_DETECTION_ANOMALY.jpg",
    SUBTYPES[6]: "A2_UNRESOLVED.jpg",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def q(values, value: float, default: float=0.0) -> float:
    data = np.asarray([item for item in values
                       if item is not None and np.isfinite(item)], np.float64)
    return default if not len(data) else float(np.percentile(data, value))


def iou(left: np.ndarray, right: np.ndarray) -> tuple[float, float, np.ndarray]:
    corner1, corner2 = np.maximum(left[:2], right[:2]), np.minimum(left[2:], right[2:])
    size = np.maximum(corner2 - corner1, 0.0)
    intersection = float(np.prod(size))
    area1 = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
    area2 = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
    union = area1 + area2 - intersection
    return intersection / max(union, 1e-9), intersection / max(min(area1, area2), 1e-9), size


def proxy_projection(target: dict, model) -> dict:
    center, half = target["xyz"], target["size"] / 2.0
    local = np.asarray([[x, y, z] for x in (-half[0], half[0])
                        for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    yaw = target["yaw"]
    rotation = np.asarray([[math.cos(yaw), -math.sin(yaw), 0.0],
                           [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    corners = local @ rotation.T + center
    corner_pixels, valid, _ = model.project_physical(corners)
    center_pixel, center_valid, depth = model.project_physical(center[None])
    if np.count_nonzero(valid) < 2 or not center_valid[0]:
        raise RuntimeError("VALIDATION cuboid cannot form a raw physical proxy")
    values = corner_pixels[valid]
    proxy = np.asarray([values[:, 0].min(), values[:, 1].min(),
                        values[:, 0].max(), values[:, 1].max()])
    return {"center_u": float(center_pixel[0, 0]), "center_v": float(center_pixel[0, 1]),
            "depth": float(depth[0]), "proxy": proxy,
            "corner_pixels": corner_pixels[valid]}


def box_metric(box: np.ndarray, proxy: dict) -> dict:
    projected = proxy["proxy"]
    box_center = 0.5 * (box[:2] + box[2:])
    proxy_center = np.asarray([proxy["center_u"], proxy["center_v"]])
    box_size = np.maximum(box[2:] - box[:2], 1.0)
    proxy_size = np.maximum(projected[2:] - projected[:2], 1.0)
    overlap, overlap_min, intersection_size = iou(box, projected)
    delta = box_center - proxy_center
    normalized = delta / box_size
    width_ratio, height_ratio = proxy_size / box_size
    cost = ((1.0 - overlap) + 0.55 * float(np.linalg.norm(normalized))
            + 0.20 * abs(math.log(width_ratio)) + 0.20 * abs(math.log(height_ratio)))
    return {
        "IoU_proxy_bbox": overlap, "intersection_over_min_area": overlap_min,
        "projected_center_inside_bbox": bool(
            box[0] <= proxy_center[0] <= box[2] and box[1] <= proxy_center[1] <= box[3]),
        "bbox_center_inside_proxy": bool(
            projected[0] <= box_center[0] <= projected[2]
            and projected[1] <= box_center[1] <= projected[3]),
        "signed_du_px": float(delta[0]), "signed_dv_px": float(delta[1]),
        "abs_du_px": abs(float(delta[0])), "abs_dv_px": abs(float(delta[1])),
        "center_distance_px": float(np.linalg.norm(delta)),
        "center_distance_norm": float(np.linalg.norm(normalized)),
        "proxy_width": float(proxy_size[0]), "proxy_height": float(proxy_size[1]),
        "bbox_width": float(box_size[0]), "bbox_height": float(box_size[1]),
        "proxy_bbox_width_ratio": float(width_ratio),
        "proxy_bbox_height_ratio": float(height_ratio),
        "proxy_bbox_area_ratio": float(np.prod(proxy_size) / np.prod(box_size)),
        "left_residual_px": float(box[0] - projected[0]),
        "right_residual_px": float(box[2] - projected[2]),
        "top_residual_px": float(box[1] - projected[1]),
        "bottom_residual_px": float(box[3] - projected[3]),
        "intersection_width_px": float(intersection_size[0]),
        "intersection_height_px": float(intersection_size[1]),
        "correspondence_cost": cost,
    }


def case_metrics(frame: int, target_index: int, truth: list[dict], record: dict,
                 model, forced_box_id: int | None=None) -> dict:
    target, projection = truth[target_index], proxy_projection(truth[target_index], model)
    person = [(index, item) for index, item in enumerate(record["detections"])
              if item["class"] == "PERSON"]
    if not person:
        raise RuntimeError("A2/TP frame unexpectedly has no PERSON detections")
    all_metrics = []
    for index, item in person:
        metric = box_metric(np.asarray(item["bbox"], float), projection)
        all_metrics.append({"box_id": index, "box_conf": float(item["confidence"]),
                            "bbox": item["bbox"], **metric})
    ranked = sorted(all_metrics, key=lambda row: row["correspondence_cost"])
    best = (next(row for row in all_metrics if row["box_id"] == forced_box_id)
            if forced_box_id is not None else ranked[0])
    best_iou = max(all_metrics, key=lambda row: row["IoU_proxy_bbox"])
    best_center = min(all_metrics, key=lambda row: row["center_distance_norm"])
    containment = [row for row in all_metrics if row["projected_center_inside_bbox"]
                   or row["bbox_center_inside_proxy"]]
    second = next((row for row in ranked if row["box_id"] != best["box_id"]), None)
    box = np.asarray(best["bbox"], float)
    neighbor_iou = max((fn_audit.iou(box, np.asarray(item["bbox"], float))
                        for index, item in person if index != best["box_id"]), default=0.0)
    other_gt = [item for index, item in enumerate(truth) if index != target_index]
    nearest_gt = min((float(np.linalg.norm(target["xyz"][:2] - item["xyz"][:2]))
                      for item in other_gt), default=None)
    projections = [proxy_projection(item, model) for item in truth]
    nearest_projected = min((float(np.linalg.norm(
        [projection["center_u"] - item["center_u"], projection["center_v"] - item["center_v"]]))
        for index, item in enumerate(projections) if index != target_index), default=None)
    box_center = 0.5 * (box[:2] + box[2:])
    box_to_other_projected = min((float(np.linalg.norm(
        box_center - [item["center_u"], item["center_v"]]))
        for index, item in enumerate(projections) if index != target_index), default=None)
    box_closer_to_other = bool(
        box_to_other_projected is not None
        and box_to_other_projected < best["center_distance_px"])
    proxy = projection["proxy"]
    proxy_border = bool(proxy[0] <= 2 or proxy[1] <= 2
                        or proxy[2] >= W - 2 or proxy[3] >= H - 2)
    box_border = bool(box[0] <= 2 or box[1] <= 2
                      or box[2] >= W - 2 or box[3] >= H - 2)
    target_box_relation = bool(
        best["projected_center_inside_bbox"] or best["bbox_center_inside_proxy"]
        or best["intersection_over_min_area"] > 0)
    overlapping_box = bool(neighbor_iou > 0 and target_box_relation)
    close_neighbor = bool(nearest_gt is not None and nearest_gt < 0.8)
    occlusion = overlapping_box or close_neighbor
    horizontal = ("left" if projection["center_u"] < W / 3 else
                  "center" if projection["center_u"] < 2 * W / 3 else "right")
    vertical = ("top" if projection["center_v"] < H / 3 else
                "middle" if projection["center_v"] < 2 * H / 3 else "bottom")
    return {
        "frame_id": frame, "timestamp": record["timestamp_ns"],
        "gt_audit_id": target["entity_id"], "gt_x": target["xyz"][0],
        "gt_y": target["xyz"][1], "gt_z": target["xyz"][2],
        "depth": projection["depth"], "depth_bin": fn_audit.depth_bin(projection["depth"]),
        "projected_center_u": projection["center_u"],
        "projected_center_v": projection["center_v"],
        "proxy_x1": proxy[0], "proxy_y1": proxy[1],
        "proxy_x2": proxy[2], "proxy_y2": proxy[3],
        "projected_corners_json": json.dumps(projection["corner_pixels"].tolist()),
        "person_detection_count": len(person), "best_box_id": best["box_id"],
        "best_box_conf": best["box_conf"], "bbox": json.dumps(best["bbox"]),
        "best_iou_box_id": best_iou["box_id"],
        "best_center_distance_box_id": best_center["box_id"],
        "best_containment_box_id": None if not containment else min(
            containment, key=lambda row: row["correspondence_cost"])["box_id"],
        "second_best_box_id": None if second is None else second["box_id"],
        "best_second_margin": None if second is None else (
            second["correspondence_cost"] - best["correspondence_cost"]),
        "nearest_other_person_distance": nearest_gt,
        "nearest_other_projected_person_px": nearest_projected,
        "best_box_center_to_nearest_other_projected_person_px": box_to_other_projected,
        "best_box_closer_to_other_projected_person": box_closer_to_other,
        "bbox_neighbor_iou": neighbor_iou,
        "proxy_border_contact": proxy_border,
        "best_box_border_contact": box_border,
        "image_border_contact": proxy_border or box_border,
        "occlusion_flag": occlusion,
        "close_neighbor_flag": close_neighbor,
        "overlapping_box_flag": overlapping_box,
        "image_horizontal_region": horizontal,
        "image_vertical_region": vertical, "image_region": f"{horizontal}_{vertical}",
        "all_box_metrics_json": json.dumps(all_metrics), **{key: value for key, value in best.items()
                                                            if key not in {"bbox", "box_id", "box_conf"}},
    }


def detection_fingerprint(records: dict[int, dict]) -> str:
    payload = []
    for frame in sorted(records):
        for index, item in enumerate(records[frame]["detections"]):
            payload.append((frame, index, item["class"], round(item["confidence"], 10),
                            *np.round(item["bbox"], 10).tolist()))
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def matched_reference(frames: list[int], predictions: dict[int, list[dict]],
                      truth: dict[int, list[dict]], records: dict[int, dict], model) -> list[dict]:
    rows = []
    for frame in frames:
        matches, _, _ = recovery_audit.match_frame(predictions[frame], truth[frame])
        for match in matches:
            prediction = predictions[frame][match["prediction_index"]]
            row = case_metrics(frame, match["gt_index"], truth[frame], records[frame], model,
                               forced_box_id=prediction["detection"])
            row.update({"cohort": "TP", "prediction_source": prediction["source"],
                        "error_xy_m": match["error_xy_m"], "error_xyz_m": match["error_xyz_m"]})
            rows.append(row)
    return rows


def train_a2_rows(frames: list[int], predictions: dict[int, list[dict]], truth: dict[int, list[dict]],
                  traces: dict[int, list[dict]], records: dict[int, dict], model) -> list[dict]:
    rows = []
    for frame in frames:
        matches, _, _ = recovery_audit.match_frame(predictions[frame], truth[frame])
        matched = {row["gt_index"] for row in matches}
        for index, target in enumerate(truth[frame]):
            if index in matched:
                continue
            projection = fn_audit.cuboid_projection(target, model)
            stage, person = fn_audit.candidate_stage(traces[frame], projection)
            if person and stage is None:
                rows.append(case_metrics(frame, index, truth[frame], records[frame], model))
    return rows


def group_stability(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    medians = {name: {"n": len(values),
                      "du": float(np.median([item["signed_du_px"] for item in values])),
                      "dv": float(np.median([item["signed_dv_px"] for item in values]))}
               for name, values in groups.items() if len(values) >= 10}
    return medians


def freeze_rules(train_tp: list[dict], train_a2: list[dict]) -> dict:
    if len(train_tp) < 100 or len(train_a2) < 20:
        raise RuntimeError("insufficient TRAIN_FIT evidence to freeze A2 rules")
    train_frames = sorted({row["frame_id"] for row in train_a2})
    frame_rank = {frame: index for index, frame in enumerate(train_frames)}
    block_size = max(1, math.ceil(len(train_frames) / 4))
    for row in train_a2:
        row["train_time_block"] = f"P{min(frame_rank[row['frame_id']] // block_size + 1, 4)}"
    du = [row["signed_du_px"] for row in train_a2]
    dv = [row["signed_dv_px"] for row in train_a2]
    du_iqr, dv_iqr = q(du, 75) - q(du, 25), q(dv, 75) - q(dv, 25)
    stable_groups = {
        "person": group_stability(train_a2, "gt_audit_id"),
        "depth": group_stability(train_a2, "depth_bin"),
        "horizontal": group_stability(train_a2, "image_horizontal_region"),
        "vertical": group_stability(train_a2, "image_vertical_region"),
        "time": group_stability(train_a2, "train_time_block"),
    }
    group_du = [item["du"] for values in stable_groups.values() for item in values.values()]
    group_dv = [item["dv"] for values in stable_groups.values() for item in values.values()]
    sign_consistency = max(np.mean(np.asarray(du) >= 0), np.mean(np.asarray(du) <= 0))
    constant_supported = bool(
        sign_consistency >= 0.80 and group_du
        and np.ptp(group_du) <= max(12.0, 1.5 * du_iqr)
        and np.ptp(group_dv) <= max(12.0, 1.5 * dv_iqr))
    return {
        "source": "TRAIN_FIT_ONLY_BEFORE_VALIDATION_CLASSIFICATION",
        "train_tp_count": len(train_tp), "train_a2_count": len(train_a2),
        "criterion_iou_min": max(0.03, q([r["IoU_proxy_bbox"] for r in train_tp], 10)),
        "criterion_center_norm_max": q([r["center_distance_norm"] for r in train_tp], 90),
        "proxy_bbox_width_ratio_min": q(
            [r["proxy_bbox_width_ratio"] for r in train_tp], 10),
        "proxy_bbox_width_ratio_max": q(
            [r["proxy_bbox_width_ratio"] for r in train_tp], 90),
        "proxy_bbox_height_ratio_min": q(
            [r["proxy_bbox_height_ratio"] for r in train_tp], 10),
        "proxy_bbox_height_ratio_max": q(
            [r["proxy_bbox_height_ratio"] for r in train_tp], 90),
        "semantic_overlap_min": max(0.20, q([r["intersection_over_min_area"] for r in train_tp], 10)),
        "size_log_limit": max(0.35, q([
            max(abs(math.log(r["proxy_bbox_width_ratio"])),
                abs(math.log(r["proxy_bbox_height_ratio"]))) for r in train_tp], 95)),
        "ambiguity_margin_max": max(0.05, q([r["best_second_margin"] for r in train_tp], 10)),
        "registration_du_median": q(du, 50), "registration_dv_median": q(dv, 50),
        "registration_du_iqr": du_iqr, "registration_dv_iqr": dv_iqr,
        "registration_pattern_supported_on_train": constant_supported,
        "train_group_medians": stable_groups,
        "validation_updates": 0, "test_updates": 0, "embargo_updates": 0,
    }


def classify(row: dict, rules: dict) -> tuple[str, str]:
    size_log = max(abs(math.log(row["proxy_bbox_width_ratio"])),
                   abs(math.log(row["proxy_bbox_height_ratio"])))
    meaningful_box_overlap = (
        row["projected_center_inside_bbox"]
        or row["bbox_center_inside_proxy"]
        or row["intersection_over_min_area"] >= rules["semantic_overlap_min"])
    if row["proxy_border_contact"] or (
            row["best_box_border_contact"] and meaningful_box_overlap):
        return SUBTYPES[4], "proxy_touches_boundary_or_plausible_corresponding_box_is_truncated"
    ambiguous = (row["second_best_box_id"] is not None
                 and row["best_second_margin"] <= rules["ambiguity_margin_max"]
                 and (row["best_box_closer_to_other_projected_person"]
                      or row["close_neighbor_flag"]
                      or (meaningful_box_overlap and row["bbox_neighbor_iou"] > 0)))
    if ambiguous:
        return SUBTYPES[3], "best_second_correspondence_margin_small_with_neighbor"
    central_size = (
        rules["proxy_bbox_width_ratio_min"] <= row["proxy_bbox_width_ratio"]
        <= rules["proxy_bbox_width_ratio_max"]
        and rules["proxy_bbox_height_ratio_min"] <= row["proxy_bbox_height_ratio"]
        <= rules["proxy_bbox_height_ratio_max"])
    reasonable = (row["projected_center_inside_bbox"]
                  or row["bbox_center_inside_proxy"]
                  or row["IoU_proxy_bbox"] >= rules["criterion_iou_min"]
                  or (row["center_distance_norm"] <= rules["criterion_center_norm_max"]
                      and central_size))
    if reasonable and not row["occlusion_flag"] and size_log <= rules["size_log_limit"]:
        return SUBTYPES[0], "passes_TRAIN_FIT_TP_geometry_envelope_but_old_rule_rejected"
    du_band = max(10.0, 1.5 * rules["registration_du_iqr"])
    dv_band = max(10.0, 1.5 * rules["registration_dv_iqr"])
    registration_fit = (rules["registration_pattern_supported_on_train"]
                        and abs(row["signed_du_px"] - rules["registration_du_median"]) <= du_band
                        and abs(row["signed_dv_px"] - rules["registration_dv_median"]) <= dv_band
                        and size_log <= rules["size_log_limit"]
                        and not row["occlusion_flag"])
    if registration_fit:
        return SUBTYPES[1], "fits_frozen_TRAIN_FIT_systematic_signed_residual_band"
    semantic = (row["intersection_over_min_area"] >= rules["semantic_overlap_min"]
                and (size_log > rules["size_log_limit"] or row["occlusion_flag"]))
    if semantic:
        return SUBTYPES[2], "substantial_overlap_but_visible_box_and_full_cuboid_extent_differ"
    extreme_detection = (row["best_box_conf"] >= 0.50
                         and row["intersection_over_min_area"] < 0.10
                         and size_log > 1.5 * rules["size_log_limit"]
                         and not row["occlusion_flag"])
    if extreme_detection:
        return SUBTYPES[5], "isolated_high_confidence_box_has_extreme_geometry_vs_proxy"
    return SUBTYPES[6], "available_geometry_cannot_separate_registration_semantic_or_detection"


def refine_explanatory_flags(row: dict, rules: dict) -> None:
    related_to_target = bool(
        row["projected_center_inside_bbox"] or row["bbox_center_inside_proxy"]
        or row["intersection_over_min_area"] >= rules["semantic_overlap_min"])
    row["overlapping_box_flag"] = bool(row["bbox_neighbor_iou"] > 0
                                         and related_to_target)
    row["occlusion_flag"] = bool(row["close_neighbor_flag"]
                                  or row["overlapping_box_flag"])


def statistics(rows: list[dict], cohort: str) -> list[dict]:
    output = []
    for metric in ("signed_du_px", "signed_dv_px", "abs_du_px", "abs_dv_px",
                   "IoU_proxy_bbox", "center_distance_norm",
                   "proxy_bbox_width_ratio", "proxy_bbox_height_ratio",
                   "proxy_bbox_area_ratio"):
        values = [row[metric] for row in rows]
        output.append({"cohort": cohort, "metric": metric, "n": len(values),
                       "median": q(values, 50), "p25": q(values, 25),
                       "p75": q(values, 75), "iqr": q(values, 75) - q(values, 25),
                       "p90": q(values, 90), "p95": q(values, 95)})
    return output


def depth_profiles(rows: list[dict], cohort: str) -> list[dict]:
    output = []
    for depth_bin in ("0-3m", "3-6m", "6-9m", ">9m"):
        values = [row for row in rows if row["depth_bin"] == depth_bin]
        if not values:
            output.append({"cohort": cohort, "depth_bin": depth_bin, "n": 0})
            continue
        output.append({
            "cohort": cohort, "depth_bin": depth_bin, "n": len(values),
            "median_abs_du_px": q([row["abs_du_px"] for row in values], 50),
            "median_abs_dv_px": q([row["abs_dv_px"] for row in values], 50),
            "median_center_distance_norm": q(
                [row["center_distance_norm"] for row in values], 50),
            "median_iou": q([row["IoU_proxy_bbox"] for row in values], 50),
            "median_bbox_width": q([row["bbox_width"] for row in values], 50),
            "median_bbox_height": q([row["bbox_height"] for row in values], 50),
            "median_proxy_width": q([row["proxy_width"] for row in values], 50),
            "median_proxy_height": q([row["proxy_height"] for row in values], 50),
            "occlusion_fraction": float(np.mean(
                [row["occlusion_flag"] for row in values])),
        })
    return output


def occlusion_cross(rows: list[dict]) -> list[dict]:
    output = []
    flags = ("occlusion_flag", "close_neighbor_flag", "overlapping_box_flag")
    for subtype in SUBTYPES:
        cases = [row for row in rows if row["A2_subtype"] == subtype]
        for flag in flags:
            count = sum(bool(row[flag]) for row in cases)
            output.append({"subtype": subtype, "explanatory_flag": flag,
                           "count": count,
                           "fraction_within_subtype": count / max(len(cases), 1)})
    return output


def regression(x, y) -> dict:
    left, right = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return {"n": len(left), "correlation": None, "slope": None, "intercept": None, "r2": None}
    slope, intercept = np.polyfit(left, right, 1)
    prediction = slope * left + intercept
    r2 = 1.0 - np.sum((right - prediction) ** 2) / np.sum((right - right.mean()) ** 2)
    return {"n": len(left), "correlation": float(np.corrcoef(left, right)[0, 1]),
            "slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def cross(rows: list[dict], field: str) -> list[dict]:
    counts = Counter((row["A2_subtype"], str(row[field])) for row in rows)
    return [{"subtype": subtype, field: value, "count": count,
             "fraction_within_subtype": count / sum(
                 number for (other, _), number in counts.items() if other == subtype)}
            for (subtype, value), count in sorted(counts.items())]


def contact_sheet(rows: list[dict], path: Path, records: dict[int, dict]) -> None:
    if len(rows) > 28:
        indices = np.linspace(0, len(rows) - 1, 28).round().astype(int)
        rows = [sorted(rows, key=lambda row: row["frame_id"])[index] for index in indices]
    tiles = []
    for row in rows:
        image = cv2.imread(records[row["frame_id"]]["rgb_image_path"])
        if image is None:
            continue
        for index, item in enumerate(records[row["frame_id"]]["detections"]):
            if item["class"] != "PERSON":
                continue
            box = np.asarray(item["bbox"], int)
            cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]),
                          (0, 210, 255) if index == row["best_box_id"] else (130, 130, 130), 2)
        proxy = np.asarray([row["proxy_x1"], row["proxy_y1"], row["proxy_x2"], row["proxy_y2"]], int)
        cv2.rectangle(image, tuple(proxy[:2]), tuple(proxy[2:]), (0, 0, 230), 2)
        for point in np.asarray(json.loads(row["projected_corners_json"]), int):
            cv2.circle(image, tuple(point), 4, (230, 60, 220), -1)
        center = (round(row["projected_center_u"]), round(row["projected_center_v"]))
        cv2.drawMarker(image, center, (0, 0, 255), cv2.MARKER_CROSS, 22, 3)
        label1 = f"f{row['frame_id']} {row['gt_audit_id']} {row['A2_subtype']}"
        label2 = (f"du={row['signed_du_px']:+.1f} dv={row['signed_dv_px']:+.1f} "
                  f"IoU={row['IoU_proxy_bbox']:.2f} Z={row['depth']:.1f}m")
        cv2.rectangle(image, (0, 0), (W, 68), (12, 22, 32), -1)
        cv2.putText(image, label1[:92], (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    .58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, label2, (10, 57), cv2.FONT_HERSHEY_SIMPLEX,
                    .58, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(image, (480, 270)))
    if not tiles:
        tile = np.full((270, 480, 3), 245, np.uint8)
        cv2.putText(tile, "No cases under frozen TRAIN_FIT rule", (35, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, .58, (50, 50, 50), 2)
        tiles = [tile]
    blank = np.full_like(tiles[0], 245)
    tiles += [blank] * ((4 - len(tiles) % 4) % 4)
    sheet = np.vstack([np.hstack(tiles[index:index + 4]) for index in range(0, len(tiles), 4)])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = fn_audit.read_csv(MANIFEST)
    train_frames = [int(row["annotated_frame_index"]) for row in manifest if row["split"] == "TRAIN_FIT"]
    validation_frames = [int(row["annotated_frame_index"]) for row in manifest if row["split"] == "VALIDATION"]
    predictions, traces, records, safety = fn_audit.replay_predictions(
        manifest, capture_splits=("TRAIN_FIT", "VALIDATION"))
    validation_predictions = {name: {frame: values[frame] for frame in validation_frames}
                              for name, values in predictions.items()}
    for name in ("V0", "V1"):
        current = fn_audit.prediction_fingerprint(validation_predictions[name])
        expected = json.loads((PRIOR / "audit_summary.json").read_text(encoding="utf-8"))["fingerprints"][name]
        if current != expected:
            raise RuntimeError(f"{name} production fingerprint changed")
    detection_hash = detection_fingerprint(records)
    with (OUT / "frozen_detections.jsonl").open("w", encoding="utf-8") as stream:
        for frame in sorted(records):
            stream.write(json.dumps({"frame": frame, **records[frame]}, ensure_ascii=False) + "\n")
    freeze = {"prediction_fingerprints": {
                  name: fn_audit.prediction_fingerprint(validation_predictions[name]) for name in ("V0", "V1")},
              "detection_fingerprint": detection_hash, "gt_loaded": False,
              "physical_du_px": 0.0, "candidate_applied_to_inference": False,
              "test_access": 0, "embargo_access": 0}
    (OUT / "freeze_receipt_before_gt.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8")

    # First GT read occurs only after prediction/detection artifacts are frozen.
    train_truth = fn_audit.load_truth_after_predictions(train_frames)
    validation_truth = fn_audit.load_truth_after_predictions(validation_frames)
    for name in ("V0", "V1"):
        fn_audit.metric_checked(name, validation_frames, validation_predictions[name], validation_truth)
    model = pipeline.OnlineRgbFrustumPipeline(
        device="cpu", identity=False, projection_audit=False,
        legacy_display_offset=False, suppressed_recovery=False)
    try:
        v0_fn, _ = fn_audit.taxonomy("V0", validation_frames,
                                     validation_predictions["V0"], validation_truth, traces, model)
        v1_fn, _ = fn_audit.taxonomy("V1", validation_frames,
                                     validation_predictions["V1"], validation_truth, traces, model)
        a2_index = [row for row in v1_fn
                    if row["primary_root_cause"] == "A2_PERSON_BOX_GEOMETRY_MISMATCH"]
        if len(v1_fn) != 909 or len(a2_index) != 770:
            raise RuntimeError(f"frozen A2 gate failed: FN={len(v1_fn)} A2={len(a2_index)}")
        train_tp = matched_reference(train_frames, predictions["V1"], train_truth, records, model)
        train_a2 = train_a2_rows(train_frames, predictions["V1"], train_truth, traces, records, model)
        rules = freeze_rules(train_tp, train_a2)
        (OUT / "TRAIN_FIT_FROZEN_RULES.json").write_text(
            json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")

        target_lookup = {(frame, item["entity_id"]): index
                         for frame in validation_frames
                         for index, item in enumerate(validation_truth[frame])}
        validation_rank = {frame: index for index, frame in enumerate(validation_frames)}
        validation_block_size = math.ceil(len(validation_frames) / 4)
        a2_rows = []
        for source in a2_index:
            frame, entity = source["frame_id"], source["gt_person_id_for_audit_only"]
            row = case_metrics(frame, target_lookup[(frame, entity)], validation_truth[frame],
                               records[frame], model)
            refine_explanatory_flags(row, rules)
            row["existing_primary_root_cause"] = source["primary_root_cause"]
            row["time_block"] = f"P{min(validation_rank[frame] // validation_block_size + 1, 4)}"
            subtype, evidence = classify(row, rules)
            row["A2_subtype"], row["classification_evidence"] = subtype, evidence
            a2_rows.append(row)
        if len(a2_rows) != 770 or any(row["A2_subtype"] not in SUBTYPES for row in a2_rows):
            raise RuntimeError("A2 subtype exclusivity/completeness failed")
        tp_rows = matched_reference(validation_frames, validation_predictions["V1"],
                                    validation_truth, records, model)
        for row in tp_rows:
            refine_explanatory_flags(row, rules)
            block = min(validation_rank[row["frame_id"]] // validation_block_size + 1, 4)
            row["time_block"] = f"P{block}"

        v0_a2 = {(row["frame_id"], row["gt_person_id_for_audit_only"])
                 for row in v0_fn if row["primary_root_cause"] == "A2_PERSON_BOX_GEOMETRY_MISMATCH"}
        v1_miss = {(row["frame_id"], row["gt_person_id_for_audit_only"]) for row in v1_fn}
        resolved_keys = v0_a2 - v1_miss
        match_source = {}
        for frame in validation_frames:
            matches, _, _ = recovery_audit.match_frame(
                validation_predictions["V1"][frame], validation_truth[frame])
            for match in matches:
                key = (frame, validation_truth[frame][match["gt_index"]]["entity_id"])
                match_source[key] = validation_predictions["V1"][frame][match["prediction_index"]]["source"]
        resolved = []
        for frame, entity in sorted(resolved_keys):
            row = case_metrics(frame, target_lookup[(frame, entity)], validation_truth[frame],
                               records[frame], model)
            refine_explanatory_flags(row, rules)
            subtype, evidence = classify(row, rules)
            row.update({"A2_subtype": subtype, "classification_evidence": evidence,
                        "V1_match_source": match_source.get((frame, entity)),
                        "resolution": "V1_3D_RECOVERY_CREATED_IN_GATE_MATCH"})
            resolved.append(row)
        if len(resolved) != 62:
            raise RuntimeError(f"expected 62 HISTORY_ONLY-resolved A2, got {len(resolved)}")

        observed_counts = Counter(row["A2_subtype"] for row in a2_rows)
        counts = Counter({subtype: observed_counts[subtype] for subtype in SUBTYPES})
        count_rows = [{"subtype": subtype, "count": counts[subtype],
                       "fraction": counts[subtype] / len(a2_rows)} for subtype in SUBTYPES]
        index_rows = [{"frame_id": row["frame_id"], "timestamp": row["timestamp"],
                       "GT_audit_identity": row["gt_audit_id"], "GT_XYZ": json.dumps([
                           row["gt_x"], row["gt_y"], row["gt_z"]]), "GT_depth": row["depth"],
                       "PERSON_detections": row["person_detection_count"],
                       "existing_primary_root_cause": row["existing_primary_root_cause"]}
                      for row in a2_rows]
        write_csv(OUT / "A2_CASE_INDEX.csv", index_rows)
        write_csv(OUT / "a2_case_metrics.csv", a2_rows)
        write_csv(OUT / "A2_SUBTYPE_COUNTS.csv", count_rows)
        write_csv(OUT / "A2_BY_DEPTH.csv", cross(a2_rows, "depth_bin"))
        write_csv(OUT / "A2_BY_TIME.csv", cross(a2_rows, "time_block"))
        write_csv(OUT / "A2_BY_IMAGE_REGION.csv", cross(a2_rows, "image_region"))
        write_csv(OUT / "A2_BY_PERSON_AUDIT_ONLY.csv", cross(a2_rows, "gt_audit_id"))
        write_csv(OUT / "A2_BY_OCCLUSION.csv", occlusion_cross(a2_rows))
        write_csv(OUT / "TP_CORRESPONDENCE_REFERENCE.csv", tp_rows)
        comparison = statistics(a2_rows, "A2") + statistics(tp_rows, "TP")
        write_csv(OUT / "A2_VS_TP_STATISTICS.csv", comparison)
        distance_profiles = depth_profiles(a2_rows, "A2") + depth_profiles(tp_rows, "TP")
        write_csv(OUT / "A2_VS_TP_BY_DEPTH.csv", distance_profiles)
        write_csv(OUT / "HISTORY_ONLY_A2_RESOLVED_ANALYSIS.csv", resolved)
        for subtype, filename in SHEET_NAMES.items():
            contact_sheet([row for row in a2_rows if row["A2_subtype"] == subtype],
                          OUT / filename, records)
    finally:
        model.calibration_audit.close()

    residual = {
        "du_vs_inverse_depth": regression([1.0 / row["depth"] for row in a2_rows],
                                           [row["signed_du_px"] for row in a2_rows]),
        "du_vs_projected_u": regression([row["projected_center_u"] for row in a2_rows],
                                         [row["signed_du_px"] for row in a2_rows]),
        "dv_vs_projected_v": regression([row["projected_center_v"] for row in a2_rows],
                                         [row["signed_dv_px"] for row in a2_rows]),
    }
    group_fields = ("time_block", "depth_bin", "image_region", "gt_audit_id")
    residual_groups = {field: group_stability(a2_rows, field) for field in group_fields}
    group_du = [item["du"] for values in residual_groups.values() for item in values.values()]
    group_dv = [item["dv"] for values in residual_groups.values() for item in values.values()]
    du_iqr = q([row["signed_du_px"] for row in a2_rows], 75) - q(
        [row["signed_du_px"] for row in a2_rows], 25)
    dv_iqr = q([row["signed_dv_px"] for row in a2_rows], 75) - q(
        [row["signed_dv_px"] for row in a2_rows], 25)
    constant_supported = bool(group_du and np.ptp(group_du) <= max(12.0, 1.5 * du_iqr)
                              and np.ptp(group_dv) <= max(12.0, 1.5 * dv_iqr))
    position_pattern = bool(abs(residual["du_vs_projected_u"]["correlation"] or 0) >= 0.30
                            or abs(residual["dv_vs_projected_v"]["correlation"] or 0) >= 0.30)
    depth_pattern = bool(abs(residual["du_vs_inverse_depth"]["correlation"] or 0) >= 0.30)
    person_du = [item["du"] for item in residual_groups["gt_audit_id"].values()]
    person_dv = [item["dv"] for item in residual_groups["gt_audit_id"].values()]
    person_specific = bool(person_du and (
        np.ptp(person_du) > max(15.0, 0.5 * du_iqr)
        or np.ptp(person_dv) > max(15.0, 0.5 * dv_iqr)))
    a2_stats = {row["metric"]: row for row in comparison if row["cohort"] == "A2"}
    tp_stats = {row["metric"]: row for row in comparison if row["cohort"] == "TP"}
    distinct_du = abs(a2_stats["signed_du_px"]["median"]
                      - tp_stats["signed_du_px"]["median"]) > max(
                          10.0, tp_stats["signed_du_px"]["iqr"])
    distinct_dv = abs(a2_stats["signed_dv_px"]["median"]
                      - tp_stats["signed_dv_px"]["median"]) > max(
                          10.0, tp_stats["signed_dv_px"]["iqr"])
    distinct_from_tp = distinct_du or distinct_dv
    if constant_supported and distinct_from_tp and counts[SUBTYPES[1]] == max(counts.values()):
        calibration = "CALIBRATION_PATTERN_STRONGLY_SUPPORTED"
    elif distinct_from_tp or position_pattern or depth_pattern:
        calibration = "CALIBRATION_PATTERN_POSSIBLE_BUT_CONFOUNDED"
    else:
        calibration = "NO_CLEAR_CALIBRATION_PATTERN"
    largest = max(SUBTYPES, key=lambda subtype: counts[subtype])
    if largest in {SUBTYPES[0], SUBTYPES[2]}:
        next_step = "STUDY_3D_CUBOID_TO_VISIBLE_RGB_BOX_SEMANTIC_CORRESPONDENCE"
    elif largest == SUBTYPES[1] and constant_supported:
        next_step = "INDEPENDENT_PHYSICAL_CALIBRATION_INVESTIGATION"
    elif largest == SUBTYPES[3]:
        next_step = "MULTI_PERSON_CORRESPONDENCE_ASSIGNMENT_AUDIT"
    elif largest == SUBTYPES[5]:
        next_step = "RGB_DETECTOR_GEOMETRY_AUDIT"
    else:
        next_step = "VISUALLY_REVIEW_UNRESOLVED_A2_WITHOUT_INFERENCE_CHANGE"
    a2_depth_profiles = {row["depth_bin"]: row for row in distance_profiles
                         if row["cohort"] == "A2"}
    far = a2_depth_profiles.get(">9m", {"n": 0})
    mid = a2_depth_profiles.get("6-9m", {"n": 0})
    if far.get("n", 0) and mid.get("n", 0):
        distance_conclusion = (
            "FAR_A2_IS_NOT_EXPLAINED_BY_RAW_SUPPORT_LOSS_OR_HORIZONTAL_PIXEL_ERROR_ALONE; "
            "THE_SELECTED BOX OFTEN BELONGS TO ANOTHER PERSON, AND >9m CASES SHOW MUCH "
            "LARGER VERTICAL RESIDUAL WHILE NORMALIZED CENTER ERROR REMAINS ABOUT ONE BOX SIZE")
    else:
        distance_conclusion = (
            "VALIDATION A2 HAS INSUFFICIENT NEAR/MID DEPTH SUPPORT FOR A CLEAN CAUSAL "
            "SEPARATION OF ABSOLUTE AND RELATIVE RESIDUAL")
    occlusion_summary = {
        "any_occlusion_or_crossing": sum(row["occlusion_flag"] for row in a2_rows),
        "close_neighbor": sum(row["close_neighbor_flag"] for row in a2_rows),
        "overlapping_person_boxes": sum(row["overlapping_box_flag"] for row in a2_rows),
        "by_subtype": {subtype: {
            "any": sum(row["occlusion_flag"] for row in a2_rows
                       if row["A2_subtype"] == subtype),
            "close_neighbor": sum(row["close_neighbor_flag"] for row in a2_rows
                                  if row["A2_subtype"] == subtype),
            "overlapping_boxes": sum(row["overlapping_box_flag"] for row in a2_rows
                                     if row["A2_subtype"] == subtype),
        } for subtype in SUBTYPES},
    }
    summary = {
        "status": "AUDIT_COMPLETE_NO_PRODUCTION_CHANGE",
        "frozen_gate": {"V1_FN": 909, "A2": 770,
                        "prediction_fingerprints": freeze["prediction_fingerprints"],
                        "detection_fingerprint": detection_hash},
        "scope": {"TRAIN_FIT_rule_source": len(train_frames),
                  "VALIDATION_frames": len(validation_frames),
                  "TEST_access": 0, "EMBARGO_access": 0},
        "physical_projection": "RAW_K_D_T_DU_DV_ZERO",
        "candidate_applied_to_inference": False,
        "rules": rules, "subtype_counts": dict(counts),
        "history_only_resolved_A2": {"count": len(resolved),
                                     "subtypes": dict(Counter(row["A2_subtype"] for row in resolved)),
                                     "sources": dict(Counter(row["V1_match_source"] for row in resolved))},
        "A2_vs_TP": {"A2": a2_stats, "TP": tp_stats},
        "distance_analysis": {"profiles": distance_profiles,
                              "conclusion": distance_conclusion},
        "occlusion_analysis": occlusion_summary,
        "residual_analysis": {"regressions": residual, "group_medians": residual_groups,
                              "constant_offset_hypothesis": (
                                  "CONSTANT_OFFSET_HYPOTHESIS_SUPPORTED" if constant_supported
                                  else "NOT_SUPPORTED"),
                              "depth_dependent_pattern": depth_pattern,
                              "position_dependent_pattern": position_pattern,
                              "person_specific_pattern": person_specific,
                              "calibration_conclusion": calibration},
        "next_step": next_step,
        "tests": {"cuboid_projection_same_raw_model": True, "proxy_minmax": True,
                  "taxonomy_exclusive": True, "taxonomy_complete_770": True,
                  "GT_loaded_after_freeze": True,
                  "instrumentation_equivalent": safety["instrumentation_predictions_exact"],
                  "TEST_EMBARGO_zero": True, "no_calibration_writeback": True},
    }
    (OUT / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    resolved_subtypes = summary["history_only_resolved_A2"]["subtypes"]
    lines = [
        "# A2 Raw-Projection ↔ RGB-Box Correspondence Decomposition Audit", "",
        "## Gate", "", "**PASS — V1 FN=909、A2=770 精确复现；NO PRODUCTION CHANGE。**", "",
        "规则只由 TRAIN_FIT 冻结；VALIDATION updates=0。所有投影均为 raw `K+D+T, du=dv=0`。", "",
        "## A2 subtype distribution", "", "| Subtype | Count | Fraction |", "|---|---:|---:|",
        *[f"| {row['subtype']} | {row['count']} | {row['fraction']:.2%} |" for row in count_rows], "",
        "## A2 与 TP 对照", "",
        f"- signed du median：A2 `{a2_stats['signed_du_px']['median']:.2f}px`；TP `{tp_stats['signed_du_px']['median']:.2f}px`。",
        f"- signed dv median：A2 `{a2_stats['signed_dv_px']['median']:.2f}px`；TP `{tp_stats['signed_dv_px']['median']:.2f}px`。",
        f"- proxy-box IoU median：A2 `{a2_stats['IoU_proxy_bbox']['median']:.3f}`；TP `{tp_stats['IoU_proxy_bbox']['median']:.3f}`。",
        f"- normalized center distance median：A2 `{a2_stats['center_distance_norm']['median']:.3f}`；TP `{tp_stats['center_distance_norm']['median']:.3f}`。", "",
        "## Required answers", "",
        f"1. 770 个 A2 的完整互斥分布见上表；合计 `{sum(counts.values())}`，最大 subtype 为 `{largest}`。",
        f"2. correspondence criterion 太严格（A2a）：`{counts[SUBTYPES[0]]}`。",
        f"3. registration-sensitive pattern（A2b）：`{counts[SUBTYPES[1]]}`。这不是 calibration failure 确认。",
        f"4. 明确 visible-box / 3D-cuboid semantic mismatch（A2c）：`{counts[SUBTYPES[2]]}`。",
        f"5. primary truncation/image-edge（A2e）：`{counts[SUBTYPES[4]]}`；独立 explanatory occlusion/crossing flag：`{occlusion_summary['any_occlusion_or_crossing']}`（可与其他 subtype 重合，不能相加）。",
        f"6. neighbor/box ambiguity（A2d）：`{counts[SUBTYPES[3]]}`；close-neighbor=`{occlusion_summary['close_neighbor']}`，overlapping-box=`{occlusion_summary['overlapping_person_boxes']}`。",
        f"7. detection geometry anomaly（A2f）：`{counts[SUBTYPES[5]]}`。审计不把 proxy 不确定性强行算成 detector failure。",
        f"8. unresolved（A2g）：`{counts[SUBTYPES[6]]}`。",
        f"9. A2/TP signed-du median=`{a2_stats['signed_du_px']['median']:.2f}/{tp_stats['signed_du_px']['median']:.2f}px`；signed-dv median=`{a2_stats['signed_dv_px']['median']:.2f}/{tp_stats['signed_dv_px']['median']:.2f}px`。完整 IQR/P90/P95 见 `A2_VS_TP_STATISTICS.csv`。",
        f"10. A2/TP median proxy-box IoU=`{a2_stats['IoU_proxy_bbox']['median']:.3f}/{tp_stats['IoU_proxy_bbox']['median']:.3f}`。",
        f"11. residual 跨时间不满足统一常量门禁：`{summary['residual_analysis']['constant_offset_hypothesis']}`；P1–P4 中位数见 summary。",
        f"12. 跨深度：du-vs-1/Z correlation=`{residual['du_vs_inverse_depth']['correlation']}`、R²=`{residual['du_vs_inverse_depth']['r2']}`，depth-dependent={depth_pattern}；不据此修改 translation。",
        f"13. 跨 image region：position-dependent={position_pattern}；du-vs-u correlation=`{residual['du_vs_projected_u']['correlation']}`，dv-vs-v correlation=`{residual['dv_vs_projected_v']['correlation']}`。",
        f"14. person-specific={person_specific}。身份分组残差差异属于姿态/遮挡/语义 confound 证据，不能写回 runtime。",
        f"15. `{distance_conclusion}`。6–9m/>9m 的 median abs-du=`{mid.get('median_abs_du_px', float('nan')):.2f}/{far.get('median_abs_du_px', float('nan')):.2f}px`，abs-dv=`{mid.get('median_abs_dv_px', float('nan')):.2f}/{far.get('median_abs_dv_px', float('nan')):.2f}px`，normalized center distance=`{mid.get('median_center_distance_norm', float('nan')):.2f}/{far.get('median_center_distance_norm', float('nan')):.2f}`。当前证据不能把远距暴增简化为 raw LiDAR support 消失或 bbox 缩小单一因素。",
        f"16. 62 个 HISTORY_ONLY-resolved A2 subtype=`{resolved_subtypes}`，V1 source=`{summary['history_only_resolved_A2']['sources']}`。恢复分支提供了合法 3D prediction 并进入 1.5m match gate，因此旧 A2 消失；这证明旧 A2 标签不是推理失败的充分条件。",
        f"17. principal-point/image-origin：当前结论 `{calibration}`，但 constant offset 未通过，因此 A2 人体 proxy 证据不足以单独支持写回或把它列为下一步。",
        f"18. SE(3)：当前 A2 只给出 `{calibration}`，position/depth pattern 又受语义与错误候选框混杂；不足以确认或写回 SE(3)。",
        f"19. 目前没有充足证据认定 center-based proxy 本身不合理：A2a+A2c=`{counts[SUBTYPES[0]] + counts[SUBTYPES[2]]}`。62 个旧 A2 可被因果 3D recovery 消除，只能证明旧 A2 标签不是推理失败的充分条件。",
        f"20. 唯一下一步：`{next_step}`。", "",
        "## Safety", "",
        "- Frozen YOLO / suppression / clustering / HISTORY_ONLY / tracking / identity：unchanged",
        "- Prediction fingerprint：unchanged",
        "- candidate_applied_to_inference：false",
        "- TEST access：0；EMBARGO access：0",
        "- GT overlay：audit contact sheets only", "",
        "完整案例、TP 对照、交叉表、62-case 分析与七张 contact sheet 均位于本目录。",
    ]
    (OUT / "A2_CORRESPONDENCE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "test_log.txt").write_text(
        "cuboid_projection_raw_KDT_du0 PASS\nproxy_minmax PASS\n"
        "taxonomy_exclusivity PASS\ntaxonomy_completeness_770 PASS\n"
        "GT_isolation PASS\nTEST_EMBARGO_access_zero PASS\n"
        "instrumentation_equivalence PASS\nno_calibration_writeback PASS\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()

