"""Benchmark symmetric soft point ownership on the raw du=dv=0 projection.

The experiment is report-only.  It changes neither calibration nor runtime
configuration and never uses +48, GT, future frames or legacy output as truth.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from online_v4 import pipeline, support  # noqa: E402

OUT = ROOT / "outputs" / "soft_ownership_audit"
MARGINS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)
RGB_VERTICAL_MARGIN = 0.03
FRAMES = 65


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def soft_owners(pixels: np.ndarray, boxes: list[np.ndarray], horizontal_margin: float) -> np.ndarray:
    """Assign each point once using a symmetric normalized uncertainty band."""
    if not boxes or not len(pixels):
        return np.full(len(pixels), -1, np.int32)
    points = np.asarray(pixels, float)
    values = np.asarray(boxes, float)
    wh = np.maximum(values[:, 2:4] - values[:, 0:2], 1.0)
    centers = 0.5 * (values[:, 0:2] + values[:, 2:4])
    outside_x = np.maximum(np.maximum(values[None, :, 0] - points[:, None, 0], 0.0),
                           points[:, None, 0] - values[None, :, 2]) / wh[None, :, 0]
    outside_y = np.maximum(np.maximum(values[None, :, 1] - points[:, None, 1], 0.0),
                           points[:, None, 1] - values[None, :, 3]) / wh[None, :, 1]
    eligible = (outside_x <= horizontal_margin) & (outside_y <= RGB_VERTICAL_MARGIN)
    center = np.linalg.norm((points[:, None, :] - centers[None, :, :]) / wh[None, :, :], axis=2)
    cost = center + 2.0 * np.hypot(outside_x, outside_y)
    cost[~eligible] = np.inf
    owner = np.argmin(cost, axis=1).astype(np.int32)
    owner[~np.any(eligible, axis=1)] = -1
    return owner


def component_choice(components: list[np.ndarray], box: np.ndarray, entity_class: str,
                     transform: np.ndarray, K: np.ndarray, D: np.ndarray) -> dict | None:
    ranked = []
    for component in components:
        score, details = support.component_score(component, box, transform, K, D, du_px=0.0)
        if np.isfinite(score): ranked.append((float(score), component, details))
    if not ranked:
        return None
    ranked.sort(key=lambda value: (value[0], -len(value[1])))
    score, component, details = ranked[0]
    center = support.cluster_center(component)
    pixel, valid, _ = support.project_points(center[None], transform, K, D, du_px=0.0, dv_px=0.0)
    inside = bool(valid[0] and box[0] <= pixel[0, 0] <= box[2] and box[1] <= pixel[0, 1] <= box[3])
    span = np.ptp(component, axis=0)
    horizontal, vertical = float(max(span[0], span[1])), float(span[2])
    if entity_class == "PERSON":
        plausible = 0.10 <= horizontal <= 1.10 and 0.10 <= vertical <= 2.20
    else:
        plausible = 0.10 <= horizontal <= 1.35 and 0.08 <= vertical <= 1.45
    second_gap = None if len(ranked) < 2 else float(ranked[1][0] - score)
    return {"score": score, "points": len(component), "center_inside_original": inside,
            "plausible_size": plausible, "horizontal_span_m": horizontal, "vertical_span_m": vertical,
            "candidate_components": len(ranked), "second_score_gap": second_gap,
            "quality_guard": bool(inside and plausible and score <= 1.25), **details}


def aggregate(rows: list[dict], margin: float) -> dict:
    selected = [row for row in rows if row["margin"] == margin]
    measured = [row for row in selected if row["measurement"]]
    scores = np.asarray([row["component_score"] for row in measured], float)
    return {
        "margin": margin, "detections": len(selected),
        "component_available": sum(row["component_available"] for row in selected),
        "measurements": len(measured), "quality_guard_measurements": sum(row["quality_guard"] for row in measured),
        "component_coverage": len(measured) / max(len(selected), 1),
        "quality_guard_coverage": sum(row["quality_guard"] for row in measured) / max(len(selected), 1),
        "center_inside_fraction": sum(row["center_inside_original"] for row in measured) / max(len(measured), 1),
        "plausible_size_fraction": sum(row["plausible_size"] for row in measured) / max(len(measured), 1),
        "ambiguous_fraction": sum(row["second_score_gap"] is not None and row["second_score_gap"] < 0.05 for row in measured) / max(len(measured), 1),
        "component_score_median": None if not len(scores) else float(np.median(scores)),
        "component_score_p95": None if not len(scores) else float(np.percentile(scores, 95)),
    }


def aggregate_suppression(rows: list[dict], policy: str) -> dict:
    selected = [row for row in rows if row["suppression_policy"] == policy]
    measured = [row for row in selected if row["measurement"]]
    return {"suppression_policy": policy, "detections": len(selected), "measurements": len(measured),
            "coverage": len(measured) / max(len(selected), 1),
            "quality_guard_coverage": sum(row["quality_guard"] for row in measured) / max(len(selected), 1),
            "center_inside_fraction": sum(row["center_inside_original"] for row in measured) / max(len(measured), 1),
            "plausible_size_fraction": sum(row["plausible_size"] for row in measured) / max(len(measured), 1)}


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    model = pipeline.OnlineRgbFrustumPipeline(device="0", identity=False, projection_audit=False,
                                              legacy_display_offset=False)
    occupancy_tree = cKDTree(model.occupancy)
    rows, suppression_rows, frame_count = [], [], 0
    try:
        for frame_index, frame in enumerate(pipeline.synchronized_bag_frames(pipeline.DEFAULT_BAG, 35.0)):
            if frame_index >= FRAMES: break
            frame_count += 1
            detections = model._detections(frame.image)
            transform = model.annotated_from_rslidar
            annotated = frame.points_rslidar @ transform[:3, :3].T + transform[:3, 3]
            pixels, valid, depth = model.project_physical(annotated)
            ground_height = annotated @ model.ground_normal + model.ground_d
            in_view = (valid & (depth > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < pipeline.IMAGE_SIZE[0]) &
                       (pixels[:, 1] >= 0) & (pixels[:, 1] < pipeline.IMAGE_SIZE[1]))
            candidate = np.flatnonzero(in_view & (ground_height >= 0.03) & (ground_height <= 2.15))
            keep, _ = model._static_keep(annotated[candidate], ground_height[candidate])
            usable = candidate[keep]
            boxes = [item["bbox"] for item in detections]
            candidate_points = annotated[candidate]
            keys = np.floor(candidate_points / model.voxel_size).astype(np.int32)
            temporal_keep = np.fromiter((tuple(map(int, key)) not in model.static for key in keys), bool, len(keys))
            distance = occupancy_tree.query(candidate_points[:, :2], workers=-1)[0]
            point_height = candidate_points[:, 2] - ground_height[candidate]
            occupancy_keep = ~((distance <= 0.07) & ((point_height <= 0.28) | (point_height >= 1.9)))
            exact_candidate_owners = soft_owners(pixels[candidate], boxes, 0.0)
            suppression_policies = {
                "TEMPORAL_PLUS_OCCUPANCY": temporal_keep & occupancy_keep,
                "TEMPORAL_ONLY": temporal_keep,
                "OCCUPANCY_ONLY": occupancy_keep,
                "NONE": np.ones(len(candidate), bool),
            }
            for policy, policy_keep in suppression_policies.items():
                policy_owners = exact_candidate_owners.copy()
                policy_owners[~policy_keep] = -1
                for index, detection in enumerate(detections):
                    assigned = policy_owners == index
                    lower, upper = (0.08, 2.15) if detection["class"] == "PERSON" else (0.03, 1.35)
                    policy_owners[assigned & ((ground_height[candidate] < lower) | (ground_height[candidate] > upper))] = -1
                if model.geometry_gpu:
                    policy_grouped = model.geometry_gpu.components(candidate_points, policy_owners, len(boxes))
                else:
                    policy_grouped = [support.adaptive_components(candidate_points[policy_owners == index]) for index in range(len(boxes))]
                for index, (detection, components) in enumerate(zip(detections, policy_grouped, strict=True)):
                    choice = component_choice(components, detection["bbox"], detection["class"],
                                              model.transform, model.K, model.D)
                    row = {"frame": frame_index, "suppression_policy": policy, "detection": index,
                           "class": detection["class"], "measurement": choice is not None,
                           "quality_guard": False, "center_inside_original": False, "plausible_size": False}
                    if choice is not None: row.update(choice)
                    suppression_rows.append(row)
            for margin in MARGINS:
                candidate_owners = soft_owners(pixels[candidate], boxes, margin)
                owners = soft_owners(pixels[usable], boxes, margin)
                filtered = owners.copy()
                for index, detection in enumerate(detections):
                    assigned = filtered == index
                    lower, upper = (0.08, 2.15) if detection["class"] == "PERSON" else (0.03, 1.35)
                    filtered[assigned & ((ground_height[usable] < lower) | (ground_height[usable] > upper))] = -1
                if model.geometry_gpu:
                    grouped = model.geometry_gpu.components(annotated[usable], filtered, len(boxes))
                else:
                    grouped = []
                    for index in range(len(boxes)):
                        grouped.append(support.adaptive_components(annotated[usable[filtered == index]]))
                for index, (detection, components) in enumerate(zip(detections, grouped, strict=True)):
                    choice = component_choice(components, detection["bbox"], detection["class"],
                                              model.transform, model.K, model.D)
                    row = {"frame": frame_index, "margin": margin, "detection": index,
                           "class": detection["class"], "confidence": detection["confidence"],
                           "bbox_x1": float(detection["bbox"][0]), "bbox_y1": float(detection["bbox"][1]),
                           "bbox_x2": float(detection["bbox"][2]), "bbox_y2": float(detection["bbox"][3]),
                           "bbox_width": float(detection["bbox"][2] - detection["bbox"][0]),
                           "candidate_owned_points_before_static": int(np.sum(candidate_owners == index)),
                           "owned_points_after_static_and_height": int(np.sum(filtered == index)),
                           "component_available": bool(components), "measurement": choice is not None,
                           "quality_guard": False, "center_inside_original": False,
                           "plausible_size": False, "second_score_gap": None, "component_score": None}
                    if choice is not None: row.update({"component_score": choice.pop("score"), **choice})
                    rows.append(row)
    finally:
        model.calibration_audit.close()
    if frame_count != FRAMES:
        raise RuntimeError(f"Expected {FRAMES} frames, received {frame_count}")
    metrics = [aggregate(rows, margin) for margin in MARGINS]
    suppression_metrics = [aggregate_suppression(suppression_rows, policy)
                           for policy in ("TEMPORAL_PLUS_OCCUPANCY", "TEMPORAL_ONLY", "OCCUPANCY_ONLY", "NONE")]
    baseline = metrics[0]
    baseline_missing = [row for row in rows if row["margin"] == 0.0 and not row["measurement"]]
    missing_cause = {
        "missing_measurements": len(baseline_missing),
        "ownership_has_fewer_than_3_points_before_static": sum(row["candidate_owned_points_before_static"] < 3 for row in baseline_missing),
        "fewer_than_3_points_after_static_and_height": sum(row["owned_points_after_static_and_height"] < 3 for row in baseline_missing),
        "median_owned_points_before_static": float(np.median([row["candidate_owned_points_before_static"] for row in baseline_missing])),
        "median_owned_points_after_static_and_height": float(np.median([row["owned_points_after_static_and_height"] for row in baseline_missing])),
    }
    for value in metrics:
        value["guard_center_vs_baseline"] = value["center_inside_fraction"] >= baseline["center_inside_fraction"] - 0.02
        value["guard_plausible_vs_baseline"] = value["plausible_size_fraction"] >= baseline["plausible_size_fraction"] - 0.02
        value["guard_ambiguity_vs_baseline"] = value["ambiguous_fraction"] <= baseline["ambiguous_fraction"] + 0.03
        value["all_quality_guards"] = bool(value["guard_center_vs_baseline"] and value["guard_plausible_vs_baseline"] and value["guard_ambiguity_vs_baseline"])
    eligible = [value for value in metrics if value["all_quality_guards"]]
    recommended = max(eligible, key=lambda value: (value["quality_guard_coverage"], -value["margin"])) if eligible else baseline
    result = {
        "status": "REPORT_ONLY", "scope": f"SCENE01_TRAINFIT_PREFIX_{FRAMES}_LIDAR",
        "physical_projection": "RAW_K_D_T_DU_DV_ZERO_NOT_PROVEN_FINAL",
        "symmetric_horizontal_margins_as_box_width_fraction": list(MARGINS),
        "vertical_margin_as_box_height_fraction": RGB_VERTICAL_MARGIN,
        "metrics": metrics, "recommended_report_only_margin": recommended["margin"],
        "suppression_ablation": suppression_metrics,
        "baseline_missing_cause": missing_cause,
        "runtime_modified": False, "legacy_offset_used": False, "ground_truth_used": False,
        "selection_rule": "maximize quality-guard coverage subject to center, size and ambiguity guardrails",
    }
    write_csv(OUT / "per_detection_variant_audit.csv", rows)
    write_csv(OUT / "variant_metrics.csv", metrics)
    write_csv(OUT / "suppression_ablation.csv", suppression_metrics)
    write_json(OUT / "soft_ownership_audit.json", result)
    lines = ["# Scene01 du=0 Soft Ownership Audit", "", "## Result", "",
             f"Report-only recommended symmetric horizontal margin: **{recommended['margin']:.0%} of bbox width**.",
             "No calibration, inference configuration, XYZ or Tracking output was modified.", "",
             "| Margin | Component coverage | Guarded coverage | Center inside | Plausible size | Ambiguous | Guards |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | :---: |"]
    for value in metrics:
        lines.append(f"| {value['margin']:.0%} | {value['component_coverage']:.2%} | {value['quality_guard_coverage']:.2%} | {value['center_inside_fraction']:.2%} | {value['plausible_size_fraction']:.2%} | {value['ambiguous_fraction']:.2%} | {'PASS' if value['all_quality_guards'] else 'FAIL'} |")
    lines += ["", "## Interpretation", "",
              "The band is symmetric and scale-normalized; it does not estimate or apply a pixel translation. Every LiDAR point still has at most one owner. Coverage alone is not the selection target: projected-center containment, 3D size plausibility and component-score ambiguity are explicit guardrails.", "",
              f"The {missing_cause['missing_measurements']} baseline misses had a median of {missing_cause['median_owned_points_before_static']:.0f} owned points before frozen suppression, but only {missing_cause['median_owned_points_after_static_and_height']:.0f} afterward. {missing_cause['fewer_than_3_points_after_static_and_height']}/{missing_cause['missing_measurements']} fell below the three-point component threshold after suppression. This localizes the prefix loss to frozen static/background handling rather than a lack of points inside the raw du=0 box.", "",
              "## Static-suppression ablation", "",
              "| Policy | Coverage | Guarded coverage | Center inside | Plausible size |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for value in suppression_metrics:
        lines.append(f"| {value['suppression_policy']} | {value['coverage']:.2%} | {value['quality_guard_coverage']:.2%} | {value['center_inside_fraction']:.2%} | {value['plausible_size_fraction']:.2%} |")
    lines += ["", "This ablation is diagnostic only. Recovering a component by disabling a frozen background prior can also recover static walls or stationary-object false positives; it is not authorization to weaken suppression in production.", "",
              "This prefix audit is not a localization-accuracy result because no independent 3D GT is used. The recommended value remains report-only until a longer chronological validation confirms the same trade-off."]
    (OUT / "SOFT_OWNERSHIP_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

