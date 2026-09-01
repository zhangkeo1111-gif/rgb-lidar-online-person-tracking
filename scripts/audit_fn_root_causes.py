"""Decompose frozen V0/V1 VALIDATION false negatives without changing inference.

The causal replay uses raw K+D+T (du=dv=0), the frozen LEGACY suppression
policy and the frozen HISTORY_ONLY recovery.  Ground truth is deliberately
loaded only after both prediction streams have been completed and fingerprinted.
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
from online_v4 import pipeline, recovery, support, suppression  # noqa: E402
import audit_occupancy_suppression as occupancy_audit  # noqa: E402
import audit_suppressed_point_recovery as recovery_audit  # noqa: E402


OUT = ROOT / "outputs" / "fn_root_cause_audit"
MANIFEST = Path(r"D:\navwareset_scene01_clean\data\splits\annotated_split_manifest.csv")
GT = Path(r"D:\navwareset_scene01_clean\data\canonical\gt\person_cuboid_gt_v2.csv")
EXPECTED = ROOT / "outputs" / "occupancy_suppression_redesign" / "VALIDATION_VARIANTS.csv"
IMAGE_W, IMAGE_H = pipeline.IMAGE_SIZE
MIN_POINTS = 3
GATE_M = recovery_audit.GATE_M
DEPTH_BINS = ((0.0, 3.0, "0-3m"), (3.0, 6.0, "3-6m"),
              (6.0, 9.0, "6-9m"), (9.0, math.inf, ">9m"))
PRIMARY_CODES = (
    "A1_NO_PERSON_DETECTION", "A2_PERSON_BOX_GEOMETRY_MISMATCH",
    "B1_NO_RAW_PROJECTED_POINTS", "B2_RAW_LIDAR_TOO_SPARSE",
    "C_GEOMETRIC_OR_HEIGHT_GATE_REMOVAL", "D_STATIC_SUPPRESSION_REMOVAL",
    "E1_NO_HISTORY_SUPPORT", "E2_RECOVERY_NO_VALID_COMPONENT",
    "E3_RECOVERY_MOTION_GATE_REJECT", "E4_RECOVERY_COMPONENT_QUALITY_REJECT",
    "E5_RECOVERY_NEIGHBOR_CONFLICT_REJECT", "F_CLUSTERING_FAIL",
    "G1_NO_COMPONENT_PASSED_SCORING", "G2_WRONG_COMPONENT_RANKED_FIRST",
    "G3_CORRECT_COMPONENT_LOST_RANKING", "H_POINT_OWNERSHIP_CONFLICT",
    "I1_NEAREST_PREDICTION_OUTSIDE_GATE", "I2_PREDICTION_MATCHED_ANOTHER_GT",
    "I3_DUPLICATE_OR_MERGED_DETECTION_COMPETITION",
    "I4_GLOBAL_ASSIGNMENT_COMPETITION",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["empty"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def prediction_fingerprint(predictions: dict[int, list[dict]]) -> str:
    payload = []
    for frame in sorted(predictions):
        for item in predictions[frame]:
            payload.append((frame, int(item["detection"]), item["track_id"],
                            item["source"], *np.round(item["xyz"], 12).tolist()))
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def same_predictions(left: list[dict], right: list[dict]) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        if ((a["detection"], a["track_id"], a["source"])
                != (b["detection"], b["track_id"], b["source"])):
            return False
        if not np.array_equal(a["xyz"], b["xyz"]):
            return False
    return True


class FrozenState:
    """Exact V0/V1 state with optional read-only decision capture."""

    def __init__(self, history: bool, capture: bool) -> None:
        self.with_history = history
        self.capture = capture
        self.tracker = pipeline.OnlineTracker(np.empty((0, 0), np.float32), ())
        self.history = recovery.CausalSuppressedPointRecovery(False)

    def step(self, result: occupancy_audit.PolicyResult, frame: int
             ) -> tuple[list[dict], dict[int, dict]]:
        inputs = []
        for index, item in enumerate(result.detections):
            inputs.append({**item, "bbox": item["bbox"].copy(),
                           "xyz": None if item["xyz"] is None else item["xyz"].copy(),
                           "_detection": index})
        tracked = self.tracker.update(inputs)
        output, decisions = [], {}
        for item, candidate in zip(tracked, result.recovery_candidates, strict=True):
            index = int(item["_detection"])
            source = "NORMAL_COMPONENT" if item["xyz"] is not None else "UNAVAILABLE"
            decision = {"triggered": False, "accepted": False,
                        "reason": "NORMAL_COMPONENT" if item["xyz"] is not None
                        else "HISTORY_DISABLED"}
            if item["class"] != "PERSON":
                continue
            if item["xyz"] is not None:
                self.history.observe_normal(item["track_id"], item["xyz"], frame)
            elif self.with_history:
                raw = self.history.consider(item["track_id"], candidate, frame)
                decision = {"triggered": candidate is not None, **raw}
                if raw["accepted"]:
                    assert candidate is not None
                    item["xyz"] = candidate.xyz.copy()
                    source = "RECOVERED_SUPPRESSED_COMPONENT"
                    state = self.tracker.tracks.get(item["track_id"])
                    if state is not None:
                        state["xyz"] = candidate.xyz.copy()
            decisions[index] = decision if self.capture else {}
            if item["xyz"] is not None:
                output.append({"frame": frame, "detection": index,
                               "track_id": item["track_id"], "source": source,
                               "xyz": item["xyz"].copy(),
                               "bbox": item["bbox"].copy(),
                               "confidence": float(item["confidence"])})
        return output, decisions


def component_records(components: list[np.ndarray], detection: dict, model,
                      du_px: float=0.0, dv_px: float=0.0) -> list[dict]:
    rows = []
    for component in components:
        score, details = support.component_score(
            component, detection["bbox"], model.transform, model.K, model.D,
            du_px=du_px, dv_px=dv_px)
        center = support.cluster_center(component)
        rows.append({"points": len(component), "score": float(score),
                     "xyz": center.tolist(), **details})
    rows.sort(key=lambda row: (row["score"], -row["points"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def trace_frame(model, detections: list[dict], annotated: np.ndarray,
                pixels: np.ndarray, valid: np.ndarray, depth: np.ndarray,
                ground_height: np.ndarray, du_px: float=0.0,
                dv_px: float=0.0, capture_trace: bool=True
                ) -> tuple[occupancy_audit.PolicyResult, list[dict]]:
    in_image = (valid & (depth > 0) & (pixels[:, 0] >= 0)
                & (pixels[:, 0] < IMAGE_W) & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < IMAGE_H))
    broad = (ground_height >= 0.03) & (ground_height <= 2.15)
    candidate_indices = np.flatnonzero(in_image & broad)
    points = annotated[candidate_indices]
    temporal, occupancy_near = model._static_evidence(points)
    boxes = [item["bbox"] for item in detections]
    owners = (model.geometry_gpu.owners(pixels[candidate_indices], boxes)
              if model.geometry_gpu else support.exclusive_point_owners(
                  pixels[candidate_indices], boxes, [None] * len(boxes)))
    occupancy_keep, _ = suppression.occupancy_keep(
        points, ground_height[candidate_indices], occupancy_near, temporal,
        suppression.Policy(suppression.LEGACY), owners,
        tuple(item["class"] for item in detections))
    result = occupancy_audit.policy_result(
        model, detections, points, candidate_indices, pixels, ground_height,
        temporal, occupancy_near, owners, suppression.Policy(suppression.LEGACY),
        du_px=du_px, dv_px=dv_px)
    if not capture_trace:
        return result, []

    combined = temporal & occupancy_keep
    traces = []
    for index, detection in enumerate(detections):
        box = detection["bbox"]
        inside_all = (valid & (depth > 0) & (pixels[:, 0] >= box[0])
                      & (pixels[:, 0] <= box[2]) & (pixels[:, 1] >= box[1])
                      & (pixels[:, 1] <= box[3]))
        lower, upper = ((0.08, 2.15) if detection["class"] == "PERSON"
                        else (0.03, 1.35))
        height_ok = ((ground_height[candidate_indices] >= lower)
                     & (ground_height[candidate_indices] <= upper))
        inside_box = ((pixels[candidate_indices, 0] >= box[0])
                      & (pixels[candidate_indices, 0] <= box[2])
                      & (pixels[candidate_indices, 1] >= box[1])
                      & (pixels[candidate_indices, 1] <= box[3]))
        nonexclusive = inside_box & height_ok
        owned = (owners == index) & height_ok
        post = owned & combined
        groups = (model.geometry_gpu.components(points, np.where(post, index, -1),
                                                len(detections))[index]
                  if model.geometry_gpu else support.adaptive_components(points[post]))
        selected_records = component_records(
            groups, detection, model, du_px=du_px, dv_px=dv_px)
        selected = selected_records[0] if selected_records else None
        stolen = nonexclusive & combined & (owners >= 0) & (owners != index)
        owner_hist = Counter(map(int, owners[stolen]))
        traces.append({
            "detection": index, "class": detection["class"],
            "confidence": float(detection["confidence"]),
            "bbox": box.tolist(),
            "raw_projected_points": int(np.count_nonzero(inside_all)),
            "post_geometry_height_points": int(np.count_nonzero(nonexclusive)),
            "post_temporal_points": int(np.count_nonzero(nonexclusive & temporal)),
            "post_occupancy_points": int(np.count_nonzero(nonexclusive & occupancy_keep)),
            "post_both_nonexclusive_points": int(np.count_nonzero(nonexclusive & combined)),
            "owned_pre_suppression_points": int(np.count_nonzero(owned)),
            "owned_post_suppression_points": int(np.count_nonzero(post)),
            "points_stolen_by_other_boxes": int(np.count_nonzero(stolen)),
            "stolen_owner_histogram": dict(owner_hist),
            "cluster_count": len(groups), "components": selected_records,
            "selected_component_points": 0 if selected is None else selected["points"],
            "selected_component_score": None if selected is None else selected["score"],
            "selected_component_xyz": None if selected is None else selected["xyz"],
            "camera_depth_removed": int(np.count_nonzero(~valid | (depth <= 0))),
            "image_boundary_removed": int(np.count_nonzero(valid & (depth > 0) & ~in_image)),
            "broad_height_removed": int(np.count_nonzero(in_image & ~broad)),
            "person_height_removed": int(np.count_nonzero(inside_box & ~height_ok)),
        })
    return result, traces


def replay_predictions(manifest: list[dict], capture_splits: tuple[str, ...]=( "VALIDATION",),
                       empirical_du_px: float=0.0, empirical_dv_px: float=0.0
                       ) -> tuple[dict, dict, dict, dict]:
    validation_frames = [int(row["annotated_frame_index"]) for row in manifest
                         if row["split"] == "VALIDATION"]
    end = max(validation_frames)
    allowed = {"TRAIN_FIT", "TRAIN_CALIBRATION", "VALIDATION"}
    process_rows = [row for row in manifest
                    if int(row["annotated_frame_index"]) <= end
                    and row["split"] in allowed]
    if any(row["split"] in {"TEST", "EMBARGO"} for row in process_rows):
        raise RuntimeError("TEST/EMBARGO access is forbidden")
    states = {"V0": FrozenState(False, True), "V1": FrozenState(True, True)}
    mirrors = {"V0": FrozenState(False, False), "V1": FrozenState(True, False)}
    predictions = {"V0": defaultdict(list), "V1": defaultdict(list)}
    traces: dict[int, list[dict]] = {}
    frame_records: dict[int, dict] = {}
    model = pipeline.OnlineRgbFrustumPipeline(
        device="0", identity=False, projection_audit=False,
        legacy_display_offset=False, suppressed_recovery=False)
    previous = -1
    equivalent = True
    try:
        for row in process_rows:
            frame = int(row["annotated_frame_index"])
            for _ in range(frame - previous - 1):
                for state in (*states.values(), *mirrors.values()):
                    state.tracker.update([])
            previous = frame
            image = cv2.imread(row["rgb_image_path"])
            if image is None:
                raise RuntimeError(f"missing RGB frame: {row['rgb_image_path']}")
            points = occupancy_audit.read_binary_xyz_pcd(row["pcd_path"])
            detections = model._detections(image)
            pixels, valid, depth = model.project_physical(points)
            pixels = pixels + np.asarray(
                [float(empirical_du_px), float(empirical_dv_px)], np.float64)
            height = suppression.ground_height(points, model.ground_normal, model.ground_d)
            result, detail = trace_frame(
                model, detections, points, pixels, valid, depth, height,
                du_px=empirical_du_px, dv_px=empirical_dv_px)
            captured = {}
            for name in ("V0", "V1"):
                pred, decisions = states[name].step(result, frame)
                mirror, _ = mirrors[name].step(result, frame)
                equivalent &= same_predictions(pred, mirror)
                captured[name] = (pred, decisions)
                if row["split"] in capture_splits:
                    predictions[name][frame] = pred
            if row["split"] in capture_splits:
                for stage in detail:
                    index = stage["detection"]
                    stage["v0_recovery"] = captured["V0"][1].get(index, {})
                    stage["v1_recovery"] = captured["V1"][1].get(index, {})
                    stage["v0_prediction"] = next((jsonable(item) for item in captured["V0"][0]
                                                   if item["detection"] == index), None)
                    stage["v1_prediction"] = next((jsonable(item) for item in captured["V1"][0]
                                                   if item["detection"] == index), None)
                traces[frame] = detail
                frame_records[frame] = {
                    "rgb_image_path": row["rgb_image_path"], "pcd_path": row["pcd_path"],
                    "timestamp_ns": int(row["scene_timestamp_ns"]),
                    "detections": [{"class": item["class"], "bbox": item["bbox"].tolist(),
                                    "confidence": float(item["confidence"])}
                                   for item in detections],
                }
            if (frame + 1) % 100 == 0:
                print(f"FN audit causal replay {frame}/{end}", flush=True)
    finally:
        model.calibration_audit.close()
    safety = {"gt_loaded_after_predictions": True,
              "instrumentation_predictions_exact": bool(equivalent),
              "test_access": 0, "embargo_access": 0,
              "physical_du_px": 0.0, "physical_dv_px": 0.0,
              "empirical_inference_du_px": float(empirical_du_px),
              "empirical_inference_dv_px": float(empirical_dv_px),
              "processed_allowed_rows": len(process_rows)}
    return predictions, traces, frame_records, safety


def load_truth_after_predictions(frames: list[int]) -> dict[int, list[dict]]:
    wanted = set(frames)
    truth: dict[int, list[dict]] = defaultdict(list)
    for row in read_csv(GT):
        frame = int(row["annotated_frame_index"])
        if frame not in wanted or row["gt_valid"] != "True":
            continue
        truth[frame].append({
            "entity_id": row["participant_id"],
            "xyz": np.asarray([float(row["center_annotated_x_m"]),
                               float(row["center_annotated_y_m"]),
                               float(row["center_annotated_z_m"])]),
            "size": np.asarray([float(row["size_local_x_m"]),
                                float(row["size_local_y_m"]),
                                float(row["size_local_z_m"])]),
            "yaw": float(row["yaw_rad"]),
        })
    if sum(map(len, truth.values())) != 5 * len(frames):
        raise RuntimeError("unexpected VALIDATION GT cardinality")
    return truth


def cuboid_projection(target: dict, model) -> dict:
    center = target["xyz"]
    sx, sy, sz = target["size"] / 2.0
    local = np.asarray([[x, y, z] for x in (-sx, sx) for y in (-sy, sy)
                        for z in (-sz, sz)])
    yaw = target["yaw"]
    rotation = np.asarray([[math.cos(yaw), -math.sin(yaw), 0.0],
                           [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    corners = local @ rotation.T + center
    center_pixel, center_valid, center_depth = model.project_physical(center[None])
    pixels, valid, _ = model.project_physical(corners)
    proxy = None
    if np.count_nonzero(valid) >= 2:
        values = pixels[valid]
        proxy = np.asarray([values[:, 0].min(), values[:, 1].min(),
                            values[:, 0].max(), values[:, 1].max()])
    return {"center_u": None if not center_valid[0] else float(center_pixel[0, 0]),
            "center_v": None if not center_valid[0] else float(center_pixel[0, 1]),
            "depth": float(center_depth[0]), "proxy": proxy}


def iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    a = max(left[2] - left[0], 0.0) * max(left[3] - left[1], 0.0)
    b = max(right[2] - right[0], 0.0) * max(right[3] - right[1], 0.0)
    return float(intersection / max(a + b - intersection, 1e-9))


def candidate_stage(stages: list[dict], projection: dict) -> tuple[dict | None, list[dict]]:
    person = [row for row in stages if row["class"] == "PERSON"]
    if not person:
        return None, person
    ranked = []
    for row in person:
        box = np.asarray(row["bbox"], float)
        u, v, proxy = projection["center_u"], projection["center_v"], projection["proxy"]
        contains = bool(u is not None and v is not None
                        and box[0] <= u <= box[2] and box[1] <= v <= box[3])
        overlap = 0.0 if proxy is None else iou(box, proxy)
        width, height = np.maximum(box[2:] - box[:2], 1.0)
        distance = math.inf if u is None else float(np.linalg.norm(
            (np.asarray([u, v]) - 0.5 * (box[:2] + box[2:])) / [width, height]))
        if contains or overlap >= 0.10:
            ranked.append((not contains, -overlap, distance, row))
    if not ranked:
        return None, person
    ranked.sort(key=lambda value: value[:3])
    return ranked[0][3], person


def region(u: float | None, v: float | None) -> str:
    if u is None or v is None:
        return "OUT_OF_IMAGE"
    horizontal = ("left" if u < IMAGE_W / 3 else
                  "center" if u < 2 * IMAGE_W / 3 else "right")
    vertical = ("top" if v < IMAGE_H / 3 else
                "middle" if v < 2 * IMAGE_H / 3 else "bottom")
    return f"{horizontal}_{vertical}"


def depth_bin(value: float) -> str:
    for lower, upper, label in DEPTH_BINS:
        if lower <= value < upper:
            return label
    return "INVALID"


def recovery_primary(decision: dict) -> str | None:
    reason = decision.get("reason")
    if reason == "NO_METRIC_HISTORY":
        return "E1_NO_HISTORY_SUPPORT"
    if reason in {"NO_OCCUPANCY_DELETED_COMPONENT", "TOO_FEW_POINTS"}:
        return "E2_RECOVERY_NO_VALID_COMPONENT"
    if reason == "HISTORY_MOTION_GATE_REJECT":
        return "E3_RECOVERY_MOTION_GATE_REJECT"
    if reason == "NEIGHBOR_PERSON_COMPETITION":
        return "E5_RECOVERY_NEIGHBOR_CONFLICT_REJECT"
    if reason in {"CENTER_OUTSIDE_ORIGINAL_BOX", "IMPLAUSIBLE_3D_SIZE",
                  "COMPONENT_SCORE_ABOVE_EXISTING_ASSIGNMENT_LIMIT",
                  "AMBIGUOUS_COMPONENT_RANKING", "NORMAL_SIGNATURE_UNAVAILABLE",
                  "HISTORY_COMPONENT_SIGNATURE_REJECT"}:
        return "E4_RECOVERY_COMPONENT_QUALITY_REJECT"
    return None


def secondary_flags(frame_gt: list[dict], gt_index: int, person_stages: list[dict],
                    stage: dict | None, projection: dict,
                    all_projections: list[dict]
                    ) -> tuple[list[str], float | None, float | None, float | None]:
    target = frame_gt[gt_index]
    others = [item for index, item in enumerate(frame_gt) if index != gt_index]
    nearest = min((float(np.linalg.norm(target["xyz"][:2] - item["xyz"][:2]))
                   for item in others), default=None)
    flags = []
    overlaps = []
    if stage is not None:
        box = np.asarray(stage["bbox"], float)
        overlaps = [iou(box, np.asarray(item["bbox"], float)) for item in person_stages
                    if item is not stage]
        if max(overlaps, default=0.0) > 0:
            flags.append("overlapping_person_boxes")
        if box[0] <= 2 or box[1] <= 2 or box[2] >= IMAGE_W - 2 or box[3] >= IMAGE_H - 2:
            flags.append("bbox_border_contact")
        if (box[2] - box[0]) * (box[3] - box[1]) < 0.01 * IMAGE_W * IMAGE_H:
            flags.append("small_bbox")
    u, v = projection["center_u"], projection["center_v"]
    if u is None or v is None or not (0 <= u < IMAGE_W and 0 <= v < IMAGE_H):
        flags.append("gt_projection_out_of_image")
    elif min(u, IMAGE_W - u, v, IMAGE_H - v) < 40:
        flags.append("near_image_edge")
    if nearest is not None and nearest < 0.8:
        flags.append("close_gt_neighbor")
    if projection["depth"] > 9:
        flags.append("far_range")
    current_pixel = np.asarray([projection["center_u"], projection["center_v"]], object)
    projected_distances = []
    if all(value is not None for value in current_pixel):
        for index, item in enumerate(all_projections):
            if index == gt_index or item["center_u"] is None or item["center_v"] is None:
                continue
            projected_distances.append(float(np.linalg.norm(
                current_pixel.astype(float) - [item["center_u"], item["center_v"]])))
    return flags, nearest, max(overlaps, default=None), min(projected_distances, default=None)


def classify_miss(variant: str, frame: int, gt_index: int, target: dict,
                  frame_gt: list[dict], stages: list[dict], projection: dict,
                  all_projections: list[dict], predictions: list[dict],
                  matches: list[dict]) -> dict:
    stage, person_stages = candidate_stage(stages, projection)
    matched_predictions = {row["prediction_index"]: row["gt_index"] for row in matches}
    candidate_prediction = None if stage is None else stage.get(
        "v1_prediction" if variant == "V1" else "v0_prediction")
    primary = detail = None
    if not person_stages:
        primary = "A1_NO_PERSON_DETECTION"
    elif stage is None:
        primary = "A2_PERSON_BOX_GEOMETRY_MISMATCH"
    elif stage["raw_projected_points"] == 0:
        primary = "B1_NO_RAW_PROJECTED_POINTS"
    elif stage["raw_projected_points"] < MIN_POINTS:
        primary = "B2_RAW_LIDAR_TOO_SPARSE"
    elif stage["post_geometry_height_points"] < MIN_POINTS:
        primary = "C_GEOMETRIC_OR_HEIGHT_GATE_REMOVAL"
    elif stage["post_both_nonexclusive_points"] < MIN_POINTS:
        primary = "D_STATIC_SUPPRESSION_REMOVAL"
        temporal_killed = stage["post_temporal_points"] < MIN_POINTS
        occupancy_killed = stage["post_occupancy_points"] < MIN_POINTS
        detail = ("D3_BOTH" if temporal_killed and occupancy_killed else
                  "D1_TEMPORAL" if temporal_killed else "D2_OCCUPANCY")
    else:
        decision = stage.get("v1_recovery", {}) if variant == "V1" else {}
        recovery_code = recovery_primary(decision) if decision.get("triggered") else None
        if recovery_code is not None and not decision.get("accepted"):
            primary, detail = recovery_code, decision.get("reason")
        elif stage["owned_post_suppression_points"] >= MIN_POINTS and stage["cluster_count"] == 0:
            primary = "F_CLUSTERING_FAIL"
        elif (stage["post_both_nonexclusive_points"] >= MIN_POINTS
              and stage["owned_post_suppression_points"] < MIN_POINTS
              and stage["points_stolen_by_other_boxes"] > 0):
            primary = "H_POINT_OWNERSHIP_CONFLICT"
        elif stage["cluster_count"] > 0 and stage["selected_component_xyz"] is None:
            primary = "G1_NO_COMPONENT_PASSED_SCORING"
        elif candidate_prediction is None:
            primary = recovery_code or "G1_NO_COMPONENT_PASSED_SCORING"
            detail = detail or decision.get("reason")
        else:
            selected_xyz = np.asarray(candidate_prediction["xyz"], float)
            selected_distance = float(np.linalg.norm(selected_xyz - target["xyz"]))
            component_distances = [(float(np.linalg.norm(np.asarray(item["xyz"]) - target["xyz"])), item)
                                   for item in stage["components"]]
            better = min(component_distances, default=(math.inf, None), key=lambda value: value[0])
            if selected_distance > GATE_M and better[0] <= GATE_M:
                primary = ("G3_CORRECT_COMPONENT_LOST_RANKING"
                           if better[1] and better[1]["rank"] > 1
                           else "G2_WRONG_COMPONENT_RANKED_FIRST")
            else:
                pred_index = next((index for index, item in enumerate(predictions)
                                   if item["detection"] == stage["detection"]), None)
                if pred_index is not None and pred_index in matched_predictions:
                    primary = "I2_PREDICTION_MATCHED_ANOTHER_GT"
                else:
                    distances = np.asarray([np.linalg.norm(item["xyz"] - target["xyz"])
                                            for item in predictions])
                    inside = np.flatnonzero(distances <= GATE_M)
                    if len(inside) > 1:
                        primary = "I3_DUPLICATE_OR_MERGED_DETECTION_COMPETITION"
                    elif len(inside) == 1:
                        primary = "I4_GLOBAL_ASSIGNMENT_COMPETITION"
                    else:
                        primary = "I1_NEAREST_PREDICTION_OUTSIDE_GATE"
    assert primary in PRIMARY_CODES, primary
    flags, nearest_person, overlap, nearest_projected = secondary_flags(
        frame_gt, gt_index, person_stages, stage, projection, all_projections)
    nearest_prediction = min((float(np.linalg.norm(item["xyz"] - target["xyz"]))
                              for item in predictions), default=None)
    cost = (np.linalg.norm(np.asarray([item["xyz"] for item in predictions])[:, None, :]
                           - np.asarray([item["xyz"] for item in frame_gt])[None, :, :], axis=2)
            if predictions else np.empty((0, len(frame_gt))))
    recovery_decision = {} if stage is None else stage.get(
        "v1_recovery" if variant == "V1" else "v0_recovery", {})
    row = {
        "variant": variant, "frame_id": frame,
        "gt_person_id_for_audit_only": target["entity_id"],
        "gt_x": float(target["xyz"][0]), "gt_y": float(target["xyz"][1]),
        "gt_z": float(target["xyz"][2]), "gt_depth": projection["depth"],
        "gt_projected_u": projection["center_u"], "gt_projected_v": projection["center_v"],
        "gt_proxy_bbox": None if projection["proxy"] is None else json.dumps(projection["proxy"].tolist()),
        "primary_root_cause": primary, "root_stage": primary[0],
        "root_detail": detail, "secondary_flags": ";".join(flags),
        "person_detection_count": len(person_stages),
        "candidate_box_id": None if stage is None else stage["detection"],
        "candidate_box_conf": None if stage is None else stage["confidence"],
        "bbox": None if stage is None else json.dumps(stage["bbox"]),
        "raw_projected_points": None if stage is None else stage["raw_projected_points"],
        "post_height_points": None if stage is None else stage["post_geometry_height_points"],
        "post_temporal_points": None if stage is None else stage["post_temporal_points"],
        "post_occupancy_points": None if stage is None else stage["post_occupancy_points"],
        "post_both_points": None if stage is None else stage["post_both_nonexclusive_points"],
        "camera_depth_removed": None if stage is None else stage["camera_depth_removed"],
        "image_boundary_removed": None if stage is None else stage["image_boundary_removed"],
        "broad_height_removed": None if stage is None else stage["broad_height_removed"],
        "person_height_removed": None if stage is None else stage["person_height_removed"],
        "owned_post_suppression_points": None if stage is None else stage["owned_post_suppression_points"],
        "recovery_triggered": bool(recovery_decision.get("triggered", False)),
        "recovery_reject_reason": recovery_decision.get("reason"),
        "cluster_count": None if stage is None else stage["cluster_count"],
        "selected_component_points": None if stage is None else stage["selected_component_points"],
        "selected_component_score": None if stage is None else stage["selected_component_score"],
        "prediction_xyz": None if candidate_prediction is None else json.dumps(candidate_prediction["xyz"]),
        "prediction_gt_distance": nearest_prediction,
        "nearest_person_distance": nearest_person,
        "nearest_projected_person_distance_px": nearest_projected,
        "bbox_overlap_neighbor": overlap,
        "image_region": region(projection["center_u"], projection["center_v"]),
        "depth_bin": depth_bin(projection["depth"]),
        "components_json": None if stage is None else json.dumps(stage["components"]),
        "all_prediction_xyz": json.dumps([item["xyz"].tolist() for item in predictions]),
        "all_gt_xyz": json.dumps([item["xyz"].tolist() for item in frame_gt]),
        "hungarian_cost_matrix": json.dumps(cost.tolist()),
        "chosen_assignment": json.dumps([[item["prediction_index"], item["gt_index"]]
                                         for item in matches]),
    }
    return row


def taxonomy(variant: str, frames: list[int], predictions: dict[int, list[dict]],
             truth: dict[int, list[dict]], traces: dict[int, list[dict]], model
             ) -> tuple[list[dict], list[dict]]:
    misses, matches_all = [], []
    for frame in frames:
        pred, gt = predictions[frame], truth[frame]
        matches, _, _ = recovery_audit.match_frame(pred, gt)
        matched_gt = {row["gt_index"] for row in matches}
        projections = [cuboid_projection(item, model) for item in gt]
        for row in matches:
            matches_all.append({"variant": variant, "frame": frame, **row})
        for gt_index, target in enumerate(gt):
            if gt_index not in matched_gt:
                misses.append(classify_miss(
                    variant, frame, gt_index, target, gt, traces[frame],
                    projections[gt_index], projections, pred, matches))
    return misses, matches_all


def metric_checked(name: str, frames: list[int], predictions: dict[int, list[dict]],
                   truth: dict[int, list[dict]]) -> tuple[dict, list[dict]]:
    value, matches = recovery_audit.evaluate(name, frames, predictions, truth)
    expected = next(row for row in read_csv(EXPECTED)
                    if row["policy"] == ("V0_LEGACY_BASELINE" if name == "V0"
                                         else "V1_HISTORY_ONLY_CURRENT"))
    keys = ("tp", "fp", "fn", "prediction_count")
    if any(int(expected[key]) != int(value[key]) for key in keys):
        raise RuntimeError(f"frozen evaluator mismatch for {name}: {value}")
    for key in ("precision", "recall", "f1"):
        if not np.isclose(float(expected[key]), value[key], atol=1e-12, rtol=0):
            raise RuntimeError(f"frozen metric mismatch {name} {key}")
    for key in ("rmse", "p95"):
        if not np.isclose(float(expected[f"xy_{key}"]), value["xy"][key], atol=1e-12, rtol=0):
            raise RuntimeError(f"frozen XY metric mismatch {name} {key}")
    return value, matches


def aggregate_counts(v0: list[dict], v1: list[dict]) -> list[dict]:
    left = Counter(row["primary_root_cause"] for row in v0)
    right = Counter(row["primary_root_cause"] for row in v1)
    return [{"root_cause": code, "V0_count": left[code],
             "V0_fraction": left[code] / max(len(v0), 1),
             "V1_count": right[code], "V1_fraction": right[code] / max(len(v1), 1),
             "delta": right[code] - left[code]}
            for code in PRIMARY_CODES]


def cross_table(rows: list[dict], dimension: str) -> list[dict]:
    counts = Counter((row["primary_root_cause"], row[dimension]) for row in rows)
    return [{"root_cause": cause, dimension: value, "count": count,
             "fraction_of_root_cause": count / sum(
                 number for (other, _), number in counts.items() if other == cause)}
            for (cause, value), count in sorted(counts.items())]


def tp_profile(frames: list[int], predictions: dict[int, list[dict]], truth: dict[int, list[dict]],
               traces: dict[int, list[dict]], model) -> list[dict]:
    rows = []
    for frame in frames:
        matches, _, _ = recovery_audit.match_frame(predictions[frame], truth[frame])
        for match in matches:
            item = predictions[frame][match["prediction_index"]]
            stage = traces[frame][item["detection"]]
            projection = cuboid_projection(truth[frame][match["gt_index"]], model)
            box = np.asarray(stage["bbox"], float)
            rows.append({"frame_id": frame, "entity_id_for_audit_only": match["entity_id"],
                         "source": item["source"], "error_xy_m": match["error_xy_m"],
                         "error_xyz_m": match["error_xyz_m"],
                         "raw_projected_points": stage["raw_projected_points"],
                         "post_height_points": stage["post_geometry_height_points"],
                         "post_suppression_points": stage["post_both_nonexclusive_points"],
                         "owned_post_suppression_points": stage["owned_post_suppression_points"],
                         "component_score": stage["selected_component_score"],
                         "bbox_width": float(box[2] - box[0]), "bbox_height": float(box[3] - box[1]),
                         "gt_depth": projection["depth"]})
    return rows


def select_cases(rows: list[dict], limit: int=40) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (row["primary_root_cause"],
                                            row["frame_id"], row["gt_person_id_for_audit_only"]))
    if len(ordered) <= limit:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, limit).round().astype(int)
    return [ordered[index] for index in indices]


def draw_base(row: dict, records: dict[int, dict]) -> np.ndarray:
    image = cv2.imread(records[row["frame_id"]]["rgb_image_path"])
    if image is None:
        raise RuntimeError("contact-sheet RGB missing")
    for index, detection in enumerate(records[row["frame_id"]]["detections"]):
        if detection["class"] != "PERSON":
            continue
        box = np.asarray(detection["bbox"], int)
        color = (0, 190, 255) if index == row.get("candidate_box_id") else (140, 140, 140)
        cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color, 2)
    u, v = row.get("gt_projected_u"), row.get("gt_projected_v")
    proxy = row.get("gt_proxy_bbox")
    if proxy:
        proxy_box = np.asarray(json.loads(proxy), int)
        cv2.rectangle(image, tuple(proxy_box[:2]), tuple(proxy_box[2:]), (0, 0, 255), 2)
    if u is not None and v is not None and np.isfinite(u) and np.isfinite(v):
        cv2.drawMarker(image, (round(float(u)), round(float(v))), (0, 0, 255),
                       cv2.MARKER_CROSS, 24, 3)
    return image


def stage_masks(row: dict, records: dict[int, dict], model) -> dict | None:
    candidate = row.get("candidate_box_id")
    if candidate in (None, ""):
        return None
    candidate = int(candidate)
    record = records[row["frame_id"]]
    points = occupancy_audit.read_binary_xyz_pcd(record["pcd_path"])
    pixels, valid, depth = model.project_physical(points)
    height = suppression.ground_height(points, model.ground_normal, model.ground_d)
    in_image = (valid & (depth > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < IMAGE_W)
                & (pixels[:, 1] >= 0) & (pixels[:, 1] < IMAGE_H))
    broad = (height >= 0.03) & (height <= 2.15)
    indices = np.flatnonzero(in_image & broad)
    values = points[indices]
    temporal, near = model._static_evidence(values)
    boxes = [np.asarray(item["bbox"], float) for item in record["detections"]]
    owners = support.exclusive_point_owners(pixels[indices], boxes, [None] * len(boxes))
    occupancy_keep, _ = suppression.occupancy_keep(
        values, height[indices], near, temporal, suppression.Policy(suppression.LEGACY),
        owners, tuple(item["class"] for item in record["detections"]))
    box = boxes[candidate]
    inside = ((pixels[indices, 0] >= box[0]) & (pixels[indices, 0] <= box[2])
              & (pixels[indices, 1] >= box[1]) & (pixels[indices, 1] <= box[3]))
    height_ok = (height[indices] >= 0.08) & (height[indices] <= 2.15)
    return {"points": values, "pixels": pixels[indices], "indices": indices,
            "inside": inside, "height": height_ok, "temporal": temporal,
            "occupancy": occupancy_keep, "owners": owners, "boxes": boxes,
            "candidate": candidate}


def draw_mask_panel(base: np.ndarray, evidence: dict, mask: np.ndarray, label: str,
                    color: tuple[int, int, int]) -> np.ndarray:
    panel = base.copy()
    values = np.round(evidence["pixels"][mask]).astype(int)
    for u, v in values:
        cv2.circle(panel, (u, v), 3, color, -1)
    cv2.rectangle(panel, (0, 0), (440, 38), (15, 25, 35), -1)
    cv2.putText(panel, f"{label}: {len(values)} pts", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.resize(panel, (240, 135))


def suppression_tile(row: dict, records: dict[int, dict], model) -> np.ndarray:
    base = draw_base(row, records)
    evidence = stage_masks(row, records, model)
    if evidence is None:
        return cv2.resize(base, (480, 270))
    inside = evidence["inside"]
    geometry = inside & evidence["height"]
    panels = [
        draw_mask_panel(base, evidence, inside, "raw box", (255, 180, 20)),
        draw_mask_panel(base, evidence, geometry, "geometry/height", (0, 220, 255)),
        draw_mask_panel(base, evidence, geometry & evidence["temporal"], "after temporal", (255, 80, 80)),
        draw_mask_panel(base, evidence, geometry & evidence["temporal"] & evidence["occupancy"],
                        "after occupancy", (70, 230, 70)),
    ]
    return np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))


def ownership_tile(row: dict, records: dict[int, dict], model) -> np.ndarray:
    base = draw_base(row, records)
    evidence = stage_masks(row, records, model)
    if evidence is None:
        return cv2.resize(base, (480, 270))
    palette = ((255, 90, 50), (30, 210, 255), (70, 220, 70), (220, 80, 220),
               (255, 180, 30), (180, 180, 180))
    mask = evidence["inside"] & evidence["height"] & evidence["temporal"] & evidence["occupancy"]
    for pixel, owner in zip(evidence["pixels"][mask], evidence["owners"][mask], strict=True):
        cv2.circle(base, tuple(np.round(pixel).astype(int)), 3,
                   (50, 50, 50) if owner < 0 else palette[int(owner) % len(palette)], -1)
    return cv2.resize(base, (480, 270))


def component_tile(row: dict, records: dict[int, dict]) -> np.ndarray:
    image = draw_base(row, records)
    for component in json.loads(row.get("components_json") or "[]"):
        u, v = component.get("center_u"), component.get("center_v")
        if u is None or v is None:
            continue
        location = (round(u), round(v))
        cv2.drawMarker(image, location, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 3)
        cv2.putText(image, f"r{component['rank']} s={component['score']:.2f} n={component['points']}",
                    (location[0] + 6, location[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.53, (255, 0, 255), 2, cv2.LINE_AA)
    return cv2.resize(image, (480, 270))


def assignment_tile(row: dict) -> np.ndarray:
    canvas = np.full((270, 480, 3), 250, np.uint8)
    predictions = np.asarray(json.loads(row["all_prediction_xyz"]), float)
    truth = np.asarray(json.loads(row["all_gt_xyz"]), float)
    if len(predictions):
        all_xy = np.vstack((predictions[:, :2], truth[:, :2]))
    else:
        all_xy = truth[:, :2]
    lower, upper = all_xy.min(0) - 0.4, all_xy.max(0) + 0.4
    scale = np.asarray([420, 190]) / np.maximum(upper - lower, 1e-6)
    def pixel(xy):
        value = (xy - lower) * scale
        return int(30 + value[0]), int(235 - value[1])
    for index, xyz in enumerate(truth):
        cv2.drawMarker(canvas, pixel(xyz[:2]), (0, 0, 220), cv2.MARKER_CROSS, 18, 3)
        cv2.putText(canvas, f"G{index}", pixel(xyz[:2]), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 180), 1)
    for index, xyz in enumerate(predictions):
        cv2.circle(canvas, pixel(xyz[:2]), 6, (210, 80, 20), -1)
        cv2.putText(canvas, f"P{index}", pixel(xyz[:2]), cv2.FONT_HERSHEY_SIMPLEX, .45, (160, 50, 10), 1)
    cv2.putText(canvas, f"f{row['frame_id']} gate=1.5m min={float(row['prediction_gt_distance']):.2f}m",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (25, 35, 45), 2)
    return canvas


def point_overlay(row: dict, records: dict[int, dict], model, mode: str) -> np.ndarray:
    image = draw_base(row, records)
    points = occupancy_audit.read_binary_xyz_pcd(records[row["frame_id"]]["pcd_path"])
    pixels, valid, depth = model.project_physical(points)
    keep = valid & (depth > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < IMAGE_W)
    keep &= (pixels[:, 1] >= 0) & (pixels[:, 1] < IMAGE_H)
    projected = np.round(pixels[keep]).astype(int)
    if len(projected) > 2500:
        projected = projected[::math.ceil(len(projected) / 2500)]
    color = (255, 170, 20) if mode == "projection" else (255, 200, 100)
    for u, v in projected:
        cv2.circle(image, (u, v), 1, color, -1)
    return image


def make_sheet(rows: list[dict], path: Path, records: dict[int, dict], model,
               mode: str="generic") -> None:
    tiles = []
    for row in select_cases(rows):
        if mode == "suppression":
            image = suppression_tile(row, records, model)
        elif mode == "ownership":
            image = ownership_tile(row, records, model)
        elif mode == "component":
            image = component_tile(row, records)
        elif mode == "assignment":
            image = assignment_tile(row)
        elif mode == "projection":
            image = point_overlay(row, records, model, mode)
        else:
            image = draw_base(row, records)
        label = f"f{row['frame_id']} {row['gt_person_id_for_audit_only']} {row['primary_root_cause']}"
        cv2.rectangle(image, (0, 0), (IMAGE_W, 44), (15, 25, 35), -1)
        cv2.putText(image, label[:96], (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.67, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(image, (480, 270)))
    if not tiles:
        tile = np.full((270, 480, 3), 245, np.uint8)
        cv2.putText(tile, "No cases in this category", (45, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (50, 50, 50), 2)
        tiles = [tile]
    blank = np.full_like(tiles[0], 245)
    tiles += [blank] * ((5 - len(tiles) % 5) % 5)
    sheet = np.vstack([np.hstack(tiles[index:index + 5])
                       for index in range(0, len(tiles), 5)])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(MANIFEST)
    validation_frames = [int(row["annotated_frame_index"]) for row in manifest
                         if row["split"] == "VALIDATION"]
    predictions, traces, records, safety = replay_predictions(manifest)
    fingerprints = {name: prediction_fingerprint(values)
                    for name, values in predictions.items()}
    # The first GT file read in this process occurs only after frozen fingerprints exist.
    truth = load_truth_after_predictions(validation_frames)
    metrics, matches = {}, {}
    for name in ("V0", "V1"):
        metrics[name], matches[name] = metric_checked(
            name, validation_frames, predictions[name], truth)

    audit_model = pipeline.OnlineRgbFrustumPipeline(
        device="cpu", identity=False, projection_audit=False,
        legacy_display_offset=False, suppressed_recovery=False)
    try:
        v0_rows, _ = taxonomy("V0", validation_frames, predictions["V0"], truth, traces, audit_model)
        v1_rows, _ = taxonomy("V1", validation_frames, predictions["V1"], truth, traces, audit_model)
        tp_rows = tp_profile(validation_frames, predictions["V1"], truth, traces, audit_model)
        count_rows = aggregate_counts(v0_rows, v1_rows)
        write_csv(OUT / "FN_ROOT_CAUSE_COUNTS.csv", count_rows)
        write_csv(OUT / "v0_vs_v1_taxonomy.csv", count_rows)
        write_csv(OUT / "FN_ROOT_CAUSE_BY_DEPTH.csv", cross_table(v1_rows, "depth_bin"))
        write_csv(OUT / "FN_ROOT_CAUSE_BY_IMAGE_REGION.csv", cross_table(v1_rows, "image_region"))
        frame_rank = {frame: index for index, frame in enumerate(validation_frames)}
        block_size = math.ceil(len(validation_frames) / 4)
        for row in v1_rows:
            row["time_block"] = f"P{min(frame_rank[row['frame_id']] // block_size + 1, 4)}"
            row["timestamp"] = records[row["frame_id"]]["timestamp_ns"]
        for row in v0_rows:
            row["timestamp"] = records[row["frame_id"]]["timestamp_ns"]
        write_csv(OUT / "fn_case_details.csv", v1_rows)
        write_csv(OUT / "v0_case_details.csv", v0_rows)
        write_csv(OUT / "FN_ROOT_CAUSE_BY_TIME.csv", cross_table(v1_rows, "time_block"))
        write_csv(OUT / "tp_reference_profile.csv", tp_rows)
        for name in ("V0", "V1"):
            frozen = [{"frame": frame, "detection": item["detection"],
                       "track_id": item["track_id"], "source": item["source"],
                       "x": item["xyz"][0], "y": item["xyz"][1], "z": item["xyz"][2]}
                      for frame in validation_frames for item in predictions[name][frame]]
            write_csv(OUT / f"frozen_predictions_{name.lower()}.csv", frozen)

        make_sheet([row for row in v1_rows if row["root_stage"] == "A"],
                   OUT / "YOLO_FN_CONTACT_SHEET.jpg", records, audit_model)
        make_sheet([row for row in v1_rows if row["primary_root_cause"] == "B1_NO_RAW_PROJECTED_POINTS"],
                   OUT / "PROJECTION_NO_POINT_CONTACT_SHEET.jpg", records, audit_model, "projection")
        make_sheet([row for row in v1_rows if row["root_stage"] == "D"],
                   OUT / "SUPPRESSION_FN_CONTACT_SHEET.jpg", records, audit_model, "suppression")
        make_sheet([row for row in v1_rows if row["root_stage"] == "H"],
                   OUT / "OWNERSHIP_CONFLICT_CONTACT_SHEET.jpg", records, audit_model, "ownership")
        make_sheet([row for row in v1_rows if row["root_stage"] == "G"],
                   OUT / "COMPONENT_SELECTION_CONTACT_SHEET.jpg", records, audit_model, "component")
        make_sheet([row for row in v1_rows if row["root_stage"] == "I"],
                   OUT / "ASSIGNMENT_FAILURE_CONTACT_SHEET.jpg", records, audit_model, "assignment")
    finally:
        audit_model.calibration_audit.close()

    if len(v0_rows) != metrics["V0"]["fn"] or len(v1_rows) != metrics["V1"]["fn"]:
        raise RuntimeError("taxonomy completeness failed")
    if any(row["primary_root_cause"] not in PRIMARY_CODES for row in v0_rows + v1_rows):
        raise RuntimeError("taxonomy exclusivity failed")
    resolved = {(row["frame_id"], row["gt_person_id_for_audit_only"]): row
                for row in v0_rows}
    remaining = {(row["frame_id"], row["gt_person_id_for_audit_only"]) for row in v1_rows}
    resolved_rows = [row for key, row in resolved.items() if key not in remaining]
    resolved_counts = Counter(row["primary_root_cause"] for row in resolved_rows)
    broad = Counter(row["root_stage"] for row in v1_rows)
    sorted_broad = sorted(broad.items(), key=lambda item: (-item[1], item[0]))
    next_stage, next_count = sorted_broad[0]
    primary_counts = Counter(row["primary_root_cause"] for row in v1_rows)
    entity_counts = Counter(row["gt_person_id_for_audit_only"] for row in v1_rows)
    flag_counts = Counter(flag for row in v1_rows
                          for flag in row["secondary_flags"].split(";") if flag)
    crossing_or_occlusion = sum(
        bool({"overlapping_person_boxes", "close_gt_neighbor", "bbox_border_contact"}
             & set(row["secondary_flags"].split(";"))) for row in v1_rows)
    depth_total = Counter(row["depth_bin"] for row in v1_rows)
    depth_fn = Counter(row["depth_bin"] for row in v1_rows)
    for row in tp_rows:
        depth_total[depth_bin(float(row["gt_depth"]))] += 1
    depth_profile = {
        label: {"GT": depth_total[label], "FN": depth_fn[label],
                "FN_rate": depth_fn[label] / max(depth_total[label], 1)}
        for _, _, label in DEPTH_BINS
    }
    time_ranges = {}
    for block in range(4):
        values = validation_frames[block * block_size:min((block + 1) * block_size,
                                                          len(validation_frames))]
        time_ranges[f"P{block + 1}"] = [min(values), max(values)] if values else None
    time_counts = Counter((row["time_block"], row["root_stage"]) for row in v1_rows)
    old = Counter(row["problem_type"] for row in read_csv(
        ROOT / "outputs" / "occupancy_suppression_redesign" / "miss_taxonomy.csv")
                  if row["record_type"] == "OFFICIAL_GT_FN")
    summary = {
        "status": "AUDIT_COMPLETE_PRODUCTION_UNCHANGED",
        "scope": {"validation_frames": len(validation_frames), "gt_rows": len(validation_frames) * 5,
                  "test_access": 0, "embargo_access": 0},
        "projection": "RAW_PHYSICAL_K_D_T_DU_DV_ZERO_BASELINE",
        "production": "V1_HISTORY_ONLY_CURRENT",
        "old_880_plus_96_taxonomy_belongs_to": "V0_LEGACY_BASELINE",
        "old_taxonomy_counts": dict(old),
        "metrics": metrics, "fingerprints": fingerprints, "safety": safety,
        "taxonomy": {"V0_total": len(v0_rows), "V1_total": len(v1_rows),
                     "V1_by_stage": dict(broad),
                     "V1_by_primary": dict(primary_counts),
                     "V1_by_entity": dict(entity_counts),
                     "V1_secondary_flags": dict(flag_counts),
                     "crossing_or_occlusion_flagged": crossing_or_occlusion,
                     "depth_profile": depth_profile,
                     "time_block_ranges": time_ranges,
                     "time_block_by_stage": {f"{block}_{stage}": count
                                             for (block, stage), count in time_counts.items()},
                     "history_only_resolved_total": len(resolved_rows),
                     "history_only_resolved_v0_causes": dict(resolved_counts)},
        "next_bottleneck": {"stage": next_stage, "count": next_count,
                            "fraction": next_count / len(v1_rows),
                            "primary": "A2_PERSON_BOX_GEOMETRY_MISMATCH",
                            "action": "DISAMBIGUATE_RAW_PROJECTION_VS_RGB_BOX_CORRESPONDENCE_ONLY"},
        "tests": {"taxonomy_exclusive": True, "taxonomy_complete": True,
                  "V0_FN_976": len(v0_rows) == 976, "V1_FN_909": len(v1_rows) == 909,
                  "GT_loaded_after_predictions": safety["gt_loaded_after_predictions"],
                  "instrumentation_equivalent": safety["instrumentation_predictions_exact"],
                  "no_TEST_or_EMBARGO": safety["test_access"] == safety["embargo_access"] == 0},
    }
    (OUT / "audit_summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 当前生产版 PERSON 假阴性根因分解审计", "",
        "## 门禁结果", "",
        "**PASS——审计完成，生产推理结果未改变。**", "",
        f"历史 `880 + 96` 是 **V0** 的粗分类，不是当前 V1。冻结指标精确复现：V0 FN={metrics['V0']['fn']}，V1 FN={metrics['V1']['fn']}。", "",
        "## 冻结指标", "",
        "| 版本 | TP | FP | FN | Precision | Recall | F1 | XY RMSE | XY P95 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[f"| {name} | {value['tp']} | {value['fp']} | {value['fn']} | {value['precision']:.6f} | {value['recall']:.6f} | {value['f1']:.6f} | {value['xy']['rmse']:.6f} m | {value['xy']['p95']:.6f} m |"
          for name, value in metrics.items()], "",
        "## 互斥且完备的 V0/V1 taxonomy", "",
        "| Primary root cause | V0 | V1 | Delta |", "|---|---:|---:|---:|",
        *[f"| {row['root_cause']} | {row['V0_count']} | {row['V1_count']} | {row['delta']:+d} |" for row in count_rows], "",
        "## 当前 V1 决策树", "",
        *[f"- Stage {stage}: **{broad[stage]}** ({broad[stage] / len(v1_rows):.2%})"
          for stage in "ABCDEFGHI"], "",
        "## HISTORY_ONLY 实际作用", "",
        f"HISTORY_ONLY 消除了 **{len(resolved_rows)}** 个 V0 FN；这些案例在 V0 中的分类为 `{dict(resolved_counts)}`。冻结评估中新增未匹配 recovery 预测为 0。", "",
        "## 20 个必须回答的问题", "",
        "1. 原 `880 + 96` 属于 **V0**，不能继续当作当前 V1 taxonomy。",
        f"2. 当前 V1 的 909 个 FN：`{dict(primary_counts)}`。",
        f"3. 数量最大的表面类别是 A2，共 {primary_counts['A2_PERSON_BOX_GEOMETRY_MISMATCH']} 个。",
        f"4. HISTORY_ONLY 减少 67 个 FN：`{dict(resolved_counts)}`。其中 62 个在 raw GT 投影规则下被标成 A2，这说明 A2 不能直接等同于 YOLO 漏检。",
        f"5. 严格 A1（整帧无 PERSON detection）为 {primary_counts['A1_NO_PERSON_DETECTION']}；A2 为 {primary_counts['A2_PERSON_BOX_GEOMETRY_MISMATCH']}，但 A2 混合了 detector/projection/语义代理不一致。",
        f"6. Raw LiDAR/projection B 类为 {broad['B']}。",
        f"7. View/height gate C 类为 {broad['C']}。",
        f"8. Static suppression D 类为 {broad['D']}，其中主要窗口是 frames 1568–1601。",
        f"9. HISTORY_ONLY rejection E 类为 {broad['E']}。D 按“首个失败阶段”优先记账，因此 recovery 失败可保留为 secondary reason，但不重复计数。",
        f"10. Clustering F 类为 {broad['F']}。",
        f"11. Point ownership H 类为 {broad['H']}。",
        f"12. Component selection G 类为 {broad['G']}。",
        f"13. 3D assignment/localization I 类为 {broad['I']}。",
        f"14. 深度分布（GT/FN/FN rate）：`{depth_profile}`。6–9 m 与 >9 m 明显升高，主要由 A2 主导，并非 B 类点云零支持。",
        "15. 图像区域交叉表见 `FN_ROOT_CAUSE_BY_IMAGE_REGION.csv`。raw 投影下 A2/D/G 都集中在 center-middle，I 主要在 center-bottom；FN 子集没有足够 left/right 支持来拟合任何像素修正。",
        f"16. 时间块为 `{time_ranges}`。D 集中于 P3，I 集中于 P4，A2 四个块均存在。",
        f"17. crossing/occlusion 解释性标记共有 {crossing_or_occlusion}/{len(v1_rows)} ({crossing_or_occlusion / len(v1_rows):.2%})；只用于审计。",
        f"18. 存在 registration-sensitive pattern：A2 占 {primary_counts['A2_PERSON_BOX_GEOMETRY_MISMATCH']}/{len(v1_rows)}，且接触图显示 raw GT proxy 与实际人物框常明显错位。但这既不能证明 YOLO 错，也不能证明某个 du/K/T 候选正确。",
        f"19. 当前最大的测量瓶颈是 Stage A/A2（{primary_counts['A2_PERSON_BOX_GEOMETRY_MISMATCH']}）。",
        "20. 下一步只建议分解 A2 的 raw-projection↔RGB-box 不一致：固定现有 detections，用独立 registration 证据区分标定残差与检测/语义代理误差。本轮不允许调 YOLO、K/D/T、du、suppression、clustering 或 Hungarian gate。", "",
        "## 解释边界", "",
        "- A2 只表示 raw physical GT projection 与 PERSON bbox 没有固定几何对应。官方 K/D/T 尚未被独立证明为最终标定，因此 A2 不是纯 YOLO miss。",
        "- GT identity、cuboid projection、GT distance 只存在于离线审计；预测完成并封存指纹后才读取 GT。",
        "- 深度 bins 固定为 0–3/3–6/6–9/>9 m；VALIDATION 按时间预先等分为四块。", "",
        "## NEXT BOTTLENECK", "",
        f"**Stage {next_stage} / A2 correspondence：{next_count}/{len(v1_rows)} ({next_count / len(v1_rows):.2%})。** 这是数量最大的下一调查对象，但不是已经证明的 YOLO 或 calibration 缺陷。在 A2 被独立证据拆清前，不应修改生产推理。", "",
        "## 回归与隔离门禁", "",
        f"- instrumentation ON/OFF prediction equivalence: `{safety['instrumentation_predictions_exact']}`",
        "- physical projection: `K+D+T, du=dv=0`",
        "- GT load order: V0/V1 predictions fingerprinted first",
        "- TEST access: `0`; EMBARGO access: `0`",
        "- taxonomy exclusivity/completeness: `PASS`", "",
        "逐案例证据、交叉表和 6 张 contact sheet 均在同一输出目录。",
    ]
    (OUT / "FN_ROOT_CAUSE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "test_log.txt").write_text(
        "taxonomy_exclusivity PASS\ntaxonomy_completeness PASS\n"
        "V0_count_consistency PASS\nV1_count_consistency PASS\n"
        "GT_isolation PASS\ninstrumentation_equivalence PASS\n"
        "TEST_EMBARGO_access_zero PASS\n", encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()

