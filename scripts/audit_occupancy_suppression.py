"""TRAIN_FIT/VALIDATION-only occupancy-suppression redesign audit.

Predictions are generated causally with raw K+D+T and du=dv=0.  Ground truth
is loaded only after each scored split has been predicted.  TEST and EMBARGO
rows are rejected before any sensor stream is opened.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from online_v4 import pipeline, recovery, support, suppression  # noqa: E402
import audit_soft_ownership as soft_audit  # noqa: E402
import audit_suppressed_point_recovery as recovery_audit  # noqa: E402


OUT = ROOT / "outputs" / "occupancy_suppression_redesign"
MANIFEST = Path(r"D:\navwareset_scene01_clean\data\splits\annotated_split_manifest.csv")
HEIGHT_GRID = (0.15, 0.20, 0.25, 0.30, 0.35)
CROSSING_FRAMES = set(range(1360, 1386)) | {1403} | set(range(1707, 1720))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_binary_xyz_pcd(path: str | Path) -> np.ndarray:
    """Read the canonical binary XYZ PCD without touching bag-only holdout rows."""
    header: dict[str, list[str]] = {}
    with Path(path).open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError(f"PCD header is incomplete: {path}")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break
        if header.get("FIELDS") != ["x", "y", "z"] or header.get("DATA") != ["binary"]:
            raise RuntimeError(f"Expected binary XYZ PCD: {path}")
        count = int(header["POINTS"][0])
        raw = stream.read()
    values = np.frombuffer(raw, dtype="<f4", count=3 * count)
    if values.size != 3 * count:
        raise RuntimeError(f"PCD point count mismatch: {path}")
    return values.reshape(count, 3).astype(np.float64)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values), q))


def policy_key(policy: suppression.Policy) -> str:
    if policy.name in {suppression.GROUND_LIMITED, suppression.HEIGHT_BANDED}:
        return f"{policy.name}_h{policy.low_height_m:.2f}"
    return policy.name


def choose_component(components: list[np.ndarray], detection: dict, model,
                     du_px: float=0.0, dv_px: float=0.0
                     ) -> tuple[np.ndarray | None, dict]:
    ranked = []
    for component in components:
        score, details = support.component_score(
            component, detection["bbox"], model.transform, model.K, model.D,
            du_px=du_px, dv_px=dv_px)
        if np.isfinite(score):
            ranked.append((float(score), -len(component), component, details))
    if not ranked:
        return None, {"quality_guard": False, "candidate_components": 0}
    ranked.sort(key=lambda value: (value[0], value[1]))
    score, _, component, details = ranked[0]
    center = support.cluster_center(component)
    pixel, valid, _ = support.project_points(
        center[None], model.transform, model.K, model.D,
        du_px=du_px, dv_px=dv_px)
    box = detection["bbox"]
    inside = bool(valid[0] and box[0] <= pixel[0, 0] <= box[2]
                  and box[1] <= pixel[0, 1] <= box[3])
    span = np.ptp(component, axis=0)
    horizontal, vertical = float(max(span[0], span[1])), float(span[2])
    if detection["class"] == "PERSON":
        plausible = 0.10 <= horizontal <= 1.10 and 0.10 <= vertical <= 2.20
    else:
        plausible = 0.10 <= horizontal <= 1.35 and 0.08 <= vertical <= 1.45
    second_gap = None if len(ranked) < 2 else float(ranked[1][0] - score)
    return component, {
        **details, "component_score": score, "component_points": len(component),
        "candidate_components": len(ranked), "center_inside_original": inside,
        "plausible_size": plausible,
        "quality_guard": bool(inside and plausible and score <= 1.25),
        "horizontal_span_m": horizontal, "vertical_span_m": vertical,
        "second_score_gap": second_gap,
        "center_u": None if not valid[0] else float(pixel[0, 0]),
        "center_v": None if not valid[0] else float(pixel[0, 1]),
    }


@dataclass
class PolicyResult:
    detections: list[dict]
    quality: list[dict]
    recovery_candidates: list[recovery.RecoveryCandidate | None]
    stages: list[dict]
    point_stats: dict[str, int]
    runtime_ms: float


class VariantState:
    def __init__(self, name: str, with_history: bool) -> None:
        self.name = name
        self.with_history = with_history
        self.tracker = pipeline.OnlineTracker(np.empty((0, 0), np.float32), ())
        self.history = recovery.CausalSuppressedPointRecovery(False)
        self.counts: dict[str, Counter] = defaultdict(Counter)

    def update(self, result: PolicyResult, frame: int, split: str
               ) -> tuple[list[dict], list[dict]]:
        detections = [{**item, "bbox": item["bbox"].copy(),
                       "xyz": None if item["xyz"] is None else item["xyz"].copy()}
                      for item in result.detections]
        tracked = self.tracker.update(detections)
        predictions, visuals = [], []
        for index, (item, quality, candidate) in enumerate(zip(
                tracked, result.quality, result.recovery_candidates, strict=True)):
            if item["class"] != "PERSON":
                continue
            counts = self.counts[split]
            counts["person_boxes"] += 1
            source = "NORMAL_COMPONENT" if item["xyz"] is not None else "UNAVAILABLE"
            accepted = False
            if item["xyz"] is not None:
                counts["normal_components"] += 1
                counts["guarded_normal_components"] += int(quality.get("quality_guard", False))
                counts["components_rejected_by_quality_gate"] += int(
                    not quality.get("quality_guard", False))
                self.history.observe_normal(item["track_id"], item["xyz"], frame)
            elif self.with_history:
                decision = self.history.consider(item["track_id"], candidate, frame)
                accepted = bool(decision["accepted"])
                counts["history_attempts"] += int(candidate is not None)
                counts["history_recovered"] += int(accepted)
                if accepted:
                    assert candidate is not None
                    item["xyz"] = candidate.xyz.copy()
                    item["component_points"] = candidate.point_count
                    source = "RECOVERED_SUPPRESSED_COMPONENT"
                    state = self.tracker.tracks.get(item["track_id"])
                    if state is not None:
                        state["xyz"] = candidate.xyz.copy()
            if item["xyz"] is None:
                continue
            visual = {
                "frame": frame, "variant": self.name, "track_id": item["track_id"],
                "source": source, "detection": index,
                "bbox_x1": float(item["bbox"][0]), "bbox_y1": float(item["bbox"][1]),
                "bbox_x2": float(item["bbox"][2]), "bbox_y2": float(item["bbox"][3]),
                "x": float(item["xyz"][0]), "y": float(item["xyz"][1]),
                "z": float(item["xyz"][2]),
                "component_points": int(item.get("component_points", 0)),
                "quality_guard": bool(quality.get("quality_guard", accepted)),
                "horizontal_span_m": quality.get("horizontal_span_m"),
                "vertical_span_m": quality.get("vertical_span_m"),
                "center_u": quality.get("center_u"), "center_v": quality.get("center_v"),
                "neighbor_competition": bool(
                    (candidate.neighbor_competition if accepted and candidate is not None else False)
                    or quality.get("containing_person_boxes", 0) > 1),
                "inside_robot_box": bool(quality.get("inside_robot_box", False)),
            }
            visuals.append(visual)
            predictions.append({
                "xyz": item["xyz"].copy(), "track_id": item["track_id"],
                "source": source, "detection": index, **visual})
        return predictions, visuals


def policy_result(model, detections: list[dict], candidate_points: np.ndarray,
                  candidate_indices: np.ndarray, pixels: np.ndarray,
                  ground_height: np.ndarray, temporal: np.ndarray,
                  occupancy_near: np.ndarray, owners: np.ndarray,
                  policy: suppression.Policy, du_px: float=0.0,
                  dv_px: float=0.0) -> PolicyResult:
    started = time.perf_counter()
    classes = tuple(item["class"] for item in detections)
    occupancy_keep, decision = suppression.occupancy_keep(
        candidate_points, ground_height[candidate_indices], occupancy_near,
        temporal, policy, owners, classes)
    keep = suppression.combined_keep(temporal, occupancy_keep)
    filtered = owners.copy()
    filtered[~keep] = -1
    for index, detection in enumerate(detections):
        lower, upper = ((0.08, 2.15) if detection["class"] == "PERSON"
                        else (0.03, 1.35))
        invalid = ((ground_height[candidate_indices] < lower)
                   | (ground_height[candidate_indices] > upper))
        filtered[(filtered == index) & invalid] = -1
    if model.geometry_gpu:
        groups = model.geometry_gpu.components(candidate_points, filtered, len(detections))
    else:
        groups = [support.adaptive_components(candidate_points[filtered == index])
                  for index in range(len(detections))]

    suppressed = temporal & ~occupancy_keep
    recovery_owners = owners.copy()
    recovery_owners[~suppressed] = -1
    for index, detection in enumerate(detections):
        if detection["class"] != "PERSON":
            recovery_owners[recovery_owners == index] = -1
            continue
        invalid = ((ground_height[candidate_indices] < 0.08)
                   | (ground_height[candidate_indices] > 2.15))
        recovery_owners[(recovery_owners == index) & invalid] = -1
    if model.geometry_gpu:
        recovery_groups = model.geometry_gpu.components(
            candidate_points, recovery_owners, len(detections))
    else:
        recovery_groups = [support.adaptive_components(
            candidate_points[recovery_owners == index])
            for index in range(len(detections))]

    person_boxes = [item["bbox"] for item in detections if item["class"] == "PERSON"]
    output, qualities, recovery_candidates, stages = [], [], [], []
    for index, (detection, components) in enumerate(zip(detections, groups, strict=True)):
        component, quality = choose_component(
            components, detection, model, du_px=du_px, dv_px=dv_px)
        item = {**detection, "bbox": detection["bbox"].copy(), "xyz": None,
                "component_points": 0}
        if component is not None:
            item["xyz"] = support.cluster_center(component)
            item["component_points"] = len(component)
            u, v = quality.get("center_u"), quality.get("center_v")
            if u is not None and v is not None:
                quality["containing_person_boxes"] = sum(
                    other["class"] == "PERSON"
                    and other["bbox"][0] <= u <= other["bbox"][2]
                    and other["bbox"][1] <= v <= other["bbox"][3]
                    for other in detections)
                quality["inside_robot_box"] = any(
                    other["class"] == "ROBOT"
                    and other["bbox"][0] <= u <= other["bbox"][2]
                    and other["bbox"][1] <= v <= other["bbox"][3]
                    for other in detections)
        if detection["class"] == "PERSON" and component is None:
            candidate, _ = recovery_audit.choose_recovery_candidate(
                recovery_groups[index], detection, person_boxes, model,
                du_px=du_px, dv_px=dv_px)
        else:
            candidate = None
        before = int(np.count_nonzero(owners == index))
        after = int(np.count_nonzero(filtered == index))
        stages.append({
            "detection": index, "class": detection["class"],
            "bbox_x1": float(detection["bbox"][0]),
            "bbox_y1": float(detection["bbox"][1]),
            "bbox_x2": float(detection["bbox"][2]),
            "bbox_y2": float(detection["bbox"][3]),
            "points_before_suppression": before,
            "points_after_suppression_and_height": after,
            "component_count": len(components),
            "component_available": component is not None,
            "quality_guard": bool(quality.get("quality_guard", False)),
            "center_u": quality.get("center_u"), "center_v": quality.get("center_v"),
        })
        output.append(item); qualities.append(quality); recovery_candidates.append(candidate)
    person_owner = np.zeros(len(owners), bool)
    for index, item in enumerate(detections):
        if item["class"] == "PERSON":
            person_owner |= owners == index
    point_stats = {
        "points_before_suppression": len(candidate_points),
        "points_removed_temporal": int(np.count_nonzero(~temporal)),
        "points_removed_occupancy": int(np.count_nonzero(temporal & ~occupancy_keep)),
        "points_restored_by_new_logic": decision["points_restored_by_new_logic"],
        "person_box_occupancy_removals": int(np.count_nonzero(
            person_owner & temporal & ~occupancy_keep)),
        "person_box_restored_points": decision["person_box_restored_points"],
    }
    return PolicyResult(output, qualities, recovery_candidates, stages,
                        point_stats, 1000.0 * (time.perf_counter() - started))


def metric(policy: str, frames: list[int], predictions: dict[int, list[dict]],
           truth: dict[int, list[dict]], state: VariantState) -> tuple[dict, list[dict]]:
    value, matches = recovery_audit.evaluate(policy, frames, predictions, truth)
    errors = [row["error_xy_m"] for row in matches]
    value["xy"]["mae"] = None if not errors else float(np.mean(errors))
    counts = state.counts["TRAIN_FIT" if frames and frames[0] < 979 else "VALIDATION"]
    boxes = counts["person_boxes"]
    measurements = value["prediction_count"]
    guarded = counts["guarded_normal_components"] + counts["history_recovered"]
    value.update({
        "component_coverage": measurements / max(boxes, 1),
        "guarded_component_coverage": guarded / max(boxes, 1),
        "person_boxes": boxes,
        "normal_components": counts["normal_components"],
        "normal_components_guarded": counts["guarded_normal_components"],
        "components_rejected_by_shape_quality_gate": counts["components_rejected_by_quality_gate"],
        "components_recovered_through_history_only": counts["history_recovered"],
    })
    return value, matches


def select_config(metrics: dict[str, dict], names: list[str], reference: dict) -> str:
    def admissible(value: dict) -> bool:
        return (value["precision"] >= reference["precision"] - 0.005
                and value["xy"]["rmse"] <= reference["xy"]["rmse"] + 0.05
                and value["xy"]["p95"] <= reference["xy"]["p95"] + 0.10)
    return max(names, key=lambda name: (
        admissible(metrics[name]), metrics[name]["precision"], metrics[name]["f1"],
        -metrics[name]["fp"], -metrics[name]["xy"]["rmse"]))


def flatten(value: dict) -> dict:
    row = {key: item for key, item in value.items() if not isinstance(item, dict)}
    for prefix in ("xy", "xyz", "recovered_xy", "recovered_xyz"):
        for key, item in value.get(prefix, {}).items():
            row[f"{prefix}_{key}"] = item
    return row


def point_statistics(rows: dict[tuple[str, str], Counter], timings: dict[tuple[str, str], list[float]],
                     aliases: dict[str, str]) -> list[dict]:
    output = []
    for (split, config), counts in rows.items():
        label = aliases.get(config, config)
        values = timings[(split, config)]
        output.append({"split": split, "variant": label, **counts,
                       "suppression_and_component_mean_ms": float(np.mean(values)),
                       "suppression_and_component_p95_ms": percentile(values, 95),
                       "suppression_and_component_p99_ms": percentile(values, 99)})
    return output


def taxonomy_misses(frames: list[int], predictions: dict[int, list[dict]],
                    truth: dict[int, list[dict]], stages: dict[int, list[dict]], model) -> list[dict]:
    rows = []
    for frame in frames:
        pred, gt = predictions[frame], truth[frame]
        matches, _, _ = recovery_audit.match_frame(pred, gt)
        matched_gt = {row["gt_index"] for row in matches}
        for gt_index, target in enumerate(gt):
            if gt_index in matched_gt:
                continue
            pixel, valid, depth = model.project_physical(target["xyz"][None])
            stage_rows = stages.get(frame, [])
            containing = []
            if valid[0] and depth[0] > 0:
                for row in stage_rows:
                    if (row["class"] == "PERSON"
                            and row["bbox_x1"] <= pixel[0, 0] <= row["bbox_x2"]
                            and row["bbox_y1"] <= pixel[0, 1] <= row["bbox_y2"]):
                        containing.append(row)
            if not containing:
                reason, stage = "A_NO_PROJECTED_POINTS_OR_PERSON_BOX", None
            else:
                stage = min(containing, key=lambda row: abs(
                    0.5 * (row["bbox_x1"] + row["bbox_x2"]) - pixel[0, 0]))
                if stage["points_before_suppression"] < 3:
                    reason = "A_NO_PROJECTED_POINTS_IN_BOX"
                elif stage["points_after_suppression_and_height"] < 3:
                    reason = "B_SUPPRESSION_LEAVES_TOO_FEW_POINTS"
                elif stage["component_count"] == 0:
                    reason = "C_CLUSTERING_FAILS"
                elif not stage["quality_guard"]:
                    reason = "D_SHAPE_OR_QUALITY_GATE"
                elif stage["component_available"]:
                    reason = "E_COMPONENT_ASSIGNMENT_FAILS"
                else:
                    reason = "F_TRACKING_ASSOCIATION_FAILURE"
            rows.append({
                "record_type": "OFFICIAL_GT_FN", "frame": frame,
                "entity_id": target["entity_id"], "problem_type": reason,
                "gt_x": float(target["xyz"][0]), "gt_y": float(target["xyz"][1]),
                "projected_u": None if not valid[0] else float(pixel[0, 0]),
                "projected_v": None if not valid[0] else float(pixel[0, 1]),
                **({} if stage is None else stage)})
    return rows


def taxonomy_detection_misses(stages: dict[int, list[dict]]) -> list[dict]:
    """Classify every RGB PERSON observation for which baseline builds no component."""
    rows = []
    for frame, values in stages.items():
        for stage in values:
            if stage["class"] != "PERSON" or stage["component_available"]:
                continue
            if stage["points_before_suppression"] < 3:
                reason = "A_NO_PROJECTED_POINTS_IN_BOX"
            elif stage["points_after_suppression_and_height"] < 3:
                reason = "B_SUPPRESSION_LEAVES_TOO_FEW_POINTS"
            elif stage["component_count"] == 0:
                reason = "C_CLUSTERING_FAILS"
            elif not stage["quality_guard"]:
                reason = "D_SHAPE_OR_QUALITY_GATE"
            else:
                reason = "E_COMPONENT_ASSIGNMENT_FAILS"
            rows.append({"record_type": "RGB_PERSON_NO_COMPONENT",
                         "frame": frame, "problem_type": reason, **stage})
    return rows


def taxonomy_fp(variant: str, frames: list[int], predictions: dict[int, list[dict]],
                truth: dict[int, list[dict]], baseline: dict[int, list[dict]], model) -> list[dict]:
    rows = []
    for frame in frames:
        pred = predictions[frame]
        matches, _, _ = recovery_audit.match_frame(pred, truth[frame])
        matched = {row["prediction_index"] for row in matches}
        base_matches, _, _ = recovery_audit.match_frame(baseline[frame], truth[frame])
        base_unmatched = [baseline[frame][index] for index in range(len(baseline[frame]))
                          if index not in {row["prediction_index"] for row in base_matches}]
        for index, item in enumerate(pred):
            if index in matched:
                continue
            height = suppression.ground_height(
                np.asarray([[item["x"], item["y"], item["z"]]]),
                model.ground_normal, model.ground_d)[0]
            horizontal = item.get("horizontal_span_m")
            vertical = item.get("vertical_span_m")
            if item.get("neighbor_competition"):
                kind = "neighbor_person_contamination"
            elif item.get("inside_robot_box"):
                kind = "robot_contamination"
            elif height < 0.25 or (vertical is not None and vertical < 0.10):
                kind = "wall_or_floor"
            elif horizontal is not None and horizontal > 0.9 and height < 1.4:
                kind = "furniture_or_static_structure"
            elif any(np.linalg.norm(np.asarray([item["x"], item["y"]])
                                    - other["xyz"][:2]) < 0.75
                     for other in pred if other is not item):
                kind = "split_person"
            else:
                kind = "unknown"
            incremental = not any(np.linalg.norm(
                np.asarray([item["x"], item["y"]]) - other["xyz"][:2]) < 0.25
                for other in base_unmatched)
            rows.append({**item, "variant": variant, "problem_type": kind,
                         "ground_height_m": float(height),
                         "incremental_vs_history_only": incremental})
    return rows


def contact_sheet(manifest: list[dict], rows: list[dict], path: Path,
                  title_key: str="problem_type") -> None:
    selected, seen = [], set()
    for row in rows:
        if row["frame"] in seen:
            continue
        selected.append(row); seen.add(row["frame"])
        if len(selected) == 16:
            break
    tiles = []
    for row in selected:
        image = cv2.imread(manifest[row["frame"]]["rgb_image_path"])
        if image is None:
            continue
        if all(key in row for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
            box = np.asarray([row["bbox_x1"], row["bbox_y1"],
                              row["bbox_x2"], row["bbox_y2"]], int)
            cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 0, 230), 3)
        u, v = row.get("center_u", row.get("projected_u")), row.get("center_v", row.get("projected_v"))
        if u is not None and v is not None and np.isfinite(u) and np.isfinite(v):
            cv2.drawMarker(image, (int(u), int(v)), (0, 220, 255), cv2.MARKER_CROSS, 24, 3)
        label = f"f{row['frame']} {row.get(title_key, '')}"
        cv2.putText(image, label[:74], (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (0, 0, 220), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(image, (480, 270)))
    if not tiles:
        tiles = [np.full((270, 480, 3), 245, np.uint8)]
        cv2.putText(tiles[0], "No cases", (160, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (60, 60, 60), 2, cv2.LINE_AA)
    blank = np.full_like(tiles[0], 245)
    tiles += [blank] * ((4 - len(tiles) % 4) % 4)
    sheet = np.vstack([np.hstack(tiles[index:index + 4])
                       for index in range(0, len(tiles), 4)])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def duplicate_gt_occupation(frames: list[int], predictions: dict[int, list[dict]],
                            truth: dict[int, list[dict]]) -> int:
    """Count unmatched recovered components competing with an already matched normal GT."""
    duplicates = 0
    for frame in frames:
        pred, gt = predictions[frame], truth[frame]
        matches, _, _ = recovery_audit.match_frame(pred, gt)
        matched_pred = {row["prediction_index"] for row in matches}
        normal_targets = [gt[row["gt_index"]]["xyz"] for row in matches
                          if pred[row["prediction_index"]]["source"] == "NORMAL_COMPONENT"]
        for index, item in enumerate(pred):
            if (index in matched_pred
                    or item["source"] != "RECOVERED_SUPPRESSED_COMPONENT"):
                continue
            duplicates += int(any(
                np.linalg.norm(item["xyz"] - target) <= recovery_audit.GATE_M
                for target in normal_targets))
    return duplicates


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(MANIFEST)
    train_frames = [int(row["annotated_frame_index"]) for row in manifest
                    if row["split"] == "TRAIN_FIT"]
    validation_frames = [int(row["annotated_frame_index"]) for row in manifest
                         if row["split"] == "VALIDATION"]
    process_end = max(validation_frames)
    allowed_splits = {"TRAIN_FIT", "TRAIN_CALIBRATION", "VALIDATION"}
    process_manifest = [row for row in manifest
                        if int(row["annotated_frame_index"]) <= process_end
                        and row["split"] in allowed_splits]
    if any(row["split"] in {"TEST", "EMBARGO"} for row in process_manifest):
        raise RuntimeError("TEST/EMBARGO would be accessed")
    if train_frames != list(range(979)):
        raise RuntimeError("TRAIN_FIT is not the expected frozen 0..978 prefix")

    policies = [
        suppression.Policy(suppression.LEGACY),
        suppression.Policy(suppression.PHYSICAL_DIRECT),
        *[suppression.Policy(suppression.GROUND_LIMITED, value) for value in HEIGHT_GRID],
        *[suppression.Policy(suppression.HEIGHT_BANDED, value, 1.90) for value in HEIGHT_GRID],
        suppression.Policy(suppression.TEMPORAL_PRIMARY, 0.25),
        suppression.Policy(suppression.PERSON_CONDITIONAL),
    ]
    states: dict[tuple[str, bool], VariantState] = {}
    for policy in policies:
        key = policy_key(policy)
        states[(key, False)] = VariantState(key, False)
        states[(key, True)] = VariantState(key + "+HISTORY_ONLY", True)
    predictions: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    visuals: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    stage_rows: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    timings: dict[tuple[str, str], list[float]] = defaultdict(list)
    selected_policy_keys: set[str] | None = None
    validation_aliases: dict[str, tuple[str, bool]] = {}
    train_metrics: dict[str, dict] = {}
    train_matches: dict[str, list[dict]] = {}
    freeze_receipt: dict = {}
    equivalence = {"checked": False, "keep_mask_equal": False,
                   "owner_mask_equal": False, "component_xyz_equal": False}

    model = pipeline.OnlineRgbFrustumPipeline(
        device="0", identity=False, projection_audit=False,
        legacy_display_offset=False, suppressed_recovery=False)
    cpu_tree = cKDTree(model.occupancy)
    normal_norm = float(np.linalg.norm(model.ground_normal))
    try:
        previous_frame = -1
        frozen = False
        for row in process_manifest:
            frame_index = int(row["annotated_frame_index"])
            if frame_index > 978 and not frozen:
                truth = recovery_audit.load_truth(train_frames)
                for name in list(predictions):
                    if name.startswith("VALIDATION::"):
                        continue
                    key, history_flag = name.rsplit("::", 1)
                    state = states[(key, history_flag == "HISTORY")]
                    value, matches = metric(name, train_frames, predictions[name], truth, state)
                    train_metrics[name] = value; train_matches[name] = matches
                legacy_history = policy_key(suppression.Policy(suppression.LEGACY)) + "::HISTORY"
                reference = train_metrics[legacy_history]
                gl_names = [f"{policy_key(suppression.Policy(suppression.GROUND_LIMITED, h))}::HISTORY" for h in HEIGHT_GRID]
                hb_names = [f"{policy_key(suppression.Policy(suppression.HEIGHT_BANDED, h, 1.9))}::HISTORY" for h in HEIGHT_GRID]
                selected_gl = select_config(train_metrics, gl_names, reference).split("::")[0]
                selected_hb = select_config(train_metrics, hb_names, reference).split("::")[0]
                fixed = [selected_gl, selected_hb, suppression.TEMPORAL_PRIMARY,
                         suppression.PERSON_CONDITIONAL]
                best_history_name = select_config(
                    train_metrics, [name + "::HISTORY" for name in fixed], reference)
                best_key = best_history_name.split("::")[0]
                validation_aliases = {
                    "V0_LEGACY_BASELINE": (suppression.LEGACY, False),
                    "V1_HISTORY_ONLY_CURRENT": (suppression.LEGACY, True),
                    "V2_PHYSICAL_HEIGHT_DIRECT_REFERENCE": (suppression.PHYSICAL_DIRECT, False),
                    "V3_GROUND_LIMITED": (selected_gl, False),
                    "V4_HEIGHT_BANDED": (selected_hb, False),
                    "V5_TEMPORAL_PRIMARY_OCCUPANCY_SECONDARY": (suppression.TEMPORAL_PRIMARY, False),
                    "V6_PERSON_CONDITIONAL_PROTECTION": (suppression.PERSON_CONDITIONAL, False),
                    "V7_BEST_REDESIGN_PLUS_HISTORY_ONLY": (best_key, True),
                }
                selected_policy_keys = {key for key, _ in validation_aliases.values()}
                freeze_receipt = {
                    "frozen_after_train_fit_frame": 978,
                    "train_fit_frames": len(train_frames),
                    "selected_ground_limited": selected_gl,
                    "selected_height_banded": selected_hb,
                    "selected_best_redesign": best_key,
                    "selection_reference": legacy_history,
                    "validation_updates": 0, "test_updates": 0,
                    "embargo_updates": 0,
                }
                frozen = True
            active_keys = ({policy_key(policy) for policy in policies}
                           if not frozen else (selected_policy_keys or set()))
            for _ in range(frame_index - previous_frame - 1):
                for key in active_keys:
                    states[(key, False)].tracker.update([])
                    states[(key, True)].tracker.update([])
            previous_frame = frame_index
            image = cv2.imread(row["rgb_image_path"])
            if image is None:
                raise RuntimeError(f"RGB frame missing: {row['rgb_image_path']}")
            annotated = read_binary_xyz_pcd(row["pcd_path"])
            detections = model._detections(image)
            pixels, valid, depth = model.project_physical(annotated)
            ground_height = suppression.ground_height(
                annotated, model.ground_normal, model.ground_d)
            in_view = (valid & (depth > 0) & (pixels[:, 0] >= 0)
                       & (pixels[:, 0] < pipeline.IMAGE_SIZE[0])
                       & (pixels[:, 1] >= 0) & (pixels[:, 1] < pipeline.IMAGE_SIZE[1]))
            candidate_indices = np.flatnonzero(
                in_view & (ground_height >= 0.03) & (ground_height <= 2.15))
            candidate_points = annotated[candidate_indices]
            temporal, near = model._static_evidence(candidate_points)
            boxes = [item["bbox"] for item in detections]
            owners = (model.geometry_gpu.owners(pixels[candidate_indices], boxes)
                      if model.geometry_gpu else support.exclusive_point_owners(
                          pixels[candidate_indices], boxes, [None] * len(boxes)))

            if not equivalence["checked"] and model.geometry_gpu:
                keys = np.floor(candidate_points / model.voxel_size).astype(np.int32)
                cpu_temporal = np.fromiter(
                    (tuple(map(int, key)) not in model.static for key in keys),
                    bool, len(keys))
                cpu_near = cpu_tree.query(candidate_points[:, :2], workers=-1)[0] <= 0.07
                cpu_owners = support.exclusive_point_owners(
                    pixels[candidate_indices], boxes, [None] * len(boxes))
                equivalence.update({
                    "checked": True,
                    "keep_mask_equal": bool(np.array_equal(temporal, cpu_temporal)
                                            and np.array_equal(near, cpu_near)),
                    "owner_mask_equal": bool(np.array_equal(owners, cpu_owners)),
                })
                gpu_groups = model.geometry_gpu.components(candidate_points, owners, len(boxes))
                cpu_groups = [support.adaptive_components(candidate_points[cpu_owners == index])
                              for index in range(len(boxes))]
                gpu_centers = sorted(tuple(np.round(support.cluster_center(c), 10))
                                     for group in gpu_groups for c in group)
                cpu_centers = sorted(tuple(np.round(support.cluster_center(c), 10))
                                     for group in cpu_groups for c in group)
                equivalence["component_xyz_equal"] = gpu_centers == cpu_centers

            split = row["split"]
            active = policies if frame_index < 979 else [
                policy for policy in policies if policy_key(policy) in (selected_policy_keys or set())]
            for policy in active:
                key = policy_key(policy)
                result = policy_result(
                    model, detections, candidate_points, candidate_indices,
                    pixels, ground_height, temporal, near, owners, policy)
                stats[(split, key)].update(result.point_stats)
                timings[(split, key)].append(result.runtime_ms)
                for history_flag in (False, True):
                    state = states[(key, history_flag)]
                    pred, visual = state.update(result, frame_index, split)
                    if frame_index < 979:
                        name = f"{key}::{'HISTORY' if history_flag else 'BASE'}"
                        predictions[name][frame_index] = pred
                        visuals[name][frame_index] = visual
                        if key == suppression.LEGACY and not history_flag:
                            stage_rows["TRAIN_FIT"][frame_index] = result.stages
                    elif split == "VALIDATION":
                        for alias, state_key in validation_aliases.items():
                            if state_key == (key, history_flag):
                                predictions["VALIDATION::" + alias][frame_index] = pred
                                visuals["VALIDATION::" + alias][frame_index] = visual
                                if alias == "V0_LEGACY_BASELINE":
                                    stage_rows["VALIDATION"][frame_index] = result.stages
            if (frame_index + 1) % 100 == 0:
                print(f"occupancy redesign allowed frame {frame_index}/{process_end}", flush=True)
    finally:
        model.calibration_audit.close()

    validation_truth = recovery_audit.load_truth(validation_frames)
    validation_metrics, validation_matches = {}, {}
    for alias, (key, history_flag) in validation_aliases.items():
        name = "VALIDATION::" + alias
        state = states[(key, history_flag)]
        value, matches = metric(alias, validation_frames, predictions[name],
                                validation_truth, state)
        validation_metrics[alias] = value; validation_matches[alias] = matches

    v1 = validation_metrics["V1_HISTORY_ONLY_CURRENT"]
    v7 = validation_metrics["V7_BEST_REDESIGN_PLUS_HISTORY_ONLY"]
    duplicate_gt = duplicate_gt_occupation(
        validation_frames,
        predictions["VALIDATION::V7_BEST_REDESIGN_PLUS_HISTORY_ONLY"],
        validation_truth)
    net_fp = v7["fp"] - v1["fp"]
    selected_config = validation_aliases["V7_BEST_REDESIGN_PLUS_HISTORY_ONLY"][0]
    runtime_values = timings[("VALIDATION", selected_config)]
    baseline_runtime_values = timings[("VALIDATION", suppression.LEGACY)]
    promotion_checks = {
        "f1_strictly_improves": v7["f1"] > v1["f1"],
        "precision_within_0_002": v7["precision"] >= v1["precision"] - 0.002,
        "fp_increase_at_most_5": net_fp <= 5,
        "xy_rmse_not_worse_by_0_02m": v7["xy"]["rmse"] <= v1["xy"]["rmse"] + 0.02,
        "xy_p95_not_worse_by_0_05m": v7["xy"]["p95"] <= v1["xy"]["p95"] + 0.05,
        "no_duplicate_gt_occupation": duplicate_gt == 0,
        "no_neighbor_identity_transition": v7["neighbor_identity_transitions"] == 0,
        "suppression_component_p95_below_100ms": (percentile(runtime_values, 95) or math.inf) < 100.0,
        "cpu_cuda_equivalent": all(equivalence.values()),
    }
    promoted = all(promotion_checks.values())

    train_rows = [flatten(value) for value in train_metrics.values()]
    validation_rows = [flatten(validation_metrics[key]) for key in validation_aliases]
    aliases = {key: alias for alias, (key, _) in validation_aliases.items()}
    point_rows = point_statistics(stats, timings, aliases)
    official_misses = taxonomy_misses(
        validation_frames, predictions["VALIDATION::V0_LEGACY_BASELINE"],
        validation_truth, stage_rows["VALIDATION"], model)
    detection_misses = taxonomy_detection_misses(stage_rows["VALIDATION"])
    misses = official_misses + detection_misses
    fp_rows = taxonomy_fp(
        "V7_BEST_REDESIGN_PLUS_HISTORY_ONLY", validation_frames,
        predictions["VALIDATION::V7_BEST_REDESIGN_PLUS_HISTORY_ONLY"],
        validation_truth,
        predictions["VALIDATION::V1_HISTORY_ONLY_CURRENT"], model)
    crossing_rows = [row for frame in sorted(CROSSING_FRAMES & set(validation_frames))
                     for row in visuals["VALIDATION::V7_BEST_REDESIGN_PLUS_HISTORY_ONLY"].get(frame, [])]

    write_csv(OUT / "TRAIN_FIT_VARIANTS.csv", train_rows)
    write_csv(OUT / "VALIDATION_VARIANTS.csv", validation_rows)
    write_csv(OUT / "suppression_point_statistics.csv", point_rows)
    write_csv(OUT / "miss_taxonomy.csv", misses)
    write_csv(OUT / "fp_taxonomy.csv", fp_rows)
    write_csv(OUT / "variant_metrics.csv", validation_rows)
    contact_sheet(manifest, sorted(misses, key=lambda row: (
        row.get("record_type") != "RGB_PERSON_NO_COMPONENT",
        row["problem_type"] != "B_SUPPRESSION_LEAVES_TOO_FEW_POINTS",
        row["problem_type"])),
                  OUT / "CONTACT_SHEET_MISSES.jpg")
    contact_sheet(manifest, sorted(fp_rows, key=lambda row: (
        not row["incremental_vs_history_only"], row["problem_type"])),
                  OUT / "CONTACT_SHEET_FP.jpg")
    contact_sheet(manifest, crossing_rows,
                  OUT / "CONTACT_SHEET_CROSSING.jpg", "variant")

    summary = {
        "status": "PROMOTED" if promoted else "NO_OCCUPANCY_REDESIGN_PROMOTED",
        "scope": {"train_fit_frames": len(train_frames),
                  "validation_frames": len(validation_frames),
                  "test_access": 0, "embargo_access": 0},
        "physical_projection": "RAW_K_D_T_DU_DV_ZERO_NOT_PROVEN_FINAL",
        "ground_plane": {"normal": model.ground_normal.tolist(),
                         "normal_norm": normal_norm, "d": float(model.ground_d)},
        "freeze_receipt": freeze_receipt,
        "validation_metrics": validation_metrics,
        "promotion_checks": promotion_checks,
        "cpu_cuda_equivalence": equivalence,
        "net_v7_vs_v1": {"tp": v7["tp"] - v1["tp"],
                         "fp": net_fp, "fn": v7["fn"] - v1["fn"],
                         "component_coverage_pp": 100 * (v7["component_coverage"] - v1["component_coverage"]),
                         "guarded_coverage_pp": 100 * (v7["guarded_component_coverage"] - v1["guarded_component_coverage"]),
                         "f1": v7["f1"] - v1["f1"],
                         "xy_rmse_m": v7["xy"]["rmse"] - v1["xy"]["rmse"],
                         "xy_p95_m": v7["xy"]["p95"] - v1["xy"]["p95"]},
        "taxonomies": {
                       "official_gt_fn": dict(Counter(row["problem_type"] for row in official_misses)),
                       "rgb_person_no_component": dict(Counter(row["problem_type"] for row in detection_misses)),
                       "fp": dict(Counter(row["problem_type"] for row in fp_rows)),
                       "incremental_fp": sum(row["incremental_vs_history_only"] for row in fp_rows)},
        "runtime": {"current_legacy_suppression_component_mean_ms": float(np.mean(baseline_runtime_values)),
                    "current_legacy_suppression_component_p95_ms": percentile(baseline_runtime_values, 95),
                    "current_legacy_suppression_component_p99_ms": percentile(baseline_runtime_values, 99),
                    "selected_suppression_component_mean_ms": float(np.mean(runtime_values)),
                    "selected_suppression_component_p95_ms": percentile(runtime_values, 95),
                    "selected_suppression_component_p99_ms": percentile(runtime_values, 99),
                    "selected_minus_legacy_mean_ms": float(np.mean(runtime_values) - np.mean(baseline_runtime_values)),
                    "production_lidar_p95_ms_unchanged": 55.03},
        "candidate_applied_to_inference": False,
    }
    (OUT / "variant_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    delta = summary["net_v7_vs_v1"]
    report = [
        "# Physically Correct Occupancy Suppression Redesign", "",
        "## Decision", "",
        f"**{summary['status']}**. The audit did not modify production inference while evaluating candidates.", "",
        "Raw physical projection remains `K + D + T`, `du=dv=0`; all calibration candidates remain report-only. TEST and EMBARGO access are both zero.", "",
        "## Frozen VALIDATION comparison", "",
        "| Variant | Coverage | Guarded coverage | Precision | Recall | F1 | XY RMSE | XY P95 | TP/FP/FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in validation_aliases:
        value = validation_metrics[name]
        report.append(
            f"| {name} | {value['component_coverage']:.2%} | {value['guarded_component_coverage']:.2%} | "
            f"{value['precision']:.3f} | {value['recall']:.3f} | {value['f1']:.3f} | "
            f"{value['xy']['rmse']:.3f} m | {value['xy']['p95']:.3f} m | "
            f"{value['tp']}/{value['fp']}/{value['fn']} |")
    report += [
        "", "## Required answers", "",
        "1. **Legacy error.** It used `point_z - signed_ground_distance`. For a nearly horizontal plane this is approximately the plane's absolute Z, not point-to-plane height, so the occupancy threshold was dimensionally coupled to the old bug.",
        "2. **Why direct signed distance adds FP.** The old 0.28/1.90 m thresholds were tuned around the wrong feature. Reusing them protects many points near walls/furniture; coverage rises but incorrect components also survive.",
        f"3. **New physical definition.** `ground_height=(n·p+d)/||n||`; measured normal norm is `{normal_norm:.12f}`.",
        "4. **Occupancy priority.** In redesigned candidates occupancy is secondary; temporal-static evidence remains the primary hard prior.",
        "5. **Temporal use.** Every final keep mask is `temporal_keep AND occupancy_keep`; no occupancy rule restores a temporal-static point.",
        "6. **PERSON protection.** Only temporal-keep points exclusively owned by one current PERSON box and within 0.08–2.15 m signed ground height can avoid occupancy-only deletion.",
        f"7–8. **Net TP/FP.** V7 versus current V1: `{delta['tp']:+d}` TP and `{delta['fp']:+d}` FP.",
        f"9–10. **Coverage.** Component `{delta['component_coverage_pp']:+.2f}` pp; guarded `{delta['guarded_coverage_pp']:+.2f}` pp.",
        f"11. **P/R/F1.** V1 `{v1['precision']:.3f}/{v1['recall']:.3f}/{v1['f1']:.3f}`; V7 `{v7['precision']:.3f}/{v7['recall']:.3f}/{v7['f1']:.3f}`.",
        f"12. **XY error.** RMSE `{delta['xy_rmse_m']:+.3f}` m; P95 `{delta['xy_p95_m']:+.3f}` m.",
        f"13. **Crossing/occlusion.** Neighbor transitions: `{v7['neighbor_identity_transitions']}`; see `CONTACT_SHEET_CROSSING.jpg`.",
        f"14. **HISTORY_ONLY.** V7 history recovered `{v7['components_recovered_through_history_only']}` VALIDATION components.",
        f"15. **Runtime.** Selected decision+component mean/P95/P99: `{summary['runtime']['selected_suppression_component_mean_ms']:.3f}` / `{summary['runtime']['selected_suppression_component_p95_ms']:.3f}` / `{summary['runtime']['selected_suppression_component_p99_ms']:.3f}` ms; mean delta versus legacy `{summary['runtime']['selected_minus_legacy_mean_ms']:+.3f}` ms. No candidate was promoted, so production LiDAR P95 remains the measured `55.03 ms`.",
        f"16. **CPU/CUDA.** `{equivalence}`.",
        "17. **Data access.** TEST=0, EMBARGO=0.",
        f"18. **Promotion.** `{'YES' if promoted else 'NO'}`. Candidate inference write-back remains false in this audit.",
        f"19–20. **Reason.** Promotion checks: `{promotion_checks}`. " + ("The frozen candidate improves the current HISTORY_ONLY Pareto point under every gate." if promoted else "At least one mandatory precision/error/FP/equivalence gate failed, so current HISTORY_ONLY production is retained."),
        "", "## Miss and FP taxonomy", "",
        f"Official GT FN taxonomy: `{summary['taxonomies']['official_gt_fn']}`.", "",
        f"RGB PERSON observations with no baseline component: `{summary['taxonomies']['rgb_person_no_component']}`. This is the detection-centric taxonomy comparable to the earlier 65-frame prefix audit; it must not be conflated with official GT FN taxonomy.", "",
        f"V7 FP taxonomy (automatic heuristic, visually auditable): `{summary['taxonomies']['fp']}`; incremental versus V1: `{summary['taxonomies']['incremental_fp']}`.", "",
        "## Pareto reading", "",
        "Use `VALIDATION_VARIANTS.csv` for Recall–Precision, Coverage–XY RMSE and Coverage–FP comparisons. The selected result is not the maximum-coverage row by construction: physics correctness and precision precede recall/coverage.", "",
        "## Files", "",
        "`TRAIN_FIT_VARIANTS.csv`, `VALIDATION_VARIANTS.csv`, `suppression_point_statistics.csv`, `miss_taxonomy.csv`, `fp_taxonomy.csv`, the three contact sheets, `variant_metrics.json`, and this report.",
    ]
    (OUT / "OCCUPANCY_SUPPRESSION_REDESIGN_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

