"""Audit causal recovery of PERSON components removed only by occupancy prior.

The experiment is strictly report-only. Runtime inference remains unchanged;
GT is loaded only after predictions have been produced and is used solely for
offline scoring and high-risk selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from online_v4 import pipeline, recovery, support  # noqa: E402

CLEAN_ROOT = Path(r"D:\navwareset_scene01_clean")
OUT = ROOT / "outputs" / "suppressed_point_recovery_audit"
SPLIT = CLEAN_ROOT / "data/splits/annotated_split_manifest.csv"
GT = CLEAN_ROOT / "data/canonical/gt/person_cuboid_gt_v2.csv"
GATE_M = 1.5
POLICIES = ("BASELINE", "GROUND_HEIGHT_SEMANTICS_CORRECTED",
            "HISTORY_ONLY", "HISTORY_GUARDED_HEIGHT_CORRECTION",
            "HEIGHT_CORRECTION_HISTORY_OR_3FRAME_SEED",
            "HISTORY_OR_3FRAME_SEED")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def distribution(values: list[float]) -> dict:
    if not values:
        return {"rmse": None, "median": None, "p95": None, "max": None}
    data = np.asarray(values, np.float64)
    return {"rmse": float(np.sqrt(np.mean(data ** 2))),
            "median": float(np.median(data)),
            "p95": float(np.percentile(data, 95)),
            "max": float(np.max(data))}


def component_signature(component: np.ndarray, box: np.ndarray, model,
                        du_px: float=0.0, dv_px: float=0.0
                        ) -> recovery.ComponentSignature:
    center = support.cluster_center(component)
    pixel, valid, _ = support.project_points(
        center[None], model.transform, model.K, model.D,
        du_px=du_px, dv_px=dv_px)
    if not valid[0]:
        raise RuntimeError("Selected component center cannot be projected")
    width, height = np.maximum(box[2:4] - box[0:2], 1.0)
    target_u = 0.5 * (box[0] + box[2])
    target_v = box[1] + 0.54 * height
    span = np.ptp(component, axis=0)
    return recovery.ComponentSignature(
        horizontal_span_m=float(max(span[0], span[1])),
        vertical_span_m=float(span[2]), point_count=len(component),
        center_u_norm=float((pixel[0, 0] - target_u) / width),
        center_v_norm=float((pixel[0, 1] - target_v) / height))


class NormalConsistencyEstimator:
    """Collect only normal-component causal changes from TRAIN_FIT."""

    def __init__(self) -> None:
        self.history: dict[str, dict] = {}
        self.values: dict[str, list[float]] = defaultdict(list)

    def observe(self, track_id: str, xyz: np.ndarray,
                signature: recovery.ComponentSignature, frame_index: int) -> None:
        xyz = np.asarray(xyz, np.float64)
        previous = self.history.get(track_id)
        velocity = np.zeros(3, np.float64)
        if previous is not None:
            gap = int(frame_index) - int(previous["frame_index"])
            if 0 < gap <= 10:
                prediction = previous["xyz"] + previous["velocity"] * gap
                self.values["innovation_m"].append(
                    float(np.linalg.norm(xyz[:2] - prediction[:2])))
                deltas = recovery.CausalSuppressedPointRecovery._signature_deltas(
                    signature, previous["signature"])
                for key, value in deltas.items():
                    self.values[key].append(value)
                instant = (xyz - previous["xyz"]) / gap
                velocity = 0.72 * previous["velocity"] + 0.28 * instant
        velocity[2] = 0.0
        self.history[track_id] = {"xyz": xyz.copy(), "velocity": velocity,
                                  "signature": signature,
                                  "frame_index": int(frame_index)}

    @staticmethod
    def _robust_limit(values: list[float]) -> float:
        data = np.asarray(values, np.float64)
        if len(data) < 100:
            raise RuntimeError("Too few normal TRAIN_FIT transitions to freeze a gate")
        median = float(np.median(data))
        sigma = 1.4826 * float(np.median(np.abs(data - median)))
        robust = median + 3.0 * sigma
        return float(min(np.percentile(data, 99),
                         max(robust, np.percentile(data, 90))))

    def freeze(self) -> recovery.FrozenConsistencyThresholds:
        return recovery.FrozenConsistencyThresholds(
            innovation_m=min(support.TRACK_COMPONENT_GATE_M,
                             self._robust_limit(self.values["innovation_m"])),
            horizontal_log_ratio=self._robust_limit(
                self.values["horizontal_log_ratio"]),
            vertical_log_ratio=self._robust_limit(
                self.values["vertical_log_ratio"]),
            point_count_log_ratio=self._robust_limit(
                self.values["point_count_log_ratio"]),
            center_u_norm_delta=self._robust_limit(
                self.values["center_u_norm_delta"]),
            center_v_norm_delta=self._robust_limit(
                self.values["center_v_norm_delta"]))


def load_manifest_scope(score_split: str, frames: int | None
                        ) -> tuple[list[dict], list[int], str]:
    all_rows = read_csv(SPLIT)
    if frames is not None:
        score_rows = [row for row in all_rows
                      if row["split"] == "TRAIN_FIT"][:frames]
        label = f"TRAIN_FIT_PREFIX_{frames}"
    else:
        score_rows = [row for row in all_rows if row["split"] == score_split]
        label = score_split
    score_frames = [int(row["annotated_frame_index"]) for row in score_rows]
    if not score_frames:
        raise RuntimeError(f"No frames in score scope {label}")
    process_end = max(score_frames)
    process_rows = all_rows[:process_end + 1]
    if [int(row["annotated_frame_index"]) for row in process_rows] != list(range(process_end + 1)):
        raise RuntimeError("Manifest is not chronological and contiguous")
    return process_rows, score_frames, label


def load_truth(score_frames: list[int]) -> dict[int, list[dict]]:
    wanted = set(score_frames)
    truth: dict[int, list[dict]] = defaultdict(list)
    for row in read_csv(GT):
        frame = int(row["annotated_frame_index"])
        if frame in wanted and row["gt_valid"] == "True":
            truth[frame].append({
                "entity_id": row["participant_id"],
                "xyz": np.asarray([float(row["center_annotated_x_m"]),
                                   float(row["center_annotated_y_m"]),
                                   float(row["center_annotated_z_m"])])})
    if sum(map(len, truth.values())) != 5 * len(score_frames):
        raise RuntimeError("Unexpected PERSON GT count in score scope")
    return truth


def choose_recovery_candidate(components: list[np.ndarray], detection: dict,
                              person_boxes: list[np.ndarray], model,
                              du_px: float=0.0, dv_px: float=0.0
                              ) -> tuple[recovery.RecoveryCandidate | None, dict]:
    ranked = []
    for component in components:
        score, details = support.component_score(
            component, detection["bbox"], model.transform, model.K, model.D,
            du_px=du_px, dv_px=dv_px)
        if np.isfinite(score):
            ranked.append((float(score), -len(component), component, details))
    if not ranked:
        return None, {"candidate_components": 0}
    ranked.sort(key=lambda value: (value[0], value[1]))
    score, _, component, details = ranked[0]
    center = support.cluster_center(component)
    pixel, valid, _ = support.project_points(
        center[None], model.transform, model.K, model.D,
        du_px=du_px, dv_px=dv_px)
    inside = bool(valid[0]
                  and detection["bbox"][0] <= pixel[0, 0] <= detection["bbox"][2]
                  and detection["bbox"][1] <= pixel[0, 1] <= detection["bbox"][3])
    containing_people = 0 if not valid[0] else sum(
        box[0] <= pixel[0, 0] <= box[2] and box[1] <= pixel[0, 1] <= box[3]
        for box in person_boxes)
    span = np.ptp(component, axis=0)
    horizontal, vertical = float(max(span[0], span[1])), float(span[2])
    second_gap = None if len(ranked) < 2 else float(ranked[1][0] - score)
    candidate = recovery.RecoveryCandidate(
        xyz=center, point_count=len(component), component_score=score,
        projected_center_inside=inside,
        plausible_size=0.10 <= horizontal <= 1.10 and 0.10 <= vertical <= 2.20,
        neighbor_competition=containing_people > 1,
        ambiguous_component=second_gap is not None and second_gap < 0.05,
        signature=component_signature(
            component, detection["bbox"], model, du_px=du_px, dv_px=dv_px))
    return candidate, {
        **details, "candidate_components": len(ranked),
        "second_score_gap": second_gap, "horizontal_span_m": horizontal,
        "vertical_span_m": vertical, "containing_person_boxes": containing_people,
        "center_x": float(center[0]), "center_y": float(center[1]),
        "center_z": float(center[2]), "center_u": float(pixel[0, 0]),
        "center_v": float(pixel[0, 1]), "candidate_points": len(component)}


def match_frame(predictions: list[dict], truth: list[dict]
                ) -> tuple[list[dict], int, int]:
    if not predictions or not truth:
        return [], len(predictions), len(truth)
    pred = np.asarray([row["xyz"] for row in predictions], np.float64)
    gt = np.asarray([row["xyz"] for row in truth], np.float64)
    cost = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    pp, gg = linear_sum_assignment(cost)
    matches = []
    for p, g in zip(pp, gg, strict=True):
        if cost[p, g] <= GATE_M:
            delta = pred[p] - gt[g]
            matches.append({"prediction_index": int(p), "gt_index": int(g),
                            "entity_id": truth[int(g)]["entity_id"],
                            "error_xy_m": float(np.linalg.norm(delta[:2])),
                            "error_xyz_m": float(np.linalg.norm(delta))})
    return matches, len(predictions) - len(matches), len(truth) - len(matches)


def evaluate(policy: str, frames: list[int], predictions: dict[int, list[dict]],
             truth: dict[int, list[dict]]) -> tuple[dict, list[dict]]:
    tp = fp = fn = recovered_matches = recovered_unmatched = neighbor_transitions = 0
    xy, xyz, recovered_xy, recovered_xyz, rows = [], [], [], [], []
    track_entities: dict[str, str] = {}
    for frame in frames:
        pred, gt = predictions[frame], truth[frame]
        matches, local_fp, local_fn = match_frame(pred, gt)
        tp += len(matches); fp += local_fp; fn += local_fn
        matched_pred = {row["prediction_index"] for row in matches}
        for row in matches:
            item = pred[row["prediction_index"]]
            xy.append(row["error_xy_m"]); xyz.append(row["error_xyz_m"])
            if item["source"] == "RECOVERED_SUPPRESSED_COMPONENT":
                recovered_matches += 1
                recovered_xy.append(row["error_xy_m"])
                recovered_xyz.append(row["error_xyz_m"])
                previous = track_entities.get(item["track_id"])
                neighbor_transitions += int(previous is not None
                                            and previous != row["entity_id"])
                track_entities[item["track_id"]] = row["entity_id"]
            rows.append({"policy": policy, "frame": frame,
                         "track_id": item["track_id"],
                         "source": item["source"], **row})
        recovered_unmatched += sum(
            item["source"] == "RECOVERED_SUPPRESSED_COMPONENT"
            for index, item in enumerate(pred) if index not in matched_pred)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return ({"policy": policy, "frames": len(frames), "gt_count": 5 * len(frames),
             "prediction_count": sum(len(value) for value in predictions.values()),
             "tp": tp, "fp": fp, "fn": fn,
             "precision": precision, "recall": recall,
             "f1": 2 * precision * recall / max(precision + recall, 1e-12),
             "xy": distribution(xy), "xyz": distribution(xyz),
             "recovered_xy": distribution(recovered_xy),
             "recovered_xyz": distribution(recovered_xyz),
             "recovered_hungarian_matches": recovered_matches,
             "recovered_unmatched": recovered_unmatched,
             "neighbor_identity_transitions": neighbor_transitions,
             "assignment": "3D_HUNGARIAN_ONE_TO_ONE", "gate_m": GATE_M}, rows)


def render_contact_sheet(manifest: list[dict], audit_rows: list[dict],
                         match_rows: list[dict], path: Path,
                         policy: str | None=None,
                         selection: str="risk") -> None:
    errors = {(row["policy"], row["frame"], row["track_id"]): row["error_xy_m"]
              for row in match_rows}
    pool = [row for row in audit_rows if policy is None or row["policy"] == policy]
    if selection == "accepted":
        pool = [row for row in pool if row.get("accepted")]
    elif selection == "neighbor":
        pool = [row for row in pool if row.get("neighbor_competition")]
    priority = sorted(pool, key=lambda row: (
        row.get("accepted", False), row.get("neighbor_competition", False),
        errors.get((row["policy"], row["frame"], row["track_id"]), 9.0)),
        reverse=True)
    selected, seen = [], set()
    for row in priority:
        eligible = (selection in {"accepted", "neighbor"}
                    or row.get("accepted") or row.get("neighbor_competition"))
        if row["frame"] in seen or not eligible:
            continue
        selected.append(row); seen.add(row["frame"])
        if len(selected) == 16:
            break
    if not selected:
        return
    tiles = []
    for row in selected:
        image = cv2.imread(manifest[row["frame"]]["rgb_image_path"])
        if image is None:
            continue
        box = np.asarray([row[f"bbox_{key}"]
                          for key in ("x1", "y1", "x2", "y2")], int)
        color = (0, 180, 0) if row.get("accepted") else (0, 0, 220)
        cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color, 3)
        if row.get("center_u") is not None:
            cv2.drawMarker(image, (int(row["center_u"]), int(row["center_v"])),
                           color, cv2.MARKER_CROSS, 24, 3)
        label = f"f{row['frame']} {row['policy']} {row['reason']}"
        cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, color, 2, cv2.LINE_AA)
        tiles.append(cv2.resize(image, (480, 270)))
    if not tiles:
        return
    blank = np.full_like(tiles[0], 245)
    tiles += [blank] * ((4 - len(tiles) % 4) % 4)
    sheet = np.vstack([np.hstack(tiles[index:index + 4])
                       for index in range(0, len(tiles), 4)])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def run(frames: int | None, score_split: str) -> dict:
    manifest, score_frames, scope_label = load_manifest_scope(score_split, frames)
    score_set = set(score_frames)
    output = OUT / scope_label.lower()
    output.mkdir(parents=True, exist_ok=True)
    model = pipeline.OnlineRgbFrustumPipeline(
        device="0", identity=False, projection_audit=False,
        legacy_display_offset=False)
    tracker = pipeline.OnlineTracker(np.empty((0, 0), np.float32), ())
    estimator = NormalConsistencyEstimator()
    strict_manager = recovery.CausalSuppressedPointRecovery(False)
    frozen_thresholds: recovery.FrozenConsistencyThresholds | None = None
    managers = {
        "HISTORY_ONLY": recovery.CausalSuppressedPointRecovery(False),
        "HISTORY_GUARDED_HEIGHT_CORRECTION": strict_manager,
        "HEIGHT_CORRECTION_HISTORY_OR_3FRAME_SEED": (
            recovery.CausalSuppressedPointRecovery(True)),
        "HISTORY_OR_3FRAME_SEED": recovery.CausalSuppressedPointRecovery(True),
    }
    predictions = {policy: defaultdict(list) for policy in POLICIES}
    audits: list[dict] = []
    counts = Counter()
    occupancy_tree = cKDTree(model.occupancy)
    processed = 0
    try:
        frames_iter = pipeline.synchronized_bag_frames(pipeline.DEFAULT_BAG, 35.0)
        for frame_index, frame in enumerate(frames_iter):
            if frame_index >= len(manifest):
                break
            if frame_index == 979 and frozen_thresholds is None:
                frozen_thresholds = estimator.freeze()
                strict_manager.consistency_thresholds = frozen_thresholds
            processed += 1
            expected_stamp = int(manifest[frame_index]["raw_lidar_header_timestamp_ns"])
            if frame.lidar_timestamp_ns != expected_stamp:
                raise RuntimeError("Bag/manifest LiDAR timestamp mismatch")
            detections = model._detections(frame.image)
            transform = model.annotated_from_rslidar
            annotated = (frame.points_rslidar @ transform[:3, :3].T
                         + transform[:3, 3])
            pixels, valid, depth = model.project_physical(annotated)
            ground_height = annotated @ model.ground_normal + model.ground_d
            in_view = (valid & (depth > 0) & (pixels[:, 0] >= 0)
                       & (pixels[:, 0] < pipeline.IMAGE_SIZE[0])
                       & (pixels[:, 1] >= 0)
                       & (pixels[:, 1] < pipeline.IMAGE_SIZE[1]))
            candidate_indices = np.flatnonzero(
                in_view & (ground_height >= 0.03) & (ground_height <= 2.15))
            candidate_points = annotated[candidate_indices]
            keys = np.floor(candidate_points / model.voxel_size).astype(np.int32)
            temporal_keep = np.fromiter(
                (tuple(map(int, key)) not in model.static for key in keys),
                bool, len(keys))
            distance = occupancy_tree.query(candidate_points[:, :2], workers=-1)[0]
            # Legacy production code subtracts signed ground distance from Z.
            # That returns the ground-plane absolute Z, not height above ground.
            # Keep it as the baseline, and audit the dimensionally-correct signed
            # ground distance as an independent report-only policy.
            point_height = candidate_points[:, 2] - ground_height[candidate_indices]
            occupancy_keep = ~((distance <= 0.07)
                               & ((point_height <= 0.28)
                                  | (point_height >= 1.9)))
            corrected_height = ground_height[candidate_indices]
            corrected_occupancy_keep = ~(
                (distance <= 0.07)
                & ((corrected_height <= 0.28)
                   | (corrected_height >= 1.9)))
            boxes = [item["bbox"] for item in detections]
            person_boxes = [item["bbox"] for item in detections
                            if item["class"] == "PERSON"]
            if model.geometry_gpu:
                owners = model.geometry_gpu.owners(pixels[candidate_indices], boxes)
            else:
                owners = support.exclusive_point_owners(
                    pixels[candidate_indices], boxes, [None] * len(boxes))
            normal_owners, recovery_owners = owners.copy(), owners.copy()
            corrected_owners = owners.copy()
            normal_owners[~(temporal_keep & occupancy_keep)] = -1
            recovery_owners[~(temporal_keep & ~occupancy_keep)] = -1
            corrected_owners[~(temporal_keep & corrected_occupancy_keep)] = -1
            occupancy_deleted = temporal_keep & ~occupancy_keep
            reinstated_by_height_correction = (
                temporal_keep & corrected_occupancy_keep & ~occupancy_keep)
            for index, detection in enumerate(detections):
                lower, upper = ((0.08, 2.15) if detection["class"] == "PERSON"
                                else (0.03, 1.35))
                invalid_height = ((ground_height[candidate_indices] < lower)
                                  | (ground_height[candidate_indices] > upper))
                normal_owners[(normal_owners == index) & invalid_height] = -1
                corrected_owners[(corrected_owners == index)
                                 & invalid_height] = -1
                if detection["class"] != "PERSON":
                    recovery_owners[recovery_owners == index] = -1
                else:
                    recovery_owners[(recovery_owners == index)
                                    & invalid_height] = -1
            if model.geometry_gpu:
                normal_groups = model.geometry_gpu.components(
                    candidate_points, normal_owners, len(boxes))
                recovery_groups = model.geometry_gpu.components(
                    candidate_points, recovery_owners, len(boxes))
                corrected_groups = model.geometry_gpu.components(
                    candidate_points, corrected_owners, len(boxes))
            else:
                normal_groups = [support.adaptive_components(
                    candidate_points[normal_owners == index])
                    for index in range(len(boxes))]
                recovery_groups = [support.adaptive_components(
                    candidate_points[recovery_owners == index])
                    for index in range(len(boxes))]
                corrected_groups = [support.adaptive_components(
                    candidate_points[corrected_owners == index])
                    for index in range(len(boxes))]
            normal_xyz, normal_signatures = [], []
            corrected_xyz = []
            candidates, details = [], []
            strict_candidates, strict_details = [], []
            for index, (detection, components, suppressed,
                        corrected) in enumerate(zip(
                    detections, normal_groups, recovery_groups,
                    corrected_groups, strict=True)):
                component, _ = support.choose_component(
                    components, detection["bbox"], model.transform,
                    model.K, model.D, du_px=0.0)
                xyz = None if component is None else support.cluster_center(component)
                normal_xyz.append(xyz)
                normal_signatures.append(
                    None if component is None else component_signature(
                        component, detection["bbox"], model))
                corrected_component, _ = support.choose_component(
                    corrected, detection["bbox"], model.transform,
                    model.K, model.D, du_px=0.0)
                corrected_xyz.append(
                    None if corrected_component is None
                    else support.cluster_center(corrected_component))
                if detection["class"] == "PERSON" and xyz is None:
                    recovered, detail = choose_recovery_candidate(
                        suppressed, detection, person_boxes, model)
                else:
                    recovered, detail = None, {"candidate_components": 0}
                candidates.append(recovered); details.append(detail)
                deleted_count = int(np.count_nonzero(
                    (owners == index) & occupancy_deleted))
                reinstated_count = int(np.count_nonzero(
                    (owners == index) & reinstated_by_height_correction
                    & (corrected_owners == index)))
                if (detection["class"] == "PERSON" and xyz is None
                        and reinstated_count > 0):
                    strict_recovered, strict_detail = choose_recovery_candidate(
                        corrected, detection, person_boxes, model)
                else:
                    strict_recovered = None
                    strict_detail = {"candidate_components": 0}
                strict_detail = {
                    **strict_detail,
                    "candidate_source": "GROUND_HEIGHT_SEMANTICS_CORRECTED",
                    "occupancy_deleted_owned_points": deleted_count,
                    "height_correction_reinstated_owned_points": reinstated_count,
                }
                strict_candidates.append(strict_recovered)
                strict_details.append(strict_detail)
                detection["xyz"] = xyz
                detection["component_points"] = (0 if component is None
                                                   else len(component))
            tracked = tracker.update(detections)
            for index, item in enumerate(tracked):
                if item["class"] != "PERSON":
                    continue
                counts["person_boxes_all_processed"] += 1
                scored = frame_index in score_set
                counts["person_boxes_scored"] += int(scored)
                base = normal_xyz[index]
                corrected = corrected_xyz[index]
                if corrected is not None:
                    counts["corrected_components_all_processed"] += 1
                    counts["corrected_components_scored"] += int(scored)
                    if scored:
                        predictions["GROUND_HEIGHT_SEMANTICS_CORRECTED"][
                            frame_index].append({
                                "xyz": corrected.copy(),
                                "track_id": item["track_id"],
                                "source": "GROUND_HEIGHT_SEMANTICS_CORRECTED",
                                "detection": index})
                if base is not None:
                    counts["baseline_components_all_processed"] += 1
                    counts["baseline_components_scored"] += int(scored)
                    record = {"xyz": base.copy(), "track_id": item["track_id"],
                              "source": "NORMAL_COMPONENT", "detection": index}
                    if scored:
                        for policy in POLICIES:
                            if policy != "GROUND_HEIGHT_SEMANTICS_CORRECTED":
                                predictions[policy][frame_index].append(record.copy())
                    for manager in managers.values():
                        manager.observe_normal(item["track_id"], base, frame_index,
                                               normal_signatures[index])
                    if frame_index < 979:
                        estimator.observe(item["track_id"], base,
                                          normal_signatures[index], frame_index)
                    continue
                counts["baseline_misses_all_processed"] += 1
                counts["baseline_misses_scored"] += int(scored)
                for policy, manager in managers.items():
                    if policy in {"HISTORY_GUARDED_HEIGHT_CORRECTION",
                                  "HEIGHT_CORRECTION_HISTORY_OR_3FRAME_SEED"}:
                        policy_candidate = strict_candidates[index]
                        policy_detail = strict_details[index]
                    else:
                        policy_candidate = candidates[index]
                        policy_detail = details[index]
                    if (policy == "HISTORY_GUARDED_HEIGHT_CORRECTION"
                            and frozen_thresholds is None):
                        decision = {
                            "accepted": False,
                            "reason": "TRAIN_FIT_THRESHOLDS_NOT_FROZEN",
                            "motion_distance_m": None, "motion_gate_m": None,
                            "history_available": False, "seed_count": 0}
                    else:
                        decision = manager.consider(
                            item["track_id"], policy_candidate, frame_index)
                    row = {
                        "frame": frame_index, "policy": policy,
                        "detection": index, "track_id": item["track_id"],
                        "confidence": item["confidence"],
                        "bbox_x1": float(item["bbox"][0]),
                        "bbox_y1": float(item["bbox"][1]),
                        "bbox_x2": float(item["bbox"][2]),
                        "bbox_y2": float(item["bbox"][3]),
                        **decision, **policy_detail,
                        "neighbor_competition": (
                            False if policy_candidate is None
                            else policy_candidate.neighbor_competition)}
                    if scored:
                        audits.append(row)
                        counts[f"{policy}_attempts"] += int(
                            policy_candidate is not None)
                        counts[f"{policy}_accepted"] += int(decision["accepted"])
                        if not decision["accepted"]:
                            counts[f"{policy}_rejected_{decision['reason']}"] += 1
                    if scored and decision["accepted"]:
                        predictions[policy][frame_index].append({
                            "xyz": policy_candidate.xyz.copy(),
                            "track_id": item["track_id"],
                            "source": "RECOVERED_SUPPRESSED_COMPONENT",
                            "detection": index})
            if (frame_index + 1) % 100 == 0:
                print(f"recovery audit {frame_index + 1}/{len(manifest)}", flush=True)
    finally:
        model.calibration_audit.close()
    if processed != len(manifest):
        raise RuntimeError(f"Expected {len(manifest)} frames, received {processed}")

    # GT is intentionally loaded only after every prediction in scope exists.
    truth = load_truth(score_frames)
    metrics, match_rows = [], []
    for policy in POLICIES:
        value, rows = evaluate(policy, score_frames, predictions[policy], truth)
        value["person_box_component_coverage"] = (
            value["prediction_count"] / max(counts["person_boxes_scored"], 1))
        metrics.append(value); match_rows.extend(rows)
    baseline = metrics[0]
    baseline_entities = defaultdict(set)
    for row in match_rows:
        if row["policy"] == "BASELINE":
            baseline_entities[row["frame"]].add(row["entity_id"])
    for value in metrics[1:]:
        value["delta_component_coverage_pp"] = 100 * (
            value["person_box_component_coverage"]
            - baseline["person_box_component_coverage"])
        value["delta_f1"] = value["f1"] - baseline["f1"]
        value["delta_xy_rmse_m"] = (value["xy"]["rmse"]
                                            - baseline["xy"]["rmse"])
        value["delta_xy_p95_m"] = (value["xy"]["p95"]
                                           - baseline["xy"]["p95"])
        recovered_rows = [row for row in match_rows
                          if row["policy"] == value["policy"]
                          and row["source"] == "RECOVERED_SUPPRESSED_COMPONENT"]
        value["recovered_duplicate_existing_gt"] = sum(
            row["entity_id"] in baseline_entities[row["frame"]]
            for row in recovered_rows)
        value["net_incremental_tp"] = value["tp"] - baseline["tp"]
        value["net_incremental_fp"] = value["fp"] - baseline["fp"]
        value["net_incremental_fn"] = value["fn"] - baseline["fn"]
    by_policy = {value["policy"]: value for value in metrics}
    corrected_height = by_policy["GROUND_HEIGHT_SEMANTICS_CORRECTED"]
    history = by_policy["HISTORY_ONLY"]
    strict = by_policy["HISTORY_GUARDED_HEIGHT_CORRECTION"]
    corrected_seed = by_policy["HEIGHT_CORRECTION_HISTORY_OR_3FRAME_SEED"]

    def promotion_gate(value: dict) -> bool:
        return bool(
        value["person_box_component_coverage"]
        > baseline["person_box_component_coverage"]
        and value["precision"] >= baseline["precision"] - 0.005
        and value["xy"]["rmse"] <= baseline["xy"]["rmse"] + 0.02
        and value["xy"]["p95"] <= baseline["xy"]["p95"] + 0.05
        and value["net_incremental_tp"] > 0
        and value["net_incremental_fp"] == 0
        and value["recovered_duplicate_existing_gt"] == 0
        and value["recovered_unmatched"] == 0
        and value["neighbor_identity_transitions"] == 0)

    history_promotion = promotion_gate(history)
    strict_promotion = promotion_gate(strict)
    direct_height_promotion = promotion_gate(corrected_height)
    corrected_seed_promotion = promotion_gate(corrected_seed)
    summary = {
        "status": "PASS_REPORT_ONLY" if strict_promotion else "FAIL_REPORT_ONLY",
        "scope": f"SCENE01_{scope_label}_{len(score_frames)}_LIDAR",
        "processed_causal_prefix_frames": len(manifest),
        "physical_projection": "RAW_K_D_T_DU_DV_ZERO_NOT_PROVEN_FINAL",
        "recovery_source": "HEIGHT_SEMANTICS_CORRECTED_POINTS_AFTER_LEGACY_MISS",
        "frozen_trainfit_consistency_thresholds": (
            None if frozen_thresholds is None else asdict(frozen_thresholds)),
        "trainfit_normal_transition_samples": {
            key: len(value) for key, value in estimator.values.items()},
        "metrics": metrics, "audit_counts": dict(counts),
        "promotion_gate_history_only": history_promotion,
        "promotion_gate_direct_height_semantics_correction": direct_height_promotion,
        "promotion_gate_history_guarded_height_correction": strict_promotion,
        "promotion_gate_height_correction_causal_seed": corrected_seed_promotion,
        "runtime_modified": False, "candidate_applied_to_inference": False,
        "ground_truth_runtime_access": False,
        "ground_truth_offline_evaluation_only": True,
        "legacy_pixel_offset_used": False, "future_frames_used": False,
        "interpretation": (
            "Direct height correction is an ablation only. "
            "HISTORY_GUARDED_HEIGHT_CORRECTION applies TRAIN_FIT-frozen normal-component "
            "consistency only after a legacy occupancy-induced miss.")}
    (output / "suppressed_point_recovery_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "per_attempt_audit.csv", audits)
    write_csv(output / "matched_predictions.csv", match_rows)
    flat_metrics = [{**value, "xy": json.dumps(value["xy"]),
                     "xyz": json.dumps(value["xyz"]),
                     "recovered_xy": json.dumps(value["recovered_xy"]),
                     "recovered_xyz": json.dumps(value["recovered_xyz"])}
                    for value in metrics]
    write_csv(output / "policy_metrics.csv", flat_metrics)
    render_contact_sheet(manifest, audits, match_rows,
                         output / "HIGH_RISK_RECOVERY_CONTACT_SHEET.jpg")
    render_contact_sheet(manifest, audits, match_rows,
                         output / "HISTORY_ONLY_ACCEPTED_CONTACT_SHEET.jpg",
                         policy="HISTORY_ONLY", selection="accepted")
    render_contact_sheet(manifest, audits, match_rows,
                         output / "HISTORY_GUARDED_HEIGHT_CORRECTION_ACCEPTED.jpg",
                         policy="HISTORY_GUARDED_HEIGHT_CORRECTION", selection="accepted")
    render_contact_sheet(manifest, audits, match_rows,
                         output / "HEIGHT_CORRECTION_3FRAME_SEED_ACCEPTED.jpg",
                         policy="HEIGHT_CORRECTION_HISTORY_OR_3FRAME_SEED",
                         selection="accepted")
    render_contact_sheet(manifest, audits, match_rows,
                         output / "SEEDED_ACCEPTED_CONTACT_SHEET.jpg",
                         policy="HISTORY_OR_3FRAME_SEED", selection="accepted")
    render_contact_sheet(manifest, audits, match_rows,
                         output / "NEIGHBOR_COMPETITION_REJECTIONS.jpg",
                         policy="HISTORY_ONLY", selection="neighbor")

    lines = [
        "# Scene01 Causal Suppressed-Point Recovery Audit", "", "## Result", "",
        f"**{summary['status']}**. Production runtime remains unchanged; recovery is not enabled.", "",
        "| Policy | Box component coverage | Precision | Recall | F1 | XY RMSE | XY P95 | Net TP/FP | Duplicate existing GT | Neighbor transitions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for value in metrics:
        lines.append(
            f"| {value['policy']} | {value['person_box_component_coverage']:.2%} | "
            f"{value['precision']:.3f} | {value['recall']:.3f} | {value['f1']:.3f} | "
            f"{value['xy']['rmse']:.3f} m | {value['xy']['p95']:.3f} m | "
            f"{value.get('net_incremental_tp', 0):+d}/{value.get('net_incremental_fp', 0):+d} | "
            f"{value.get('recovered_duplicate_existing_gt', 0)} | "
            f"{value['neighbor_identity_transitions']} |")
    lines += [
        "", "## Frozen design", "",
        "Only PERSON detections with no normal component enter recovery. Projection is raw `K+D+T` with `du=dv=0`; exact original boxes and exclusive one-point/one-box ownership are retained. `GROUND_HEIGHT_SEMANTICS_CORRECTED` is a direct ablation: it uses signed ground-plane distance as height instead of subtracting that distance from absolute Z.", "",
        "Every recovery candidate must have at least three points, plausible 3D size, projected center inside the original box, component score <=1.15, no overlapping-person competition and no near-tied component. `HISTORY_ONLY` uses occupancy-deleted fragments plus the existing causal motion gate. `HISTORY_GUARDED_HEIGHT_CORRECTION` is triggered only after a legacy miss, reinstates points retained by the corrected height semantics, then requires TRAIN_FIT-frozen normal-component innovation, span, point-count and bbox-relative projection consistency. `HEIGHT_CORRECTION_HISTORY_OR_3FRAME_SEED` uses the corrected-height candidate with the existing motion gate or a causal three-frame seed; both seed policies are diagnostic only.", "",
        "## Evaluation discipline", "",
        "Predictions are produced without GT. Official cuboid centers are loaded only for offline 3D Hungarian evaluation with the existing 1.5 m gate. Promotion requires coverage improvement without material precision/RMSE/P95 regression, zero recovered FP and zero neighbor-identity transitions.", "",
        f"Promotion gates — direct height correction: **{'PASS' if direct_height_promotion else 'FAIL'}**; HISTORY_ONLY: **{'PASS' if history_promotion else 'FAIL'}**; history-guarded height correction: **{'PASS' if strict_promotion else 'FAIL'}**; corrected-height causal seed: **{'PASS' if corrected_seed_promotion else 'FAIL'}**. Even a PASS remains report-only until explicitly approved for runtime integration."]
    (output / "CAUSAL_SUPPRESSED_POINT_RECOVERY_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--score-split", choices=("TRAIN_FIT", "VALIDATION"),
                        default="VALIDATION")
    args = parser.parse_args()
    print(json.dumps(run(args.frames, args.score_split), ensure_ascii=False, indent=2))

