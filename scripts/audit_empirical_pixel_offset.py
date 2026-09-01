"""TRAIN_FIT-only 2-D pixel-offset search and frozen VALIDATION gate.

This is a Scene01 empirical registration experiment, not physical calibration.
All offset candidates are predicted before TRAIN_FIT GT is loaded. The winner
is frozen before any VALIDATION GT is loaded. TEST/EMBARGO files are never read.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from online_v4 import pipeline, support, suppression  # noqa: E402
import audit_fn_root_causes as fn_audit  # noqa: E402
import audit_occupancy_suppression as occupancy_audit  # noqa: E402
import audit_suppressed_point_recovery as recovery_audit  # noqa: E402

OUT = ROOT / "outputs" / "empirical_pixel_offset_audit"
PRIOR = Path(r"D:\navwareset_scene01_clean\outputs\projection_horizontal_offset_rootcause\audit_metrics.json")
PRIOR_LABELS = Path(r"D:\navwareset_scene01_clean\outputs\five_identity_specialists\pilot_50_per_identity\manual_identity_crop_labels.csv")
BASE_GRID = tuple((f"D{du}_V{dv}", float(du), float(dv))
                  for du in range(44, 57, 2) for dv in range(-4, 5, 2))
EXTENSION_GRID = tuple((f"D{du}_V{dv}", float(du), float(dv))
                       for du in range(58, 73, 2) for dv in range(-2, 7, 2))
VERTICAL_EXTENSION_GRID = tuple((f"D{du}_V{dv}", float(du), float(dv))
                                for du in range(60, 69, 2)
                                for dv in range(8, 15, 2))
FINE_VERTICAL_GRID = tuple((f"D{du}_V{dv}", float(du), float(dv))
                           for du in range(58, 69, 2)
                           for dv in range(15, 23))
HIGH_VERTICAL_GRID = tuple((f"D{du}_V{dv}", float(du), float(dv))
                           for du in range(58, 69, 2)
                           for dv in range(24, 41, 4))
EXPECTED_RAW_V1 = "1ae1709531b1be03c67c4bb524eb19c6a56c58395fbf1b81b1b0f3bf75f207f6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def read_metric_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    integer = {"frames", "gt_count", "prediction_count", "tp", "fp", "fn",
               "recovered_matches", "recovered_unmatched",
               "neighbor_identity_transitions"}
    numeric = {"du_px", "dv_px", "precision", "recall", "f1", "xy_rmse_m",
               "xy_p95_m", "xy_median_m"}
    for row in rows:
        for key in integer:
            row[key] = int(row[key])
        for key in numeric:
            row[key] = float(row[key])
    return rows


def flatten(scope: str, variant: str, du: float, dv: float, value: dict) -> dict:
    return {
        "scope": scope, "variant": variant, "du_px": du, "dv_px": dv,
        "frames": value["frames"], "gt_count": value["gt_count"],
        "prediction_count": value["prediction_count"], "tp": value["tp"],
        "fp": value["fp"], "fn": value["fn"],
        "precision": value["precision"], "recall": value["recall"],
        "f1": value["f1"], "xy_rmse_m": value["xy"]["rmse"],
        "xy_p95_m": value["xy"]["p95"], "xy_median_m": value["xy"]["median"],
        "recovered_matches": value["recovered_hungarian_matches"],
        "recovered_unmatched": value["recovered_unmatched"],
        "neighbor_identity_transitions": value["neighbor_identity_transitions"],
    }


def replay_variants(manifest: list[dict], variants: tuple[tuple[str, float, float], ...],
                    end_frame: int, capture_splits: tuple[str, ...]
                    ) -> tuple[dict[str, dict[int, list[dict]]], dict]:
    allowed = {"TRAIN_FIT", "TRAIN_CALIBRATION", "VALIDATION"}
    rows = [row for row in manifest
            if int(row["annotated_frame_index"]) <= end_frame and row["split"] in allowed]
    if any(row["split"] in {"TEST", "EMBARGO"} for row in rows):
        raise RuntimeError("TEST/EMBARGO access is forbidden")
    states = {name: fn_audit.FrozenState(True, False) for name, _, _ in variants}
    predictions = {name: defaultdict(list) for name, _, _ in variants}
    model = pipeline.OnlineRgbFrustumPipeline(
        device="0", identity=False, projection_audit=False,
        legacy_display_offset=False, suppressed_recovery=False)
    previous = -1
    try:
        for row in rows:
            frame = int(row["annotated_frame_index"])
            for _ in range(frame - previous - 1):
                for state in states.values():
                    state.tracker.update([])
            previous = frame
            image = cv2.imread(row["rgb_image_path"])
            if image is None:
                raise RuntimeError(f"Missing RGB frame: {row['rgb_image_path']}")
            points = occupancy_audit.read_binary_xyz_pcd(row["pcd_path"])
            detections = model._detections(image)
            raw_pixels, valid, depth = model.project_physical(points)
            height = suppression.ground_height(points, model.ground_normal, model.ground_d)
            for name, du, dv in variants:
                pixels = raw_pixels + np.asarray([du, dv], np.float64)
                result, _ = fn_audit.trace_frame(
                    model, detections, points, pixels, valid, depth, height,
                    du_px=du, dv_px=dv, capture_trace=False)
                pred, _ = states[name].step(result, frame)
                if row["split"] in capture_splits:
                    predictions[name][frame] = pred
            if (frame + 1) % 100 == 0:
                print(f"offset replay {frame}/{end_frame} variants={len(variants)}", flush=True)
    finally:
        model.calibration_audit.close()
    return predictions, {
        "processed_allowed_rows": len(rows), "test_access": 0, "embargo_access": 0,
        "raw_physical_projection_du_px": 0.0, "raw_physical_projection_dv_px": 0.0,
    }


def metric_rows(scope: str, frames: list[int], variants, predictions, truth
                ) -> tuple[list[dict], dict]:
    rows, values = [], {}
    for name, du, dv in variants:
        value, _ = recovery_audit.evaluate(name, frames, predictions[name], truth)
        rows.append(flatten(scope, name, du, dv, value)); values[name] = value
    return rows, values


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    if prior.get("test_used_for_model_or_threshold_selection") is not False:
        raise RuntimeError("Prior selector is not TEST-isolated")
    if sha256(PRIOR_LABELS) != prior["validation"]["source_sha256"]:
        raise RuntimeError("Frozen prior manual labels changed")
    manifest = fn_audit.read_csv(fn_audit.MANIFEST)
    frames = {split: [int(row["annotated_frame_index"]) for row in manifest
                      if row["split"] == split]
              for split in ("TRAIN_FIT", "VALIDATION")}

    search_receipt = {
        "base_grid_du_px": list(range(44, 57, 2)),
        "base_grid_dv_px": list(range(-4, 5, 2)),
        "extension_grid_du_px": list(range(58, 73, 2)),
        "extension_grid_dv_px": list(range(-2, 7, 2)),
        "vertical_extension_du_px": list(range(60, 69, 2)),
        "vertical_extension_dv_px": list(range(8, 15, 2)),
        "fine_vertical_du_px": list(range(58, 69, 2)),
        "fine_vertical_dv_px": list(range(15, 23)),
        "high_vertical_du_px": list(range(58, 69, 2)),
        "high_vertical_dv_px": list(range(24, 41, 4)),
        "candidate_count": (len(BASE_GRID) + len(EXTENSION_GRID)
                            + len(VERTICAL_EXTENSION_GRID)
                            + len(FINE_VERTICAL_GRID)
                            + len(HIGH_VERTICAL_GRID)),
        "prior_development_grid": str(PRIOR),
        "prior_development_grid_sha256": sha256(PRIOR),
        "gt_loaded": False, "validation_gt_loaded": False,
        "test_access": 0, "embargo_access": 0,
    }
    (OUT / "train_search_freeze_before_gt.json").write_text(
        json.dumps(search_receipt, indent=2), encoding="utf-8")
    base_path = OUT / "train_fit_offset_grid_metrics.csv"
    if base_path.exists() and len(read_metric_csv(base_path)) == len(BASE_GRID):
        base_rows = read_metric_csv(base_path)
        base_safety = {"resumed_from_frozen_predictions": True,
                       "test_access": 0, "embargo_access": 0}
    else:
        base_predictions, base_safety = replay_variants(
            manifest, BASE_GRID, max(frames["TRAIN_FIT"]), ("TRAIN_FIT",))
        train_truth = fn_audit.load_truth_after_predictions(frames["TRAIN_FIT"])
        base_rows, _ = metric_rows(
            "TRAIN_FIT_SEARCH", frames["TRAIN_FIT"], BASE_GRID,
            base_predictions, train_truth)
        write_csv(base_path, base_rows)

    extension_path = OUT / "train_fit_offset_extension_metrics.csv"
    if extension_path.exists() and len(read_metric_csv(extension_path)) == len(EXTENSION_GRID):
        extension_rows = read_metric_csv(extension_path)
        extension_safety = {"resumed_from_frozen_predictions": True,
                            "test_access": 0, "embargo_access": 0}
    else:
        extension_predictions, extension_safety = replay_variants(
            manifest, EXTENSION_GRID, max(frames["TRAIN_FIT"]), ("TRAIN_FIT",))
        train_truth = fn_audit.load_truth_after_predictions(frames["TRAIN_FIT"])
        extension_rows, _ = metric_rows(
            "TRAIN_FIT_SEARCH", frames["TRAIN_FIT"], EXTENSION_GRID,
            extension_predictions, train_truth)
        write_csv(extension_path, extension_rows)
    vertical_path = OUT / "train_fit_vertical_extension_metrics.csv"
    if (vertical_path.exists()
            and len(read_metric_csv(vertical_path)) == len(VERTICAL_EXTENSION_GRID)):
        vertical_rows = read_metric_csv(vertical_path)
        vertical_safety = {"resumed_from_frozen_predictions": True,
                           "test_access": 0, "embargo_access": 0}
    else:
        vertical_predictions, vertical_safety = replay_variants(
            manifest, VERTICAL_EXTENSION_GRID, max(frames["TRAIN_FIT"]),
            ("TRAIN_FIT",))
        train_truth = fn_audit.load_truth_after_predictions(frames["TRAIN_FIT"])
        vertical_rows, _ = metric_rows(
            "TRAIN_FIT_SEARCH", frames["TRAIN_FIT"], VERTICAL_EXTENSION_GRID,
            vertical_predictions, train_truth)
        write_csv(vertical_path, vertical_rows)
    fine_path = OUT / "train_fit_fine_vertical_metrics.csv"
    if fine_path.exists() and len(read_metric_csv(fine_path)) == len(FINE_VERTICAL_GRID):
        fine_rows = read_metric_csv(fine_path)
        fine_safety = {"resumed_from_frozen_predictions": True,
                       "test_access": 0, "embargo_access": 0}
    else:
        fine_predictions, fine_safety = replay_variants(
            manifest, FINE_VERTICAL_GRID, max(frames["TRAIN_FIT"]),
            ("TRAIN_FIT",))
        train_truth = fn_audit.load_truth_after_predictions(frames["TRAIN_FIT"])
        fine_rows, _ = metric_rows(
            "TRAIN_FIT_SEARCH", frames["TRAIN_FIT"], FINE_VERTICAL_GRID,
            fine_predictions, train_truth)
        write_csv(fine_path, fine_rows)
    high_path = OUT / "train_fit_high_vertical_metrics.csv"
    if high_path.exists() and len(read_metric_csv(high_path)) == len(HIGH_VERTICAL_GRID):
        high_rows = read_metric_csv(high_path)
        high_safety = {"resumed_from_frozen_predictions": True,
                       "test_access": 0, "embargo_access": 0}
    else:
        high_predictions, high_safety = replay_variants(
            manifest, HIGH_VERTICAL_GRID, max(frames["TRAIN_FIT"]),
            ("TRAIN_FIT",))
        train_truth = fn_audit.load_truth_after_predictions(frames["TRAIN_FIT"])
        high_rows, _ = metric_rows(
            "TRAIN_FIT_SEARCH", frames["TRAIN_FIT"], HIGH_VERTICAL_GRID,
            high_predictions, train_truth)
        write_csv(high_path, high_rows)
    search_rows = (base_rows + extension_rows + vertical_rows + fine_rows
                   + high_rows)
    write_csv(OUT / "train_fit_offset_search_metrics.csv", search_rows)
    best = max(search_rows, key=lambda row: (
        row["f1"], row["recall"], row["precision"], -row["xy_rmse_m"],
        -row["xy_p95_m"], -abs(row["du_px"] - 48.0), -abs(row["dv_px"])))
    candidate = ("TRAIN_FIT_SELECTED", float(best["du_px"]), float(best["dv_px"]))
    validation_variants = (
        ("RAW_0_0", 0.0, 0.0), ("LEGACY_48_0", 48.0, 0.0), candidate)
    freeze = {
        "selection_scope": "TRAIN_FIT_ONLY",
        "selection_metric": "max F1, recall, precision; then min RMSE/P95 and legacy distance",
        "selected_du_px": candidate[1], "selected_dv_px": candidate[2],
        "train_fit_metrics": best, "validation_gt_loaded": False,
        "candidate_semantics": "SCENE01_EMPIRICAL_PIXEL_REGISTRATION_NOT_PHYSICAL_CALIBRATION",
        "test_access": 0, "embargo_access": 0,
    }
    (OUT / "candidate_freeze_before_validation_gt.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8")

    validation_predictions, validation_safety = replay_variants(
        manifest, validation_variants, max(frames["VALIDATION"]),
        ("TRAIN_FIT", "VALIDATION"))
    validation_frame_set = set(frames["VALIDATION"])
    raw_validation_stream = {
        frame: values for frame, values in validation_predictions["RAW_0_0"].items()
        if frame in validation_frame_set}
    raw_fingerprint = fn_audit.prediction_fingerprint(raw_validation_stream)
    if raw_fingerprint != EXPECTED_RAW_V1:
        raise RuntimeError(f"Raw baseline fingerprint changed: {raw_fingerprint}")

    truth = {scope: fn_audit.load_truth_after_predictions(scope_frames)
             for scope, scope_frames in frames.items()}
    rows, metrics = [], {}
    for scope in ("TRAIN_FIT", "VALIDATION"):
        local, values = metric_rows(
            scope, frames[scope], validation_variants, validation_predictions, truth[scope])
        rows.extend(local)
        metrics.update({f"{scope}::{key}": value for key, value in values.items()})
    write_csv(OUT / "offset_variant_metrics.csv", rows)
    lookup = {(row["scope"], row["variant"]): row for row in rows}
    raw = lookup[("VALIDATION", "RAW_0_0")]
    legacy = lookup[("VALIDATION", "LEGACY_48_0")]
    proposed = lookup[("VALIDATION", "TRAIN_FIT_SELECTED")]
    gates = {
        "candidate_f1_not_below_legacy_by_0p002": proposed["f1"] >= legacy["f1"] - 0.002,
        "candidate_precision_not_below_legacy_by_0p002": proposed["precision"] >= legacy["precision"] - 0.002,
        "candidate_recall_not_below_legacy_by_0p002": proposed["recall"] >= legacy["recall"] - 0.002,
        "candidate_xy_rmse_not_above_legacy_by_0p01m": proposed["xy_rmse_m"] <= legacy["xy_rmse_m"] + 0.01,
        "candidate_xy_p95_not_above_legacy_by_0p02m": proposed["xy_p95_m"] <= legacy["xy_p95_m"] + 0.02,
        "candidate_f1_above_raw": proposed["f1"] > raw["f1"],
        "candidate_xy_rmse_below_raw": proposed["xy_rmse_m"] < raw["xy_rmse_m"],
        "raw_baseline_fingerprint_exact": raw_fingerprint == EXPECTED_RAW_V1,
        "test_embargo_zero": base_safety["test_access"] == 0
            and extension_safety["test_access"] == 0
            and vertical_safety["test_access"] == 0
            and fine_safety["test_access"] == 0
            and high_safety["test_access"] == 0
            and validation_safety["test_access"] == 0,
    }
    passed = all(gates.values())
    selected = {"du_px": candidate[1], "dv_px": candidate[2]} if passed else {
        "du_px": 48.0, "dv_px": 0.0}
    production_applied = bool(
        support.REGISTRATION_CONFIG.get("empirical_inference_enabled")
        and float(support.REGISTRATION_CONFIG.get("empirical_inference_du_px", 0.0))
            == selected["du_px"]
        and float(support.REGISTRATION_CONFIG.get("empirical_inference_dv_px", 0.0))
            == selected["dv_px"])
    summary = {
        "status": "PASS" if passed else "FAIL",
        "train_fit_selected_candidate": {"du_px": candidate[1], "dv_px": candidate[2]},
        "selected_empirical_offset": selected,
        "candidate_applied_to_online": production_applied,
        "selection_source": "TRAIN_FIT_LOCAL_2D_GRID; VALIDATION_GATE_ONLY",
        "gates": gates, "metrics": metrics,
        "grid_safety": {"base": base_safety, "extension": extension_safety,
                        "vertical_extension": vertical_safety,
                        "fine_vertical": fine_safety,
                        "high_vertical": high_safety},
        "validation_safety": validation_safety,
        "raw_baseline_fingerprint": raw_fingerprint,
        "test_access": 0, "embargo_access": 0,
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    validation_rows = [row for row in rows if row["scope"] == "VALIDATION"]
    table = "\n".join(
        f"| {row['variant']} | {row['du_px']:+.0f} | {row['dv_px']:+.0f} | "
        f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
        f"{row['xy_rmse_m']:.3f} | {row['xy_p95_m']:.3f} |"
        for row in validation_rows)
    report = f"""# Scene01 Empirical Pixel-Offset Audit

Status: **{summary['status']}**

This is Scene01 empirical RGB–LiDAR pixel registration, not a physical
calibration update. A staged local grid searched `du=44…72 px` and
`dv=-4…+40 px` using TRAIN_FIT only, with one-pixel refinement in the rising
vertical region. The winner was frozen before VALIDATION GT was loaded.

| Variant | du px | dv px | Precision | Recall | F1 | XY RMSE m | XY P95 m |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

TRAIN_FIT selected **({candidate[1]:+.0f},{candidate[2]:+.0f}) px**.
Final online selection after the frozen VALIDATION gate:
**({selected['du_px']:+.0f},{selected['dv_px']:+.0f}) px**.

TEST/EMBARGO access remained zero. Raw `K+D+T, du=dv=0` remains an audit
baseline; the selected correction is Scene01-specific empirical registration.
Production integration active: **{production_applied}**.
"""
    (OUT / "EMPIRICAL_PIXEL_OFFSET_AUDIT.md").write_text(report, encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"],
                      "train_candidate": result["train_fit_selected_candidate"],
                      "selected": result["selected_empirical_offset"]}, indent=2))

