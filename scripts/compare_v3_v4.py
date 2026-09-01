"""Post-run V3/V4 detector and 3D-pipeline comparison.

Predictions are loaded and fingerprinted before official 3D GT is read. There
is no independent 2D PERSON GT in this project, so detector P/R/F1 are not
invented; detector differences are box-level A/B statistics only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from online_v4 import pipeline, runtime, support
import evaluate_runtime_logs as evaluator

IOU_GATE = 0.50
DEPTH_BINS = ((0.0, 3.0, '0-3m'), (3.0, 6.0, '3-6m'),
              (6.0, 9.0, '6-9m'), (9.0, float('inf'), '>9m'))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict[int, dict]:
    result = {}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('processed'):
                result[int(row['rgb_frame_id'])] = row
    return result


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    xy1 = np.maximum(left[:2], right[:2])
    xy2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, float(xy2[0] - xy1[0])) * max(0.0, float(xy2[1] - xy1[1]))
    la = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    ra = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    return intersection / max(la + ra - intersection, 1e-9)


def match_boxes(left: list[dict], right: list[dict]) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if not left or not right:
        return [], list(range(len(left))), list(range(len(right)))
    iou = np.asarray([[box_iou(np.asarray(a['bbox']), np.asarray(b['bbox']))
                       for b in right] for a in left], np.float64)
    rr, cc = linear_sum_assignment(1.0 - iou)
    matched = [(int(a), int(b), float(iou[a, b])) for a, b in zip(rr, cc, strict=True)
               if iou[a, b] >= IOU_GATE]
    li = {item[0] for item in matched}
    ri = {item[1] for item in matched}
    return matched, [i for i in range(len(left)) if i not in li], [i for i in range(len(right)) if i not in ri]


def region(box: list[float]) -> str:
    center = 0.5 * (box[0] + box[2])
    return 'LEFT' if center < 1280 / 3 else ('RIGHT' if center >= 2 * 1280 / 3 else 'CENTER')


def size_bin(box: list[float]) -> str:
    height = box[3] - box[1]
    return '<50px' if height < 50 else ('50-100px' if height < 100 else
                                       ('100-200px' if height < 200 else '>200px'))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def descriptive(values: list[float]) -> dict:
    data = np.asarray(values, np.float64)
    return {'count': int(len(data)), 'mean': float(data.mean()),
            'median': float(np.median(data)), 'p05': float(np.percentile(data, 5)),
            'p95': float(np.percentile(data, 95)), 'min': float(data.min()),
            'max': float(data.max())}


def prediction_matches(pred: list[np.ndarray], gt: list[np.ndarray]) -> tuple[set[int], set[int]]:
    if not pred:
        return set(), set()
    cost = np.linalg.norm(np.asarray(pred)[:, None, :] - np.asarray(gt)[None, :, :], axis=2)
    pp, gg = linear_sum_assignment(cost)
    pairs = [(int(p), int(g)) for p, g in zip(pp, gg, strict=True)
             if cost[p, g] <= evaluator.GATE_M]
    return {p for p, _ in pairs}, {g for _, g in pairs}


def depth_label(xyz: np.ndarray) -> str:
    value = float(np.linalg.norm(xyz[:2]))
    return next(name for lower, upper, name in DEPTH_BINS if lower <= value < upper)


def score_by_depth(pred: dict[int, list[np.ndarray]], gt: dict[int, list[np.ndarray]]) -> dict:
    totals, hits = Counter(), Counter()
    for frame, truth in gt.items():
        _, matched_gt = prediction_matches(pred.get(frame, []), truth)
        for index, xyz in enumerate(truth):
            label = depth_label(xyz)
            totals[label] += 1
            hits[label] += index in matched_gt
    return {name: {'gt': totals[name], 'tp': hits[name],
                   'recall': hits[name] / max(totals[name], 1)}
            for *_, name in DEPTH_BINS}


def lidar_rows(log: dict[int, dict]) -> dict[int, dict]:
    result = {}
    for row in log.values():
        if not row.get('new_lidar_update'):
            continue
        detail = row.get('lidar_update_detail') or {}
        frame = detail.get('lidar_frame_id', row.get('latest_lidar_frame_id'))
        if frame is not None:
            result[int(frame)] = row
    return result


def classify_v4_fn(gt_xyz: np.ndarray, row: dict) -> str:
    detail = row.get('lidar_update_detail') or {}
    observations = detail.get('geometry_observations', [])
    transform, k, d = support.projection_assets()
    pixel, valid, _ = support.project_points(
        gt_xyz[None], transform, k, d,
        du_px=support.EMPIRICAL_INFERENCE_REGISTRATION['du_px'],
        dv_px=support.EMPIRICAL_INFERENCE_REGISTRATION['dv_px'])
    covering = []
    if valid[0]:
        for item in observations:
            box = item['bbox']
            if box[0] <= pixel[0, 0] <= box[2] and box[1] <= pixel[0, 1] <= box[3]:
                covering.append(item)
    if not covering:
        return 'A_COCO_BOX_MISS_OR_PROJECTION_PROXY_MISMATCH'
    measured = [np.asarray(item['xyz'], np.float64) for item in covering
                if item.get('xyz') is not None]
    if not measured:
        return 'B_BOX_PRESENT_NO_LIDAR_COMPONENT'
    if min(float(np.linalg.norm(value - gt_xyz)) for value in measured) > evaluator.GATE_M:
        return 'C_COMPONENT_XYZ_OUTSIDE_1P5M_GATE'
    return 'D_SHORT_TRACKER_OR_STATE_ASSOCIATION_FAILURE'


def contact_sheet(selected: list[int], v3: dict[int, dict], v4: dict[int, dict],
                  output: Path, title: str) -> None:
    selected = sorted(set(selected))[:12]
    images = {}
    if selected:
        for event in runtime.stream_sensor_events(pipeline.DEFAULT_BAG, max(selected) + 1, 0):
            if event.topic == pipeline.IMAGE_TOPIC and event.index in selected:
                images[event.index] = pipeline.decode_image(event.message)
    tiles = []
    for frame in selected:
        image = images.get(frame)
        if image is None:
            continue
        panels = []
        for label, row, color in (('V3 Scene01 detector', v3[frame], (185, 95, 20)),
                                  ('V4 official COCO', v4[frame], (25, 150, 120))):
            panel = cv2.resize(image, (480, 270))
            scale = np.asarray([480 / 1280, 270 / 720, 480 / 1280, 270 / 720])
            for item in row.get('person_detections', []):
                box = np.round(np.asarray(item['bbox']) * scale).astype(int)
                cv2.rectangle(panel, tuple(box[:2]), tuple(box[2:]), color, 2)
                cv2.putText(panel, f"{item['confidence']:.2f}",
                            (box[0], max(36, box[1] - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            cv2.rectangle(panel, (0, 0), (480, 30), (15, 20, 25), -1)
            cv2.putText(panel, f'{label} | RGB frame {frame}', (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.53, (245, 245, 245), 1, cv2.LINE_AA)
            panels.append(panel)
        tile = np.hstack(panels)
        cv2.putText(tile, title, (12, 260), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (250, 250, 250), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        tiles = [np.full((270, 960, 3), 245, np.uint8)]
    columns = 2
    lines = []
    for start in range(0, len(tiles), columns):
        line = tiles[start:start + columns]
        line.extend([np.full_like(tiles[0], 245)] * (columns - len(line)))
        lines.append(np.hstack(line))
    cv2.imwrite(str(output), np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--v3-log', required=True)
    parser.add_argument('--v4-log', required=True)
    parser.add_argument('--v3-summary', required=True)
    parser.add_argument('--v4-summary', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    v3_path, v4_path = Path(args.v3_log), Path(args.v4_log)
    v3, v4 = load(v3_path), load(v4_path)
    if sorted(v3) != sorted(v4) or len(v4) != 7490:
        raise RuntimeError('V3/V4 comparison requires identical complete 7,490-frame logs')
    frozen = {'v3_log_sha256': digest(v3_path), 'v4_log_sha256': digest(v4_path),
              'prediction_frames': len(v4), 'gt_loaded_after_predictions': True}

    frame_rows, cases, risk = [], [], Counter()
    for frame in sorted(v3):
        left, right = v3[frame]['person_detections'], v4[frame]['person_detections']
        matched, only_left, only_right = match_boxes(left, right)
        frame_rows.append({
            'frame_id': frame, 'v3_person_count': len(left), 'v4_person_count': len(right),
            'matched_boxes_iou_ge_0p5': len(matched), 'v3_only_boxes': len(only_left),
            'v4_only_boxes': len(only_right),
            'v3_mean_confidence': np.mean([x['confidence'] for x in left]) if left else None,
            'v4_mean_confidence': np.mean([x['confidence'] for x in right]) if right else None,
        })
        risk[frame] = len(only_left) + len(only_right)
        for source, indices, values in (('V3_ONLY', only_left, left), ('V4_ONLY', only_right, right)):
            for index in indices:
                item = values[index]
                cases.append({'frame_id': frame, 'case_type': source,
                              'confidence': item['confidence'], 'bbox': json.dumps(item['bbox']),
                              'bbox_height_px': item['bbox'][3] - item['bbox'][1],
                              'size_bin': size_bin(item['bbox']), 'image_region': region(item['bbox'])})
        for li, ri, overlap in matched:
            lbox, rbox = left[li]['bbox'], right[ri]['bbox']
            if overlap < 0.80:
                cases.append({'frame_id': frame, 'case_type': 'BOTH_BOX_DIFF_IOU_LT_0P8',
                              'confidence': right[ri]['confidence'], 'bbox': json.dumps(rbox),
                              'bbox_height_px': rbox[3] - rbox[1], 'size_bin': size_bin(rbox),
                              'image_region': region(rbox), 'matched_iou': overlap})
    write_csv(out / 'v3_v4_scene01_detection_comparison.csv', frame_rows)
    write_csv(out / 'V3_V4_PERSON_DETECTION_DIFF.csv', frame_rows)
    write_csv(out / 'detector_diff_cases.csv', cases)

    validation_frames = evaluator.split_frames('VALIDATION')
    v3_pred = evaluator.predictions(v3_path, validation_frames)
    v4_pred = evaluator.predictions(v4_path, validation_frames)
    # Official GT is first read here, after both prediction logs were loaded and fingerprinted.
    truth = evaluator.truth(validation_frames)
    v3_metrics, _ = evaluator.score('V3', 'VALIDATION', validation_frames, v3_pred, truth)
    v4_metrics, _ = evaluator.score('V4_COCO', 'VALIDATION', validation_frames, v4_pred, truth)
    v3_depth, v4_depth = score_by_depth(v3_pred, truth), score_by_depth(v4_pred, truth)
    v3_lidar, v4_lidar = lidar_rows(v3), lidar_rows(v4)
    fn_breakdown, additional = Counter(), []
    for frame in validation_frames:
        _, v3_hit = prediction_matches(v3_pred[frame], truth[frame])
        _, v4_hit = prediction_matches(v4_pred[frame], truth[frame])
        for index, xyz in enumerate(truth[frame]):
            if index in v4_hit:
                continue
            category = classify_v4_fn(xyz, v4_lidar.get(frame, {}))
            fn_breakdown[category] += 1
            if index in v3_hit:
                additional.append({'frame_id': frame, 'gt_index': index,
                                   'depth_bin': depth_label(xyz), 'category': category})

    v3_summary = json.loads(Path(args.v3_summary).read_text(encoding='utf-8'))
    v4_summary = json.loads(Path(args.v4_summary).read_text(encoding='utf-8'))
    v3_count = sum(len(row['person_detections']) for row in v3.values())
    v4_count = sum(len(row['person_detections']) for row in v4.values())
    detection = {
        'independent_2d_gt_available': False,
        'detector_precision_recall_f1': None,
        'v3_person_detections': v3_count,
        'v4_person_detections': v4_count,
        'delta_v4_minus_v3': v4_count - v3_count,
        'v3_zero_detection_frames': sum(not row['person_detections'] for row in v3.values()),
        'v4_zero_detection_frames': sum(not row['person_detections'] for row in v4.values()),
        'v3_only_box_cases': sum(row['case_type'] == 'V3_ONLY' for row in cases),
        'v4_only_box_cases': sum(row['case_type'] == 'V4_ONLY' for row in cases),
        'confidence_distribution': {
            'v3': descriptive([x['confidence'] for row in v3.values() for x in row['person_detections']]),
            'v4': descriptive([x['confidence'] for row in v4.values() for x in row['person_detections']]),
        },
        'bbox_height_bins': {
            'v3': dict(Counter(size_bin(x['bbox']) for row in v3.values() for x in row['person_detections'])),
            'v4': dict(Counter(size_bin(x['bbox']) for row in v4.values() for x in row['person_detections'])),
        },
    }
    metrics = {'prediction_freeze': frozen, 'detection_only': detection,
               'v3': v3_metrics, 'v4': v4_metrics,
               'near_person_recall': {'v3': v3_depth, 'v4': v4_depth},
               'v4_fn_root_cause_proxy': dict(fn_breakdown),
               'v4_additional_fn_vs_v3': {'count': len(additional),
                                          'by_depth': dict(Counter(x['depth_bin'] for x in additional)),
                                          'by_category': dict(Counter(x['category'] for x in additional))},
               'caveat': 'A/B/C/D is a post-hoc projection proxy, not independent 2D GT.'}
    (out / 'COCO_SCENE01_VALIDATION_METRICS.json').write_text(
        json.dumps(metrics, indent=2) + '\n', encoding='utf-8')

    def component(summary: dict) -> float:
        return summary['projection']['component_availability_per_detection']

    comparison = [
        ('RGB PERSON detection count', v3_count, v4_count),
        ('Component availability', component(v3_summary), component(v4_summary)),
        ('XYZ availability over all RGB detections', v3_summary['state_and_3d']['person_xyz_valid_rows'] / max(v3_count, 1), v4_summary['state_and_3d']['person_xyz_valid_rows'] / max(v4_count, 1)),
        ('VALIDATION Precision', v3_metrics['precision'], v4_metrics['precision']),
        ('VALIDATION Recall', v3_metrics['recall'], v4_metrics['recall']),
        ('VALIDATION F1', v3_metrics['f1'], v4_metrics['f1']),
        ('VALIDATION XY RMSE m', v3_metrics['xy']['rmse'], v4_metrics['xy']['rmse']),
        ('VALIDATION XY P95 m', v3_metrics['xy']['p95'], v4_metrics['xy']['p95']),
        ('YOLO P95 ms', v3_summary['runtime_ms']['yolo']['p95'], v4_summary['runtime_ms']['yolo']['p95']),
        ('LiDAR P95 ms', v3_summary['runtime_ms']['lidar_total']['p95'], v4_summary['runtime_ms']['lidar_total']['p95']),
        ('Online latency P95 ms', v3_summary['runtime_ms']['online_latency']['p95'], v4_summary['runtime_ms']['online_latency']['p95']),
        ('Uncapped FPS', v3_summary['rates_hz']['wall_output'], v4_summary['rates_hz']['wall_output']),
    ]
    write_csv(out / 'V3_V4_SCENE01_COMPARISON.csv', [
        {'Metric': name, 'V3': left, 'V4': right,
         'Delta_V4_minus_V3': None if left is None or right is None else right - left}
        for name, left, right in comparison])
    ranked = [frame for frame, score in risk.most_common() if score > 0]
    contact_sheet(ranked, v3, v4, out / 'DETECTOR_DIFF_CONTACT_SHEET.jpg',
                  'Box-level detector difference; IoU gate 0.50')
    near_frames = [row['frame_id'] for row in additional
                   if row['depth_bin'] in ('0-3m', '3-6m')]
    contact_sheet(near_frames, v3, v4, out / 'COCO_NEAR_PERSON_MISS_CONTACT_SHEET.jpg',
                  'V4 additional near-person FN proxy; official 3D GT used post-run')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
