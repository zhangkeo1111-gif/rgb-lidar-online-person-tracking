"""Frozen Scene01 A/B audit for official COCO YOLO11s and YOLO26s."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import compare_v3_v4 as common
import evaluate_runtime_logs as evaluator
from online_v4 import pipeline, runtime


def read_images(frames: set[int]) -> dict[int, np.ndarray]:
    images = {}
    if frames:
        for event in runtime.stream_sensor_events(pipeline.DEFAULT_BAG, max(frames) + 1, 0):
            if event.topic == pipeline.IMAGE_TOPIC and event.index in frames:
                images[event.index] = pipeline.decode_image(event.message)
    return images


def contact_sheet(frames: list[int], left: dict[int, dict], right: dict[int, dict],
                  images: dict[int, np.ndarray], output: Path, title: str,
                  notes: dict[int, str] | None = None) -> None:
    tiles = []
    for frame in list(dict.fromkeys(frames))[:12]:
        image = images.get(frame)
        if image is None:
            continue
        panels = []
        for label, row, color in (
                ('YOLO11s COCO', left[frame], (210, 120, 20)),
                ('YOLO26s COCO', right[frame], (20, 165, 110))):
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.53, (245, 245, 245), 1,
                        cv2.LINE_AA)
            panels.append(panel)
        tile = np.hstack(panels)
        note = '' if notes is None else notes.get(frame, '')
        cv2.putText(tile, f'{title} {note}'.strip(), (12, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (250, 250, 250), 1,
                    cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        tiles = [np.full((270, 960, 3), 245, np.uint8)]
    rows = []
    for start in range(0, len(tiles), 2):
        row = tiles[start:start + 2]
        row.extend([np.full_like(tiles[0], 245)] * (2 - len(row)))
        rows.append(np.hstack(row))
    cv2.imwrite(str(output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def metric_rows(y11: dict, y26: dict, y11_summary: dict, y26_summary: dict,
                y11_rt: dict, y26_rt: dict, y11_boxes: int, y26_boxes: int,
                depth11: dict, depth26: dict) -> list[dict]:
    values = [
        ('Full-sequence RGB PERSON detection rows', y11_boxes, y26_boxes, 'count'),
        ('Full-sequence propagated PERSON rows', y11_summary['counts']['person_rows'],
         y26_summary['counts']['person_rows'], 'count'),
        ('VALIDATION PERSON prediction rows', y11['prediction_count'], y26['prediction_count'], 'count'),
        ('VALIDATION TP', y11['tp'], y26['tp'], 'count'),
        ('VALIDATION FP', y11['fp'], y26['fp'], 'count'),
        ('VALIDATION FN', y11['fn'], y26['fn'], 'count'),
        ('VALIDATION Precision', y11['precision'], y26['precision'], 'ratio'),
        ('VALIDATION Recall', y11['recall'], y26['recall'], 'ratio'),
        ('VALIDATION F1', y11['f1'], y26['f1'], 'ratio'),
        ('Recall 0-3 m', depth11['0-3m']['recall'], depth26['0-3m']['recall'], 'ratio'),
        ('Recall 3-6 m', depth11['3-6m']['recall'], depth26['3-6m']['recall'], 'ratio'),
        ('Recall 6-9 m', depth11['6-9m']['recall'], depth26['6-9m']['recall'], 'ratio'),
        ('Recall >9 m', depth11['>9m']['recall'], depth26['>9m']['recall'], 'ratio'),
        ('Component availability', y11_summary['projection']['component_availability_per_detection'],
         y26_summary['projection']['component_availability_per_detection'], 'ratio'),
        ('LiDAR XYZ availability', y11_summary['projection']['measurement_availability_per_detection'],
         y26_summary['projection']['measurement_availability_per_detection'], 'ratio'),
        ('VALIDATION XY RMSE', y11['xy']['rmse'], y26['xy']['rmse'], 'm'),
        ('VALIDATION XY P95', y11['xy']['p95'], y26['xy']['p95'], 'm'),
        ('Uncapped YOLO mean', y11_summary['runtime_ms']['yolo']['mean'],
         y26_summary['runtime_ms']['yolo']['mean'], 'ms'),
        ('Uncapped YOLO P95', y11_summary['runtime_ms']['yolo']['p95'],
         y26_summary['runtime_ms']['yolo']['p95'], 'ms'),
        ('Uncapped YOLO P99', y11_summary['runtime_ms']['yolo']['p99'],
         y26_summary['runtime_ms']['yolo']['p99'], 'ms'),
        ('Uncapped YOLO max', y11_summary['runtime_ms']['yolo']['max'],
         y26_summary['runtime_ms']['yolo']['max'], 'ms'),
        ('Uncapped LiDAR P95', y11_summary['runtime_ms']['lidar_total']['p95'],
         y26_summary['runtime_ms']['lidar_total']['p95'], 'ms'),
        ('Sensor-paced online latency P95', y11_rt['runtime_ms']['online_latency']['p95'],
         y26_rt['runtime_ms']['online_latency']['p95'], 'ms'),
        ('Uncapped FPS', y11_summary['rates_hz']['wall_output'],
         y26_summary['rates_hz']['wall_output'], 'fps'),
        ('Sensor-paced FPS', y11_rt['rates_hz']['wall_output'],
         y26_rt['rates_hz']['wall_output'], 'fps'),
        ('Sensor-paced backlog drops', y11_rt['counts'].get('rgb_dropped_due_to_backlog', 0),
         y26_rt['counts'].get('rgb_dropped_due_to_backlog', 0), 'count'),
    ]
    return [{'Metric': name, 'Unit': unit, 'YOLO11s_COCO': left,
             'YOLO26s_COCO': right, 'Delta_YOLO26s_minus_YOLO11s': right - left}
            for name, left, right, unit in values]


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument('--yolo11-log', required=True)
    cli.add_argument('--yolo26-log', required=True)
    cli.add_argument('--yolo11-uncapped', required=True)
    cli.add_argument('--yolo26-uncapped', required=True)
    cli.add_argument('--yolo11-realtime', required=True)
    cli.add_argument('--yolo26-realtime', required=True)
    cli.add_argument('--output-dir', required=True)
    args = cli.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p11, p26 = Path(args.yolo11_log), Path(args.yolo26_log)
    y11, y26 = common.load(p11), common.load(p26)
    if sorted(y11) != sorted(y26) or len(y11) != 7490:
        raise RuntimeError('A/B requires identical complete 7,490-frame prediction logs')
    freeze = {'yolo11_log_sha256': common.digest(p11),
              'yolo26_log_sha256': common.digest(p26),
              'prediction_frames': len(y11), 'gt_loaded_after_predictions': True}

    frame_rows, cases = [], []
    only11_rank, only26_rank = Counter(), Counter()
    for frame in sorted(y11):
        left, right = y11[frame]['person_detections'], y26[frame]['person_detections']
        matched, only11, only26 = common.match_boxes(left, right)
        only11_rank[frame], only26_rank[frame] = len(only11), len(only26)
        frame_rows.append({'frame_id': frame, 'yolo11s_person_count': len(left),
                           'yolo26s_person_count': len(right),
                           'matched_boxes_iou_ge_0p5': len(matched),
                           'yolo11s_only_boxes': len(only11),
                           'yolo26s_only_boxes': len(only26)})
        for label, indices, items in (('YOLO11S_ONLY', only11, left),
                                      ('YOLO26S_ONLY', only26, right)):
            for index in indices:
                item = items[index]
                cases.append({'frame_id': frame, 'case_type': label,
                              'confidence': item['confidence'],
                              'bbox': json.dumps(item['bbox']),
                              'bbox_height_px': item['bbox'][3] - item['bbox'][1],
                              'size_bin': common.size_bin(item['bbox']),
                              'image_region': common.region(item['bbox'])})
    common.write_csv(out / 'YOLO11S_VS_YOLO26S_BOX_DIFF.csv', frame_rows)
    common.write_csv(out / 'YOLO11S_VS_YOLO26S_BOX_CASES.csv', cases)

    validation = evaluator.split_frames('VALIDATION')
    pred11, pred26 = evaluator.predictions(p11, validation), evaluator.predictions(p26, validation)
    # Official 3D GT is intentionally read only after both prediction logs are loaded and hashed.
    truth = evaluator.truth(validation)
    metrics11, rows11 = evaluator.score('YOLO11S_COCO', 'VALIDATION', validation, pred11, truth)
    metrics26, rows26 = evaluator.score('YOLO26S_COCO', 'VALIDATION', validation, pred26, truth)
    depth11, depth26 = common.score_by_depth(pred11, truth), common.score_by_depth(pred26, truth)

    near_miss = Counter()
    for frame in validation:
        _, hit26 = common.prediction_matches(pred26[frame], truth[frame])
        for index, xyz in enumerate(truth[frame]):
            if index not in hit26 and common.depth_label(xyz) in ('0-3m', '3-6m'):
                near_miss[frame] += 1
    fp_notes, fp_rank = {}, Counter()
    for left, right in zip(rows11, rows26, strict=True):
        frame = int(left['frame_id'])
        fp_rank[frame] = max(int(left['fp']), int(right['fp']))
        fp_notes[frame] = f"3D FP: 11s={left['fp']} 26s={right['fp']}"

    s11 = json.loads(Path(args.yolo11_uncapped).read_text(encoding='utf-8'))
    s26 = json.loads(Path(args.yolo26_uncapped).read_text(encoding='utf-8'))
    r11 = json.loads(Path(args.yolo11_realtime).read_text(encoding='utf-8'))
    r26 = json.loads(Path(args.yolo26_realtime).read_text(encoding='utf-8'))
    boxes11 = sum(len(row['person_detections']) for row in y11.values())
    boxes26 = sum(len(row['person_detections']) for row in y26.values())
    comparison = metric_rows(metrics11, metrics26, s11, s26, r11, r26,
                             boxes11, boxes26, depth11, depth26)
    common.write_csv(out / 'YOLO11S_VS_YOLO26S_COCO_COMPARISON.csv', comparison)

    result = {'prediction_freeze': freeze, 'evaluation': {
        'split': 'VALIDATION', 'assignment': '3D_HUNGARIAN_ONE_TO_ONE',
        'gate_m': evaluator.GATE_M, 'runtime_gt_used': False,
        'independent_2d_gt_available': False}, 'yolo11s': metrics11,
        'yolo26s': metrics26, 'depth_recall': {'yolo11s': depth11, 'yolo26s': depth26},
        'box_ab': {'yolo11s_rows': boxes11, 'yolo26s_rows': boxes26,
                   'yolo11s_only': sum(only11_rank.values()),
                   'yolo26s_only': sum(only26_rank.values())},
        'runtime_scope': {'detector_lidar_and_uncapped_fps': 'UNCAPPED',
                          'latency_sensor_fps_and_drops': 'SENSOR_PACED'}}
    (out / 'YOLO11S_VS_YOLO26S_COCO_METRICS.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')

    selected11 = [frame for frame, score in only11_rank.most_common() if score]
    selected26 = [frame for frame, score in only26_rank.most_common() if score]
    selected_near = [frame for frame, score in near_miss.most_common() if score]
    selected_fp = [frame for frame, score in fp_rank.most_common() if score]
    image_frames = set(selected11[:12] + selected26[:12] + selected_near[:12] + selected_fp[:12])
    images = read_images(image_frames)
    contact_sheet(selected11, y11, y26, images, out / 'YOLO11S_ONLY_CONTACT_SHEET.jpg',
                  'Box-level YOLO11s-only; not independent 2D GT')
    contact_sheet(selected26, y11, y26, images, out / 'YOLO26S_ONLY_CONTACT_SHEET.jpg',
                  'Box-level YOLO26s-only; not independent 2D GT')
    contact_sheet(selected_near, y11, y26, images,
                  out / 'NEAR_PERSON_MISSES_CONTACT_SHEET.jpg',
                  'YOLO26s 3D near-person FN on frozen VALIDATION')
    contact_sheet(selected_fp, y11, y26, images, out / 'FP_HEAVY_CASES_CONTACT_SHEET.jpg',
                  '3D FP-heavy frozen VALIDATION', fp_notes)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
