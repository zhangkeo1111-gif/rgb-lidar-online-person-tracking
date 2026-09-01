"""Frozen VALIDATION scoring of already-produced causal runtime logs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

CLEAN_ROOT = Path(r'D:\navwareset_scene01_clean')
MANIFEST = CLEAN_ROOT / 'data/splits/annotated_split_manifest.csv'
GT = CLEAN_ROOT / 'data/canonical/gt/person_cuboid_gt_v2.csv'
GATE_M = 1.5


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def split_frames(split: str) -> list[int]:
    return [int(row['annotated_frame_index']) for row in read_csv(MANIFEST)
            if row['split'] == split]


def truth(frames: list[int]) -> dict[int, list[np.ndarray]]:
    wanted = set(frames)
    values: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in read_csv(GT):
        frame = int(row['annotated_frame_index'])
        if frame in wanted and row['gt_valid'] == 'True':
            values[frame].append(np.asarray([
                float(row['center_annotated_x_m']), float(row['center_annotated_y_m']),
                float(row['center_annotated_z_m'])], np.float64))
    if sum(map(len, values.values())) != 5 * len(frames):
        raise RuntimeError('Frozen VALIDATION GT count is not five persons per frame')
    return values


def predictions(path: Path, frames: list[int]) -> dict[int, list[np.ndarray]]:
    wanted = set(frames)
    values: dict[int, list[np.ndarray]] = {frame: [] for frame in frames}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get('processed') or not row.get('new_lidar_update'):
                continue
            detail = row.get('lidar_update_detail') or {}
            frame = detail.get('lidar_frame_id', row.get('latest_lidar_frame_id'))
            if frame not in wanted:
                continue
            frame_rows = []
            for person in row.get('persons', []):
                if not person.get('xyz_valid'):
                    continue
                xyz = person.get('position_xyz')
                if xyz is None:
                    xyz = [person['x'], person['y'], person['z']]
                frame_rows.append(np.asarray(xyz, np.float64))
            values[int(frame)] = frame_rows
    return values


def distribution(values: list[float]) -> dict:
    data = np.asarray(values, np.float64)
    if not len(data):
        return {'rmse': None, 'median': None, 'p95': None, 'max': None}
    return {'rmse': float(np.sqrt(np.mean(data ** 2))),
            'median': float(np.median(data)), 'p95': float(np.percentile(data, 95)),
            'max': float(data.max())}


def score(name: str, split: str, frames: list[int], pred: dict[int, list[np.ndarray]],
          gt: dict[int, list[np.ndarray]]) -> tuple[dict, list[dict]]:
    tp = fp = fn = 0
    xy_errors: list[float] = []
    xyz_errors: list[float] = []
    frame_rows: list[dict] = []
    for frame in frames:
        pp, gg = pred[frame], gt[frame]
        if not pp:
            fn += len(gg)
            frame_rows.append({'frame_id': frame, 'prediction_count': 0, 'tp': 0, 'fp': 0, 'fn': len(gg)})
            continue
        cost = np.linalg.norm(np.asarray(pp)[:, None, :] - np.asarray(gg)[None, :, :], axis=2)
        rows, columns = linear_sum_assignment(cost)
        accepted = [(int(p), int(g)) for p, g in zip(rows, columns, strict=True)
                    if cost[p, g] <= GATE_M]
        tp += len(accepted)
        fp += len(pp) - len(accepted)
        fn += len(gg) - len(accepted)
        frame_rows.append({'frame_id': frame, 'prediction_count': len(pp), 'tp': len(accepted),
                           'fp': len(pp) - len(accepted), 'fn': len(gg) - len(accepted)})
        for p, g in accepted:
            delta = pp[p] - gg[g]
            xy_errors.append(float(np.linalg.norm(delta[:2])))
            xyz_errors.append(float(np.linalg.norm(delta)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return ({'name': name, 'scope': f'FROZEN_{split}_POST_RUNTIME_ONLY',
            'frames': len(frames), 'gt_count': 5 * len(frames),
            'prediction_count': sum(len(value) for value in pred.values()),
            'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / max(precision + recall, 1e-12),
            'xy': distribution(xy_errors), 'xyz': distribution(xyz_errors),
            'assignment': '3D_HUNGARIAN_ONE_TO_ONE', 'gate_m': GATE_M,
            'runtime_gt_used': False, 'gt_loaded_after_predictions': True}, frame_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', required=True)
    parser.add_argument('--lightweight', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('VALIDATION', 'TEST'), default='VALIDATION')
    args = parser.parse_args()
    frames = split_frames(args.split)
    full_predictions = predictions(Path(args.full), frames)
    light_predictions = predictions(Path(args.lightweight), frames)
    gt = truth(frames)
    full_metrics, full_rows = score('FULL', args.split, frames, full_predictions, gt)
    light_metrics, light_rows = score('LIGHTWEIGHT', args.split, frames, light_predictions, gt)
    result = {'split': args.split, 'full': full_metrics, 'lightweight': light_metrics}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    with output.with_name(output.stem + '_framewise.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=('frame_id', 'full_prediction_count', 'lightweight_prediction_count',
                                                    'full_tp', 'lightweight_tp', 'full_fp', 'lightweight_fp',
                                                    'full_fn', 'lightweight_fn'))
        writer.writeheader()
        for left, right in zip(full_rows, light_rows, strict=True):
            writer.writerow({'frame_id': left['frame_id'], 'full_prediction_count': left['prediction_count'],
                             'lightweight_prediction_count': right['prediction_count'], 'full_tp': left['tp'],
                             'lightweight_tp': right['tp'], 'full_fp': left['fp'], 'lightweight_fp': right['fp'],
                             'full_fn': left['fn'], 'lightweight_fn': right['fn']})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
