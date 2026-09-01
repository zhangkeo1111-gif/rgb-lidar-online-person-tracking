"""Post-run PERSON detection/state equivalence audit (no runtime GT use)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from online_v4 import pipeline, runtime


def load(path: Path) -> dict[int, dict]:
    result = {}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('processed'):
                result[int(row['rgb_frame_id'])] = row
    return result


def boxes(row: dict, lightweight: bool) -> list[tuple]:
    values = row.get('person_detections') if lightweight else row.get('persons', [])
    key = 'bbox' if lightweight else 'bbox'
    return [(tuple(float(x) for x in item[key]), float(item['confidence'])) for item in values]


def closest(rows: list[dict], bbox: list[float]) -> dict | None:
    if not rows:
        return None
    target = np.asarray(bbox, np.float64)
    return min(rows, key=lambda item: float(np.max(np.abs(np.asarray(item['bbox']) - target))))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(frame_risk: dict[int, float], full: dict[int, dict], light: dict[int, dict], output: Path) -> None:
    selected = set(sorted(frame_risk, key=frame_risk.get, reverse=True)[:12])
    tiles = []
    for event in runtime.stream_sensor_events(pipeline.DEFAULT_BAG, max(selected, default=-1) + 1, 0):
        if event.index not in selected:
            continue
        image = pipeline.decode_image(event.message)
        for title, row, color in (('FULL', full[event.index], (190, 120, 20)),
                                  ('LIGHTWEIGHT', light[event.index], (20, 170, 130))):
            tile = cv2.resize(image, (480, 270))
            scale = np.asarray([480 / 1280, 270 / 720, 480 / 1280, 270 / 720])
            for person in row.get('persons', []):
                box = np.round(np.asarray(person['bbox']) * scale).astype(int)
                cv2.rectangle(tile, tuple(box[:2]), tuple(box[2:]), color, 2)
                xyz = person.get('position_xyz')
                if xyz is None and person.get('xyz_valid'):
                    xyz = [person.get('x'), person.get('y'), person.get('z')]
                value = person['track_id'] if xyz is None else f"{person['track_id']} {xyz[0]:.2f},{xyz[1]:.2f}"
                cv2.putText(tile, value, (box[0], max(42, box[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            cv2.rectangle(tile, (0, 0), (480, 28), (15, 18, 22), -1)
            cv2.putText(tile, f'{title} frame {event.index} max delta {frame_risk[event.index]:.3f} m', (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
            tiles.append(tile)
    if not tiles:
        blank = np.full((270, 480, 3), 245, np.uint8)
        cv2.putText(blank, 'No XYZ changes', (130, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (50, 80, 100), 2, cv2.LINE_AA)
        tiles = [blank]
    columns = 4
    rows = []
    for start in range(0, len(tiles), columns):
        line = tiles[start:start + columns]
        line.extend([np.full_like(tiles[0], 245)] * (columns - len(line)))
        rows.append(np.hstack(line))
    cv2.imwrite(str(output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', required=True)
    parser.add_argument('--lightweight', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    full = load(Path(args.full))
    light = load(Path(args.lightweight))
    common = sorted(set(full) & set(light))
    detection_rows = []
    xyz_rows = []
    changed_frames: dict[int, float] = {}
    for frame_id in common:
        left = boxes(full[frame_id], False)
        right = boxes(light[frame_id], True)
        exact = left == right
        detection_rows.append({'frame_id': frame_id, 'full_count': len(left),
                               'lightweight_count': len(right), 'exact': exact})
        if not (full[frame_id].get('new_lidar_update') and light[frame_id].get('new_lidar_update')):
            continue
        for item in light[frame_id].get('persons', []):
            other = closest(full[frame_id].get('persons', []), item['bbox'])
            if other is None:
                continue
            box_delta = float(np.max(np.abs(np.asarray(item['bbox']) - np.asarray(other['bbox']))))
            if box_delta > 1e-6:
                continue
            lxyz = np.asarray(item['position_xyz'], np.float64) if item.get('position_xyz') is not None else None
            fxyz = np.asarray([other.get('x'), other.get('y'), other.get('z')], np.float64) if other.get('xyz_valid') else None
            if lxyz is None or fxyz is None:
                delta = None
            else:
                delta = float(np.linalg.norm(lxyz - fxyz))
            changed = delta is not None and delta > 1e-9
            if changed:
                changed_frames[frame_id] = max(changed_frames.get(frame_id, 0.0), float(delta))
            xyz_rows.append({'frame_id': frame_id, 'bbox': json.dumps(item['bbox']),
                             'full_track_id': other['track_id'], 'lightweight_track_id': item['track_id'],
                             'full_xyz_valid': bool(other.get('xyz_valid')), 'lightweight_xyz_valid': bool(item.get('xyz_valid')),
                             'xyz_delta_m': delta, 'changed': changed})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'person_detection_equivalence.csv', detection_rows)
    write_csv(output / 'person_xyz_equivalence.csv', xyz_rows)
    contact_sheet(changed_frames, full, light, output / 'contact_sheet_xyz_changes.jpg')
    finite = [row['xyz_delta_m'] for row in xyz_rows if row['xyz_delta_m'] is not None]
    summary = {'common_processed_rgb_frames': len(common),
               'detection_exact_frames': sum(row['exact'] for row in detection_rows),
               'detection_changed_frames': sum(not row['exact'] for row in detection_rows),
               'xyz_compared_rows': len(finite),
               'xyz_identical_rows': sum(value <= 1e-9 for value in finite),
               'xyz_changed_rows': sum(value > 1e-9 for value in finite),
               'xyz_max_delta_m': max(finite, default=None)}
    (output / 'equivalence_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
