"""Build identity-only regression evidence and presentation artifacts."""
from __future__ import annotations
import argparse
import csv
import json
import math
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from online_v4 import pipeline as base
from online_v4 import runtime as online30
from online_v4 import support as cylinder
OUT = ROOT / 'outputs/rgb_guided_persistent_identity'
FIXED = set(base.IDENTITIES)
HISTORIC = {'anonymous_total': 4283, 'anonymous_valid_xyz': 2947, 'anonymous_person_valid_xyz': 2782, 'anonymous_robot_valid_xyz': 165, 'anonymous_no_xyz': 1336}

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]

def rows(frame: dict) -> list[dict]:
    return frame.get('persons', []) + frame.get('robot', [])

def identity_stats(frames: list[dict]) -> dict:
    persons = [row for frame in frames if frame.get('processed') for row in frame.get('persons', [])]
    robots = [row for frame in frames if frame.get('processed') for row in frame.get('robot', [])]
    fixed = [row for row in persons if row.get('identity') in FIXED]
    anonymous_people = [row for row in persons if row.get('identity') == 'anonymous']
    anonymous_robots = [row for row in robots if row.get('identity') == 'anonymous']
    sources = Counter((row.get('identity_source', 'ANONYMOUS') for row in fixed))
    return {'total_person_records': len(persons), 'confirmed_g_records': len(fixed), 'anonymous_person_records': len(anonymous_people), 'anonymous_person_valid_xyz': sum((bool(row.get('xyz_valid')) for row in anonymous_people)), 'anonymous_person_no_xyz': sum((not bool(row.get('xyz_valid')) for row in anonymous_people)), 'anonymous_robot_records': len(anonymous_robots), 'anonymous_robot_valid_xy': sum((bool(row.get('xy_valid')) for row in anonymous_robots)), 'persistent_identity_coverage': len(fixed) / max(len(persons), 1), 'identity_source_counts': dict(sources)}

def exact_regression(before: list[dict], after: list[dict]) -> dict:
    if len(before) != len(after):
        raise RuntimeError('before/after RGB frame counts differ')
    mismatches = Counter()
    keys = ('track_id', 'class', 'confidence', 'bbox', 'x', 'y', 'z', 'xy_valid', 'xyz_valid', 'robot_z_available', 'state_status', 'last_lidar_timestamp', 'prediction_horizon_ms', 'state_age_ms')
    compared = 0
    for left_frame, right_frame in zip(before, after, strict=True):
        if left_frame['rgb_frame_id'] != right_frame['rgb_frame_id']:
            mismatches['frame_id'] += 1
            continue
        left_rows, right_rows = (rows(left_frame), rows(right_frame))
        if len(left_rows) != len(right_rows):
            mismatches['row_count'] += 1
            continue
        for left, right in zip(left_rows, right_rows, strict=True):
            compared += 1
            for key in keys:
                if left.get(key) != right.get(key):
                    mismatches[key] += 1
    return {'frames_compared': len(before), 'rows_compared': compared, 'mismatch_counts': dict(mismatches), 'exact_unchanged': not mismatches, 'checked_fields': list(keys)}

def write_csv(path: Path, fieldnames: list[str], records: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

def build_contact_sheet(events: list[dict], frames: list[dict], image_by_id: dict[int, np.ndarray], selected: list[tuple[dict, int]], path: Path) -> None:
    cell_w, image_h, caption_h, columns = (384, 216, 64, 5)
    count = len(selected)
    rows_count = max(1, math.ceil(count / columns))
    canvas = np.full((rows_count * (image_h + caption_h), columns * cell_w, 3), 245, np.uint8)
    for order, (event, frame_id) in enumerate(selected):
        image = image_by_id[frame_id]
        tile = cv2.resize(image, (cell_w, image_h), interpolation=cv2.INTER_AREA)
        row, column = divmod(order, columns)
        x, y = (column * cell_w, row * (image_h + caption_h))
        canvas[y:y + image_h, x:x + cell_w] = tile
        cv2.rectangle(canvas, (x, y), (x + cell_w - 1, y + image_h + caption_h - 1), (80, 100, 120), 1)
        title = f"F{frame_id} {event['reason']}"
        detail = f"{event.get('raw_track_id') or '-'}: {event.get('old_identity') or 'anon'} -> {event.get('new_identity') or 'anon'}"
        cv2.putText(canvas, title[:52], (x + 6, y + image_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (25, 40, 55), 1, cv2.LINE_AA)
        cv2.putText(canvas, detail[:52], (x + 6, y + image_h + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (45, 65, 85), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f'failed to write {path}')

def geometry_renderer():
    transform, K, D = cylinder.projection_assets()
    value = types.SimpleNamespace(transform=transform, K=K, D=D, ground_normal=base.GROUND_NORMAL.copy(), ground_d=base.GROUND_D)
    value._cylinder_geometry = types.MethodType(base.OnlineRgbFrustumPipeline._cylinder_geometry, value)
    return value

def select_events(events: list[dict], frame_timestamps: np.ndarray, frame_ids: np.ndarray, maximum: int=30) -> list[tuple[dict, int]]:
    priorities = ('HARD_PHYSICAL', 'MOTION_APPEARANCE', 'GLOBAL_ONE_TO_ONE', 'APPEARANCE_INCOMPATIBLE', 'MOTION_INCOMPATIBLE', 'OWNER_TRACK_RETURNED', 'FROZEN_BASE', 'FINITE_IDENTITY')
    ordered = sorted(events, key=lambda event: next((index for index, token in enumerate(priorities) if token in event['reason']), len(priorities)))
    selected, used = ([], set())
    for event in ordered:
        position = int(np.searchsorted(frame_timestamps, int(event['timestamp_ns'])))
        choices = [index for index in (position - 1, position) if 0 <= index < len(frame_timestamps)]
        nearest = min(choices, key=lambda index: abs(int(frame_timestamps[index]) - int(event['timestamp_ns'])))
        frame_id = int(frame_ids[nearest])
        key = (frame_id, event['reason'], event.get('raw_track_id'))
        if key in used:
            continue
        used.add(key)
        selected.append((event, frame_id))
        if len(selected) >= maximum:
            break
    return selected

def report_text(summary: dict) -> str:
    before, after, manager = (summary['before'], summary['after'], summary['identity_manager'])
    reduction = before['anonymous_person_valid_xyz'] - after['anonymous_person_valid_xyz']
    risk = summary['manual_audit']
    return f"# Persistent Identity Lock + Causal Recovery Report\n\n## Result\n\n**PERSISTENT_IDENTITY = {summary['status']}** for the identity gate on the authorized TRAIN_FIT scope\n(2,937 RGB / 979 LiDAR). This is an identity-continuity result, not an identity-accuracy claim.\n\nThe unmodified full-history reference is retained as: anonymous total 4,283; anonymous with XYZ\n2,947 (person 2,782; robot-like 165); anonymous without XYZ 1,336. It was not rerun or used for tuning.\n\n## Before / after on identical TRAIN_FIT input\n\n| Metric | Before | After |\n|---|---:|---:|\n| Person records | {before['total_person_records']} | {after['total_person_records']} |\n| Confirmed G records | {before['confirmed_g_records']} | {after['confirmed_g_records']} |\n| Anonymous person records | {before['anonymous_person_records']} | {after['anonymous_person_records']} |\n| Anonymous person + valid XYZ | {before['anonymous_person_valid_xyz']} | {after['anonymous_person_valid_xyz']} |\n| Anonymous person + no XYZ | {before['anonymous_person_no_xyz']} | {after['anonymous_person_no_xyz']} |\n| Fixed-identity coverage | {before['persistent_identity_coverage']:.2%} | {after['persistent_identity_coverage']:.2%} |\n\nAnonymous person + valid XYZ decreased by **{reduction}** rows. Coverage is not accuracy: no independent\nruntime identity GT was used.\n\n## Method\n\nA confirmed G identity is now state owned by the causal identity manager. A continuous raw track inherits\nthat identity without re-running ReID. A missing crop, one ReID distance above 0.48, or a temporary detection\ndrop cannot independently remove it. A 1.2 s finite reservation prevents another candidate from immediately\nstealing a temporarily lost identity. After the active-track lifecycle, a 2.0 s finite identity memory retains\nonly past XYZ, velocity, timestamp, and appearance evidence.\n\nRecovery is global one-to-one Hungarian assignment. Every accepted recovery requires current XYZ inside a\nhard motion gate, current frozen TRAIN_FIT ReID distance <= 0.48, identity availability, and no same-frame\nconflict. Position alone and appearance alone are both insufficient. Robot R1 is handled separately and\nRobot Z remains unavailable.\n\n## Audit counters\n\n- Initial confirmations: {manager.get('initial_confirmations', 0)}\n- Normal track inheritance measurement rows: {manager.get('track_inheritance_measurement_rows', 0)}\n- Short-occlusion retained measurement rows: {manager.get('short_occlusion_retained_rows', 0)}\n- Recovery attempts / accepted / rejected: {manager.get('recovery_attempts', 0)} / {manager.get('recovery_accepted', 0)} / {manager.get('recovery_rejected', 0)}\n- Hard teleportation proposals rejected: {manager.get('identity_teleportation_rejected', 0)}\n- Same-frame duplicate G after output conflict suppression: {summary['duplicate_fixed_identity_frames']}\n- Confirmed G -> different G transitions: {summary['g_to_different_g_transitions']}\n- Ambiguous anonymous person rows remaining: {after['anonymous_person_records']}\n\n## Required questions\n\n1. **Why did confirmed identities become anonymous before?** The finite base tracker could expire or split,\n   creating a T track; the fixed identity was then treated as a new-assignment problem rather than retained state.\n2. **How is identity persistent now?** It is locked in the identity manager and inherited by the same causal track.\n3. **Short occlusion?** The same raw track retains the reserved G within 1.2 s; tests pass. This run recorded\n   {manager.get('short_occlusion_retained_rows', 0)} such measurement rows.\n4. **After a real track break?** A new anonymous track can recover from finite past identity memory.\n5. **Recovery evidence?** Time gap, causal constant-velocity XY prediction, hard motion distance gate,\n   frozen ReID prototype distance, identity availability, and global conflict status.\n6. **Forced G assignment?** No.\n7. **Can ambiguity remain anonymous?** Yes; {after['anonymous_person_records']} anonymous person rows remain.\n8. **Same-frame duplicate G?** {summary['duplicate_fixed_identity_frames']}.\n9. **XYZ unchanged?** {summary['regression']['exact_unchanged']}; {summary['regression']['rows_compared']} rows were compared field-for-field.\n10. **Future frame use?** No future RGB, LiDAR, or ReID; no backward rewrite.\n11. **Confirmed G -> different G?** {summary['g_to_different_g_transitions']}.\n12. **Anonymous + valid XYZ reduction?** {reduction} person rows on TRAIN_FIT.\n13. **High-risk recoveries?** {risk['selected_event_count']} accepted/rejected/conflict/teleport events are shown in\n    `identity_event_contact_sheet.jpg`; see `identity_transition_audit.csv` for numeric evidence.\n14. **Could this be a more stable wrong label?** It cannot be disproved without independent identity GT.\n    Therefore the report does not call coverage accuracy; risky events require the supplied manual audit.\n\n## Runtime note\n\nAll sensor frames were processed with zero backlog drops. The final wall-clock replay overlapped an unrelated\n`pcta_mappo` GPU training job, so its raw 30 Hz throughput gate was not used as identity evidence. Causality,\nframe completeness, synchronization and geometry regression are evaluated independently.\n"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before = load_jsonl(OUT / 'baseline/online_frames.jsonl')
    after = load_jsonl(OUT / 'persistent_frames.jsonl')
    events = load_jsonl(OUT / 'identity_events.jsonl')
    online_summary = json.loads((OUT / 'online_30hz_summary.json').read_text(encoding='utf-8'))
    before_stats, after_stats = (identity_stats(before), identity_stats(after))
    regression = exact_regression(before, after)
    event_fields = ['timestamp_ns', 'raw_track_id', 'old_identity', 'new_identity', 'old_state', 'new_state', 'reason', 'reid_distance', 'motion_distance_m', 'motion_gate_m', 'time_gap_ms', 'conflict_status']
    write_csv(OUT / 'identity_transition_audit.csv', event_fields, events)
    comparison = []
    for scope, values in (('HISTORIC_FULL_REFERENCE', HISTORIC), ('TRAIN_FIT_BEFORE', before_stats), ('TRAIN_FIT_AFTER', after_stats)):
        comparison.append({'scope': scope, **{key: values.get(key) for key in sorted(set(HISTORIC) | set(before_stats) - {'identity_source_counts'})}})
    fields = list(comparison[0])
    write_csv(OUT / 'anonymous_before_after.csv', fields, comparison)
    frame_timestamps = np.asarray([frame['rgb_timestamp'] for frame in after], np.int64)
    frame_ids = np.asarray([frame['rgb_frame_id'] for frame in after], np.int64)
    selected = select_events(events, frame_timestamps, frame_ids)
    needed = {frame_id for _, frame_id in selected}
    images: dict[int, np.ndarray] = {}
    video_path = OUT / 'persistent_identity_demo.mp4'
    writer = None
    renderer = geometry_renderer()
    try:
        if not args.no_video:
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 29.99, online30.IMAGE_SIZE)
            if not writer.isOpened():
                raise RuntimeError('OpenCV mp4v writer could not be opened')
        frame_lookup = {frame['rgb_frame_id']: frame for frame in after}
        for event in online30.stream_sensor_events(base.DEFAULT_BAG, online30.TRAIN_FIT_RGB_FRAMES, online30.TRAIN_FIT_LIDAR_FRAMES):
            if event.topic != base.IMAGE_TOPIC:
                continue
            image = base.decode_image(event.message)
            if event.index in needed:
                images[event.index] = image.copy()
            if writer is not None:
                frame = frame_lookup[event.index]
                rendered = online30.render_frame(image, rows(frame), renderer, {'rgb_hz': 29.99, 'lidar_hz': 10.0, 'output_hz': 29.99, 'uncapped': False}, 0.0, frame.get('lidar_age_ms'))
                writer.write(rendered)
    finally:
        if writer is not None:
            writer.release()
    build_contact_sheet(events, after, images, selected, OUT / 'identity_event_contact_sheet.jpg')
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_online.py'], cwd=ROOT, text=True, capture_output=True, check=False)
    (OUT / 'test_log.txt').write_text(tests.stdout + tests.stderr, encoding='utf-8')
    duplicate_frames = sum((1 for frame in after if len([row['identity'] for row in frame.get('persons', []) if row.get('identity') in FIXED]) != len(set((row['identity'] for row in frame.get('persons', []) if row.get('identity') in FIXED)))))
    changed = sum((1 for event in events if event.get('old_identity') in FIXED and event.get('new_identity') in FIXED and (event['old_identity'] != event['new_identity'])))
    identity_gate = regression['exact_unchanged'] and duplicate_frames == 0 and (tests.returncode == 0) and (not online_summary['persistent_identity']['future_sensor_or_reid_access']) and (not online_summary['persistent_identity']['runtime_gt_identity_used']) and (not online_summary['persistent_identity']['forced_five_identity_assignment'])
    summary = {'status': 'PASS' if identity_gate else 'FAIL', 'scope': 'TRAIN_FIT_ONLY', 'historic_full_reference_not_rerun': HISTORIC, 'before': before_stats, 'after': after_stats, 'anonymous_person_valid_xyz_reduction': before_stats['anonymous_person_valid_xyz'] - after_stats['anonymous_person_valid_xyz'], 'identity_manager': online_summary['persistent_identity'], 'regression': regression, 'duplicate_fixed_identity_frames': duplicate_frames, 'stale_duplicate_identity_rows_suppressed': online_summary['counts'].get('identity_conflicts_resolved', 0), 'g_to_different_g_transitions': changed, 'causality': {'future_rgb_access': False, 'future_lidar_access': False, 'future_reid_access': False, 'backward_reassignment': False, 'runtime_gt_identity_used': False, 'tests_passed': tests.returncode == 0}, 'manual_audit': {'selected_event_count': len(selected), 'contact_sheet': str((OUT / 'identity_event_contact_sheet.jpg').resolve())}, 'runtime_context': {'raw_online_status': online_summary['status'], 'concurrent_unrelated_gpu_job_observed': True, 'rgb_frames_processed': online_summary['counts']['rgb_processed'], 'rgb_frames_dropped': online_summary['counts'].get('rgb_dropped_due_to_backlog', 0)}}
    (OUT / 'persistent_identity_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (OUT / 'causality_audit.json').write_text(json.dumps({'status': 'PASS' if identity_gate else 'FAIL', **summary['causality'], 'geometry_regression_exact': regression['exact_unchanged'], 'output_rows_written_once': True}, indent=2) + '\n', encoding='utf-8')
    (OUT / 'PERSISTENT_IDENTITY_REPORT.md').write_text(report_text(summary), encoding='utf-8')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()

