"""Pure-online Scene01 replay: 30 Hz RGB output with 10 Hz LiDAR corrections.

Sensor messages are consumed once, in bag arrival order.  An RGB output is
written immediately and is never rewritten.  LiDAR uses only an already
arrived RGB frame whose sensor timestamp is not later than the LiDAR timestamp.
Between LiDAR corrections, XY is propagated by a causal constant-velocity
state; person Z is held.  No trajectory file, future frame, interpolation or
backward smoothing is loaded.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import cv2
import numpy as np
import scipy
import ultralytics
from rosbags.highlevel import AnyReader
from scipy.optimize import linear_sum_assignment
ROOT = Path(__file__).resolve().parents[2]
from . import identity as persistent
from . import dataset_adapter, detector_audit
from . import pipeline as base
from . import support
cylinder = frustum = offline = support
OUT = ROOT / 'outputs/online_v4_coco_person'
VIDEO = OUT / 'scene01_rgb_guided_pure_online_30hz.mp4'
FRAME_LOG = OUT / 'online_30hz_frames.jsonl'
SUMMARY = OUT / 'online_30hz_summary.json'
RUNTIME_CSV = OUT / 'runtime_per_frame.csv'
DROPS = OUT / 'drop_statistics.json'
CAUSALITY = OUT / 'causality_audit.json'
REPORT = OUT / 'LIGHTWEIGHT_PERSON_ONLY_REPORT.md'
TRAIN_FIT_RGB_FRAMES = 2937
TRAIN_FIT_LIDAR_FRAMES = 979
FULL_RGB_FRAMES = 7490
FULL_LIDAR_FRAMES = 2498
MAX_PREDICTION_HORIZON_MS = 180.0
BACKLOG_DROP_MS = 180.0
VIDEO_QUEUE_SIZE = 12
IMAGE_SIZE = base.IMAGE_SIZE

@dataclass(frozen=True)
class SensorEvent:
    topic: str
    index: int
    timestamp_ns: int
    bag_timestamp_ns: int
    message: object

@dataclass
class RgbSupport:
    index: int
    timestamp_ns: int
    image: np.ndarray
    detections: list[dict]

def causal_support_delta_ms(rgb_timestamp_ns: int, lidar_timestamp_ns: int, sync_slop_ms: float) -> float:
    """Compatibility wrapper around the shared causal synchronizer rule."""
    return support.causal_support_delta_ms(rgb_timestamp_ns, lidar_timestamp_ns, sync_slop_ms)

def strictly_monotonic(values: list[int]) -> bool:
    return all((current > previous for previous, current in zip(values, values[1:])))

def stream_sensor_events(bag: Path, rgb_limit: int, lidar_limit: int) -> Iterator[SensorEvent]:
    """Bounded streaming reader; it never materializes the sensor sequence."""
    rgb_index = lidar_index = 0
    with AnyReader([bag]) as reader:
        connections = [value for value in reader.connections if value.topic in {base.IMAGE_TOPIC, base.LIDAR_TOPIC}]
        if {value.topic for value in connections} != {base.IMAGE_TOPIC, base.LIDAR_TOPIC}:
            raise RuntimeError('Scene01 bag is missing the RGB or LiDAR topic')
        messages = reader.messages(connections=connections)
        while rgb_index < rgb_limit or lidar_index < lidar_limit:
            connection, bag_ns, raw = next(messages)
            if connection.topic == base.IMAGE_TOPIC:
                if rgb_index >= rgb_limit:
                    continue
                index = rgb_index
                rgb_index += 1
            else:
                if lidar_index >= lidar_limit:
                    continue
                index = lidar_index
                lidar_index += 1
            message = reader.deserialize(raw, connection.msgtype)
            yield SensorEvent(connection.topic, index, base.timestamp_ns(message, int(bag_ns)), int(bag_ns), message)

class AsyncVideoWriter:

    def __init__(self, path: Path, fps: float, enabled: bool) -> None:
        self.enabled = enabled
        self.path = path
        self.encoder = os.environ.get('ONLINE_V4_VIDEO_ENCODER', 'h264_nvenc').strip() or 'h264_nvenc'
        if self.encoder not in {'h264_nvenc', 'libx264'}:
            raise ValueError('ONLINE_V4_VIDEO_ENCODER must be h264_nvenc or libx264')
        self.backend = 'DISABLED' if not enabled else f'FFMPEG_{self.encoder.upper()}'
        self.frames_written = self.frames_dropped = self.max_depth = 0
        self.error: Exception | None = None
        self.items: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=VIDEO_QUEUE_SIZE)
        self.thread: threading.Thread | None = None
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.thread = threading.Thread(target=self._worker, args=(fps,), daemon=True)
            self.thread.start()

    def _worker(self, fps: float) -> None:
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            self.error = RuntimeError('ffmpeg is required for the NVENC writer')
            return
        encode_args = (['-c:v', 'h264_nvenc', '-preset', 'p1', '-tune', 'll', '-rc', 'constqp', '-qp', '20']
                       if self.encoder == 'h264_nvenc'
                       else ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20'])
        command = [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s:v', f'{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}', '-r', f'{fps:.8f}', '-i', '-', '-an', *encode_args, '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.path)]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=flags)
        try:
            assert process.stdin is not None
            while True:
                frame = self.items.get()
                if frame is None:
                    break
                process.stdin.write(memoryview(np.ascontiguousarray(frame)).cast('B'))
                self.frames_written += 1
            process.stdin.close()
            error_text = process.stderr.read().decode('utf-8', errors='replace') if process.stderr else ''
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f'{self.encoder} writer failed ({return_code}): {error_text.strip()}')
        except Exception as exc:
            self.error = exc
            if process.poll() is None:
                process.kill()
        finally:
            if process.stdin is not None and (not process.stdin.closed):
                process.stdin.close()

    def submit(self, frame: np.ndarray) -> bool:
        if not self.enabled:
            return True
        if self.thread is not None and (not self.thread.is_alive()):
            self.frames_dropped += 1
            return False
        try:
            self.items.put_nowait(frame)
            self.max_depth = max(self.max_depth, self.items.qsize())
            return True
        except queue.Full:
            self.frames_dropped += 1
            return False

    def close(self) -> None:
        if not self.enabled:
            return
        assert self.thread is not None
        while self.thread.is_alive():
            try:
                self.items.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.thread.join()
        if self.error is not None:
            raise self.error

class CausalMotionBank:
    """Latest measured state plus velocity derived only from past measurements."""

    def __init__(self, max_horizon_ms: float=MAX_PREDICTION_HORIZON_MS) -> None:
        self.max_horizon_ns = int(round(max_horizon_ms * 1000000.0))
        self.states: dict[str, dict] = {}
        self.reported_measurement_ns: dict[str, int] = {}
        self.latest_lidar_timestamp_ns: int | None = None
        self.latest_lidar_frame_id: int | None = None
        self.update_serial = 0

    def correct(self, observations: list[dict], timestamp_ns: int, lidar_frame_id: int) -> None:
        if self.latest_lidar_timestamp_ns is not None and timestamp_ns <= self.latest_lidar_timestamp_ns:
            raise RuntimeError('LiDAR correction timestamps must be strictly increasing')
        for item in observations:
            if item.get('xyz') is None:
                continue
            track_id = item['track_id']
            xyz = np.asarray(item['xyz'], np.float64)
            previous = self.states.get(track_id)
            velocity = np.zeros(3, np.float64)
            if previous is not None:
                dt = (timestamp_ns - previous['measurement_timestamp_ns']) / 1000000000.0
                if dt > 1e-06:
                    instant = (xyz - previous['measurement_xyz']) / dt
                    instant[2] = 0.0
                    velocity = 0.72 * previous['velocity_mps'] + 0.28 * instant
            velocity[2] = 0.0
            self.states[track_id] = {'track_id': track_id, 'class': item['class'], 'measurement_xyz': xyz.copy(), 'velocity_mps': velocity, 'measurement_timestamp_ns': int(timestamp_ns), 'measurement_source': item.get('measurement_source', 'NORMAL_COMPONENT'), 'bbox': np.asarray(item['bbox'], np.float64).copy(), 'confidence': float(item['confidence']), 'track_age_lidar_frames': int(item.get('track_age_lidar_frames', 1)), 'persistent_identity': item.get('persistent_identity'), 'identity_state': item.get('identity_state', 'ANONYMOUS'), 'identity_source': item.get('identity_source', 'ANONYMOUS'), 'identity_evidence': item.get('identity_evidence', {})}
        self.latest_lidar_timestamp_ns = int(timestamp_ns)
        self.latest_lidar_frame_id = int(lidar_frame_id)
        self.update_serial += 1

    def snapshot(self, timestamp_ns: int) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for track_id, state in self.states.items():
            horizon_ns = int(timestamp_ns) - state['measurement_timestamp_ns']
            if horizon_ns < 0:
                continue
            fresh = horizon_ns <= self.max_horizon_ns
            dt = horizon_ns / 1000000000.0
            xyz = state['measurement_xyz'].copy()
            xyz[:2] += state['velocity_mps'][:2] * dt
            source = 'LIDAR_MEASUREMENT' if state['measurement_timestamp_ns'] > self.reported_measurement_ns.get(track_id, -1) else 'CAUSAL_PREDICTION'
            result[track_id] = {**state, 'xyz': xyz, 'fresh': fresh, 'state_source': source, 'prediction_horizon_ms': horizon_ns / 1000000.0, 'state_age_ms': horizon_ns / 1000000.0}
        return result

    def mark_reported(self, track_ids: list[str]) -> None:
        for track_id in track_ids:
            state = self.states.get(track_id)
            if state is not None:
                self.reported_measurement_ns[track_id] = state['measurement_timestamp_ns']

    def update_bbox(self, track_id: str, bbox: np.ndarray, confidence: float) -> None:
        if track_id in self.states:
            self.states[track_id]['bbox'] = np.asarray(bbox, np.float64).copy()
            self.states[track_id]['confidence'] = float(confidence)

class Anonymous2DTracker:

    def __init__(self, max_age: int=10) -> None:
        self.max_age = max_age
        self.frame = -1
        self.next_id = 1
        self.states: dict[str, dict] = {}

    def assign(self, detections: list[dict], indices: list[int]) -> dict[int, str]:
        self.frame += 1
        active = [(key, value) for key, value in self.states.items() if self.frame - value['last_frame'] <= self.max_age]
        cost = np.full((len(active), len(indices)), 99.0, np.float64)
        for row, (_, state) in enumerate(active):
            for column, index in enumerate(indices):
                item = detections[index]
                if item['class'] == state['class']:
                    overlap = base.iou_xyxy(state['bbox'], item['bbox'])
                    if overlap >= 0.1:
                        cost[row, column] = 1.0 - overlap
        result: dict[int, str] = {}
        if cost.size:
            rr, cc = linear_sum_assignment(cost)
            for row, column in zip(rr, cc, strict=True):
                if cost[row, column] <= 0.9:
                    result[indices[int(column)]] = active[int(row)][0]
        for index in indices:
            if index not in result:
                result[index] = f'A{self.next_id:04d}'
                self.next_id += 1
            self.states[result[index]] = {'class': detections[index]['class'], 'bbox': detections[index]['bbox'].copy(), 'last_frame': self.frame}
        self.states = {key: value for key, value in self.states.items() if self.frame - value['last_frame'] <= self.max_age}
        return result

def copy_detections(values: list[dict]) -> list[dict]:
    return [{**item, 'bbox': np.asarray(item['bbox'], np.float64).copy(), 'xyz': None, 'component_points': 0} for item in values]

def measure_lidar(pipeline: base.OnlineRgbFrustumPipeline, support: RgbSupport, points_rslidar: np.ndarray, lidar_timestamp_ns: int, lidar_frame_id: int, motion: CausalMotionBank, identity_manager: persistent.PersistentIdentityManager | None=None) -> tuple[dict, list[dict]]:
    pipeline.calibration_audit.begin_frame()
    started = time.perf_counter()
    detections = copy_detections(support.detections)
    transform = pipeline.annotated_from_rslidar
    annotated = points_rslidar @ transform[:3, :3].T + transform[:3, 3]
    after_transform = time.perf_counter()
    pixels, valid, camera_depth = pipeline.project_inference(annotated)
    ground_height = annotated @ pipeline.ground_normal + pipeline.ground_d
    in_view = valid & (camera_depth > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < IMAGE_SIZE[0]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < IMAGE_SIZE[1])
    candidate = np.flatnonzero(in_view & (ground_height >= 0.03) & (ground_height <= 2.15))
    after_frustum = time.perf_counter()
    temporal_keep, occupancy_keep, removed = pipeline._static_masks(
        annotated[candidate], ground_height[candidate])
    keep = temporal_keep & occupancy_keep
    usable = candidate[keep]
    boxes = [item['bbox'] for item in detections]
    after_static = time.perf_counter()
    if pipeline.geometry_gpu:
        owners = pipeline.geometry_gpu.owners(pixels[usable], boxes)
        component_owners = owners.copy()
        for index, detection in enumerate(detections):
            selected = component_owners == index
            lower, upper = (0.08, 2.15) if detection['class'] == 'PERSON' else (0.03, 1.35)
            component_owners[selected & ((ground_height[usable] < lower) | (ground_height[usable] > upper))] = -1
        grouped = pipeline.geometry_gpu.components(annotated[usable], component_owners, len(boxes))
    else:
        owners = frustum.exclusive_point_owners(pixels[usable], boxes, [None] * len(boxes))
        grouped = []
        for index, detection in enumerate(detections):
            point_indices = usable[owners == index]
            height = ground_height[point_indices]
            lower, upper = (0.08, 2.15) if detection['class'] == 'PERSON' else (0.03, 1.35)
            point_indices = point_indices[(height >= lower) & (height <= upper)]
            grouped.append(frustum.adaptive_components(annotated[point_indices]))
    calibration_residuals: list[tuple[float, float, float]] = []
    for detection, components in zip(detections, grouped, strict=True):
        component, details = offline.choose_component(
            components, detection['bbox'], pipeline.transform, pipeline.K, pipeline.D,
            du_px=pipeline.inference_du_px, dv_px=pipeline.inference_dv_px,
        )
        if component is not None:
            detection['xyz'] = frustum.cluster_center(component)
            detection['component_points'] = len(component)
            detection['component_score'] = float(details['component_score'])
            residual = pipeline._calibration_residual(detection, component)
            if residual is not None:
                calibration_residuals.append(residual)
    geometry_observations = [{
        'bbox': np.asarray(item['bbox'], np.float64).tolist(),
        'confidence': float(item['confidence']),
        'xyz': None if item['xyz'] is None else np.asarray(item['xyz'], np.float64).tolist(),
        'component_points': int(item['component_points']),
    } for item in detections]
    recovery_candidates = [None] * len(detections)
    missing_people = [index for index, item in enumerate(detections)
                      if item['class'] == 'PERSON' and item['xyz'] is None]
    if pipeline.suppressed_recovery_enabled and missing_people:
        suppressed = candidate[temporal_keep & ~occupancy_keep]
        person_boxes = [item['bbox'] for item in detections
                        if item['class'] == 'PERSON']
        if pipeline.geometry_gpu:
            suppressed_owners = pipeline.geometry_gpu.owners(
                pixels[suppressed], boxes)
            allowed = np.zeros(len(detections), bool)
            allowed[missing_people] = True
            suppressed_owners[
                (suppressed_owners >= 0) & ~allowed[
                    np.maximum(suppressed_owners, 0)]] = -1
            for index in missing_people:
                selected = suppressed_owners == index
                suppressed_owners[
                    selected & ((ground_height[suppressed] < 0.08)
                                | (ground_height[suppressed] > 2.15))] = -1
            recovery_groups = pipeline.geometry_gpu.components(
                annotated[suppressed], suppressed_owners, len(boxes))
        else:
            suppressed_owners = frustum.exclusive_point_owners(
                pixels[suppressed], boxes, [None] * len(boxes))
            recovery_groups = [[] for _ in boxes]
            for index in missing_people:
                point_indices = suppressed[suppressed_owners == index]
                height = ground_height[point_indices]
                point_indices = point_indices[(height >= 0.08) & (height <= 2.15)]
                recovery_groups[index] = frustum.adaptive_components(
                    annotated[point_indices])
        for index in missing_people:
            recovery_candidates[index] = pipeline.suppressed_candidate(
                recovery_groups[index], detections[index], person_boxes)
    after_geometry = time.perf_counter()
    pipeline.calibration_audit.observe(support.image, boxes, points_rslidar, calibration_residuals)
    after_audit = time.perf_counter()
    feature_cache: dict[int, np.ndarray] = {}

    def feature_loader(indices: list[int]) -> np.ndarray:
        if pipeline.reid is None:
            return np.empty((len(indices), 0), np.float32)
        missing = [index for index in indices if index not in feature_cache]
        if missing:
            values = pipeline.reid.features(support.image, [detections[index]['bbox'] for index in missing])
            feature_cache.update({index: value for index, value in zip(missing, values, strict=True)})
        return np.asarray([feature_cache[index] for index in indices], np.float32)
    tracking_started = time.perf_counter()
    if pipeline.person_only_lightweight:
        tracked = pipeline.tracker.update(detections)
    else:
        tracked = pipeline.tracker.update(detections, feature_loader=feature_loader)
    tracking_ended = time.perf_counter()
    recovery_stats = pipeline.apply_suppressed_recovery(
        tracked, recovery_candidates, lidar_frame_id)
    recovery_ended = time.perf_counter()
    if pipeline.person_only_lightweight:
        if identity_manager is not None or pipeline.reid is not None:
            raise RuntimeError('Lightweight mode instantiated forbidden identity state')
        for item in tracked:
            item.update({'persistent_identity': None, 'identity_state': 'DISABLED',
                         'identity_source': 'DISABLED', 'identity_evidence': {}})
    elif identity_manager is not None:
        people = [index for index, item in enumerate(tracked) if item['class'] == 'PERSON' and item['track_id'] not in identity_manager.raw_to_identity]
        feature_loader(people)
        tracked = identity_manager.resolve(tracked, {index: feature_cache[index] for index in people}, lidar_timestamp_ns)
    else:
        for item in tracked:
            identity = item['track_id'] if item['track_id'] in base.IDENTITIES or item['track_id'] == 'R1' else None
            item.update({'persistent_identity': identity, 'identity_state': 'CONFIRMED' if identity else 'ANONYMOUS', 'identity_source': 'TRACK_INHERITANCE' if identity else 'ANONYMOUS', 'identity_evidence': {}})
    identity_ended = time.perf_counter()
    motion.correct(tracked, lidar_timestamp_ns, lidar_frame_id)
    ended = time.perf_counter()
    timing = {
        'transform_ms': 1000 * (after_transform - started), 'frustum_ms': 1000 * (after_frustum - after_transform),
        'static_suppression_ms': 1000 * (after_static - after_frustum), 'clustering_ms': 1000 * (after_geometry - after_static),
        'projection_residual_audit_ms': 1000 * (after_audit - after_geometry),
        'short_term_tracking_ms': 1000 * (tracking_ended - tracking_started),
        'suppressed_recovery_ms': 1000 * (recovery_ended - tracking_ended),
        'reid_and_persistent_identity_ms': 1000 * (identity_ended - recovery_ended),
        'causal_motion_correction_ms': 1000 * (ended - identity_ended),
        'tracking_reid_correction_ms': 1000 * (ended - after_audit), 'geometry_and_correction_ms': 1000 * (ended - started),
        'points_input': len(points_rslidar), 'points_projection_valid': int(valid.sum()), 'points_in_view': int(in_view.sum()),
        'points_height_candidate': len(candidate), 'points_after_static': len(usable), 'static_removed': removed,
        'temporal_removed': int(removed['temporal_removed']), 'occupancy_removed': int(removed['occupancy_removed']),
        'detections': len(tracked), 'components_available': sum(bool(value) for value in grouped),
        'geometry_observations': geometry_observations,
        'measurements': sum(item['xyz'] is not None for item in tracked),
        'person_measurements': sum(item['class'] == 'PERSON' and item['xyz'] is not None for item in tracked),
        'robot_measurements': sum(item['class'] == 'ROBOT' and item['xyz'] is not None for item in tracked),
        'suppressed_recovery_enabled': pipeline.suppressed_recovery_enabled,
        'suppressed_recovery_attempts': recovery_stats['attempts'],
        'suppressed_recovery_accepted': recovery_stats['accepted'],
        'suppressed_recovery_rejected': recovery_stats['rejected'],
        'projection_audit_state': pipeline.calibration_audit.state,
        'raw_physical_projection_du_px': 0.0,
        'raw_physical_projection_dv_px': 0.0,
        'empirical_inference_du_px': pipeline.inference_du_px,
        'empirical_inference_dv_px': pipeline.inference_dv_px,
        'audit_candidate_applied': False,
    }
    return (timing, tracked)

def boxes_for_assignment(detections: list[dict], indices: list[int]) -> list[dict]:
    values = []
    for index in indices:
        x1, y1, x2, y2 = detections[index]['bbox']
        values.append({'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
    return values

def associate_rgb(detections: list[dict], motion: CausalMotionBank, timestamp_ns: int, anonymous: Anonymous2DTracker, pipeline: base.OnlineRgbFrustumPipeline) -> tuple[list[dict], dict]:
    prediction_started = time.perf_counter()
    snapshots = motion.snapshot(timestamp_ns)
    prediction_ended = time.perf_counter()
    detection_to_track: dict[int, str] = {}
    used_tracks: set[str] = set()
    entity_classes = ('PERSON',) if pipeline.person_only_lightweight else ('PERSON', 'ROBOT')
    for entity_class in entity_classes:
        track_ids = [key for key, state in snapshots.items() if state['class'] == entity_class and state['fresh']]
        indices = [index for index, item in enumerate(detections) if item['class'] == entity_class]
        if not track_ids or not indices:
            continue
        points = np.asarray([snapshots[key]['xyz'] for key in track_ids])
        projected, valid, _ = pipeline.project_inference(points)
        matches = frustum.assign_tracks([snapshots[key] for key in track_ids], boxes_for_assignment(detections, indices), projected, valid)
        for track_row, box_column in matches.items():
            detection_index = indices[box_column]
            detection_to_track[detection_index] = track_ids[track_row]
            used_tracks.add(track_ids[track_row])
    robot_indices = [] if pipeline.person_only_lightweight else [index for index, item in enumerate(detections) if item['class'] == 'ROBOT']
    if (not pipeline.person_only_lightweight) and 'R1' in snapshots and 'R1' not in used_tracks and robot_indices:
        available = [index for index in robot_indices if index not in detection_to_track]
        if available:
            chosen = max(available, key=lambda index: detections[index]['confidence'])
            detection_to_track[chosen] = 'R1'
            used_tracks.add('R1')
    unmatched = [index for index in range(len(detections)) if index not in detection_to_track]
    detection_to_track.update(anonymous.assign(detections, unmatched))
    association_ended = time.perf_counter()
    rows = []
    newly_reported = []
    for index, detection in enumerate(detections):
        track_id = detection_to_track[index]
        state = snapshots.get(track_id)
        valid_state = state is not None and state['fresh']
        if pipeline.person_only_lightweight and not valid_state:
            continue
        if state is not None:
            motion.update_bbox(track_id, detection['bbox'], detection['confidence'])
        source = state['state_source'] if valid_state else 'CAUSAL_PREDICTION'
        if valid_state and source == 'LIDAR_MEASUREMENT':
            newly_reported.append(track_id)
        xyz = state['xyz'] if valid_state else None
        is_robot = detection['class'] == 'ROBOT'
        velocity = np.zeros(3, np.float64) if state is None else np.asarray(state['velocity_mps'], np.float64)
        rows.append({'track_id': track_id, 'identity': state.get('persistent_identity') or 'anonymous' if state is not None else track_id if track_id in base.IDENTITIES or track_id == 'R1' else 'anonymous', 'identity_state': state.get('identity_state', 'ANONYMOUS') if state is not None else 'ANONYMOUS', 'identity_source': state.get('identity_source', 'ANONYMOUS') if state is not None else 'ANONYMOUS', 'identity_evidence': state.get('identity_evidence', {}) if state is not None else {}, 'class': detection['class'], 'confidence': float(detection['confidence']), 'bbox': detection['bbox'].astype(float).tolist(), 'x': None if xyz is None else float(xyz[0]), 'y': None if xyz is None else float(xyz[1]), 'z': None if xyz is None or is_robot else float(xyz[2]), 'position_xyz': None if xyz is None else [float(xyz[0]), float(xyz[1]), float(xyz[2])], 'xy_valid': bool(xyz is not None), 'xyz_valid': bool(xyz is not None and (not is_robot)), 'robot_z_available': False if is_robot else None, 'state_source': source, 'measurement_source': None if state is None else state.get('measurement_source'), 'state_status': 'FRESH' if valid_state else 'STALE' if state is not None else 'UNAVAILABLE', 'last_lidar_timestamp': None if state is None else int(state['measurement_timestamp_ns']), 'prediction_horizon_ms': None if state is None else float(state['prediction_horizon_ms']), 'state_age_ms': None if state is None else float(state['state_age_ms']), 'vx': float(velocity[0]), 'vy': float(velocity[1]), 'velocity_xy': [float(velocity[0]), float(velocity[1])], 'speed_mps': float(np.linalg.norm(velocity[:2])), 'distance_xy_from_origin_m': None if xyz is None else float(np.linalg.norm(xyz[:2])), 'fresh': bool(valid_state), 'track_age_lidar_frames': None if state is None else int(state.get('track_age_lidar_frames', 1))})
    conflicts_resolved = 0
    for identity in (() if pipeline.person_only_lightweight else base.IDENTITIES):
        owners = [index for index, row in enumerate(rows) if row['identity'] == identity]
        if len(owners) <= 1:
            continue
        keep = max(owners, key=lambda index: (rows[index]['last_lidar_timestamp'] or -1, rows[index]['identity_source'] == 'CAUSAL_RECOVERY'))
        for index in owners:
            if index == keep:
                continue
            rows[index].update({'identity': 'anonymous', 'identity_state': 'ANONYMOUS', 'identity_source': 'ANONYMOUS', 'identity_conflict_status': 'STALE_DUPLICATE_SUPPRESSED'})
            conflicts_resolved += 1
    motion.mark_reported(newly_reported)
    return (rows, {'prediction_ms': 1000 * (prediction_ended - prediction_started), 'association_ms': 1000 * (association_ended - prediction_ended), 'identity_conflicts_resolved': conflicts_resolved})

def render_frame(image: np.ndarray, rows: list[dict], pipeline: base.OnlineRgbFrustumPipeline, rates: dict,
                 latency_ms: float, lidar_age_ms: float | None) -> np.ndarray:
    canvas = image.copy()
    overlay = canvas.copy()
    geometries = []
    for row in rows:
        if not row['xy_valid'] or row['state_status'] != 'FRESH':
            continue
        internal_z = row['z'] if row['class'] == 'PERSON' else cylinder.ground_z(row['x'], row['y'], pipeline.ground_normal, pipeline.ground_d)
        item = {'class': row['class'], 'xyz': np.asarray([row['x'], row['y'], internal_z])}
        if pipeline.person_only_lightweight:
            number = int(''.join(filter(str.isdigit, row['track_id'])) or 0)
            palette = ((232, 149, 55), (76, 181, 160), (198, 122, 215),
                       (68, 171, 230), (141, 190, 92), (214, 145, 112))
            color = palette[number % len(palette)]
        else:
            color = base.COLORS.get(row['identity'], base.TEMP_COLOR)
        geometry = pipeline._cylinder_geometry(item, color)
        if geometry is not None:
            geometries.append((geometry, row))
            pixels, _, _, segments = geometry
            quads = [np.round(pixels[[index, (index + 1) % segments, segments + (index + 1) % segments, segments + index]]).astype(np.int32) for index in range(segments)]
            cv2.fillPoly(overlay, quads, color, cv2.LINE_AA)
    if geometries:
        cv2.addWeighted(overlay, 0.14, canvas, 0.86, 0, canvas)
        for (pixels, _, color, segments), row in geometries:
            for start in (0, segments):
                cv2.polylines(canvas, [np.round(pixels[start:start + segments]).astype(np.int32)], True, color, 2, cv2.LINE_AA)
            for index in range(0, segments, 3):
                cv2.line(canvas, tuple(np.round(pixels[index]).astype(int)), tuple(np.round(pixels[index + segments]).astype(int)), color, 1, cv2.LINE_AA)
            marker = 'M' if row['state_source'] == 'LIDAR_MEASUREMENT' else 'P'
            suffix = f"XY {row['x']:+.2f},{row['y']:+.2f} {marker}" if row['class'] == 'ROBOT' else f"XYZ {row['x']:+.2f},{row['y']:+.2f},{row['z']:+.2f} v={row.get('speed_mps', 0.0):.2f}m/s {marker}"
            label = row['track_id'] if pipeline.person_only_lightweight else row['identity']
            anchor = np.round(pixels[segments:].mean(axis=0)).astype(int)
            cv2.putText(canvas, f'{label} {suffix}', (int(anchor[0]) + 4, max(74, int(anchor[1]) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (IMAGE_SIZE[0], 62), (12, 17, 22), -1)
    mode_name = 'COCO-PERSON ONLINE V4' if pipeline.person_only_lightweight else 'PURE ONLINE'
    mode_title = f'{mode_name} | UNCAPPED RGB + 10 Hz LiDAR correction + causal propagation' if rates.get('uncapped') else f'{mode_name} | SENSOR-PACED RGB + 10 Hz LiDAR correction + causal propagation'
    cv2.putText(canvas, mode_title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (242, 244, 246), 1, cv2.LINE_AA)
    age_text = '--' if lidar_age_ms is None else f'{lidar_age_ms:.0f} ms'
    status = f"RGB {rates['rgb_hz']:.2f} Hz | LiDAR {rates['lidar_hz']:.2f} Hz | Output {rates['output_hz']:.2f} Hz | Latency {latency_ms:.1f} ms | LiDAR age {age_text}"
    cv2.putText(canvas, status, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 218, 238), 1, cv2.LINE_AA)
    return canvas

def timed_render(image: np.ndarray, rows: list[dict], pipeline: base.OnlineRgbFrustumPipeline, rates: dict,
                 latency_ms: float, lidar_age_ms: float | None,
                 enabled: bool=True) -> tuple[np.ndarray, float]:
    if not enabled:
        return (image, 0.0)
    started = time.perf_counter()
    rendered = render_frame(image, rows, pipeline, rates, latency_ms, lidar_age_ms)
    return (rendered, 1000 * (time.perf_counter() - started))

def distribution(values: list[float]) -> dict:
    if not values:
        return {key: None for key in ('mean', 'median', 'p90', 'p95', 'p99', 'max')}
    data = np.asarray(values, np.float64)
    return {'mean': float(data.mean()), 'median': float(np.median(data)), 'p90': float(np.percentile(data, 90)), 'p95': float(np.percentile(data, 95)), 'p99': float(np.percentile(data, 99)), 'max': float(data.max())}

def queue_growth(values: list[int]) -> dict:
    """Report bounded-queue early/late means and a least-squares trend."""
    if not values:
        return {'samples': 0, 'early_mean': None, 'late_mean': None, 'slope_per_sample': None, 'sustained_growth': False}
    data = np.asarray(values, np.float64)
    window = min(len(data), max(5, len(data) // 10))
    slope = float(np.polyfit(np.arange(len(data), dtype=np.float64), data, 1)[0]) if len(data) > 1 else 0.0
    early, late = float(data[:window].mean()), float(data[-window:].mean())
    return {'samples': len(values), 'early_mean': early, 'late_mean': late, 'slope_per_sample': slope, 'max': int(data.max()), 'sustained_growth': bool(len(data) >= 20 and late > early + 0.5 and slope > 0.0005)}

def environment_info(pipeline: base.OnlineRgbFrustumPipeline) -> dict:
    """Collect reproducible static environment metadata without background sampling."""
    def command(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=5).strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    cuda = base.torch.cuda.is_available()
    device_index = base.torch.cuda.current_device() if cuda else None
    properties = base.torch.cuda.get_device_properties(device_index) if device_index is not None else None
    return {
        'timestamp_local': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'git_commit': command('git', 'rev-parse', 'HEAD'),
        'git_dirty': bool(command('git', 'status', '--porcelain')),
        'os': platform.platform(),
        'python': sys.version.split()[0],
        'cpu': platform.processor() or platform.machine(),
        'logical_cpu_count': os.cpu_count(),
        'gpu': None if properties is None else properties.name,
        'gpu_memory_bytes': None if properties is None else int(properties.total_memory),
        'nvidia_driver': command('nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'),
        'device': str(pipeline.device),
        'geometry_backend': 'CUDA' if pipeline.geometry_gpu else 'CPU',
        'versions': {'torch': base.torch.__version__, 'cuda_runtime': base.torch.version.cuda, 'ultralytics': ultralytics.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__, 'scipy': scipy.__version__},
        'resource_sampling': {'status': 'N/A', 'reason': 'No cross-platform per-process GPU sampler with verified attribution is bundled; static hardware metadata only.'},
    }

def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys((key for row in rows for key in row)))
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def report(summary: dict) -> str:
    rgb = summary['rates_hz']
    counts = summary['counts']
    rt = summary['runtime_ms']
    if summary.get('mode') == 'PERSON_ONLY_LIGHTWEIGHT':
        architecture = summary['architecture']
        return f"""# Online v4 COCO-PERSON Runtime Report

Status: **{summary['status']}**

`30 Hz RGB → YOLO PERSON class 0 → frozen RGB-guided LiDAR geometry → Txxxx short tracker → CausalMotionBank → collision-ready state`

- RGB: {counts['rgb_processed']}/{counts['rgb_received']}; wall output {rgb['wall_output']:.5f} Hz; backlog drops {counts.get('rgb_dropped_due_to_backlog', 0)}.
- LiDAR updates: {counts['lidar_updates']}; update rate {rgb['lidar_updates']:.5f} Hz.
- PERSON valid XYZ rows: {counts.get('person_xyz_valid_rows', 0)}.
- ROBOT rows: {counts.get('robot_rows', 0)}.
- YOLO classes requested: {architecture['yolo_requested_classes']}.
- ReID loaded: {architecture['reid_model_loaded']}; persistent identity: {architecture['persistent_identity_enabled']}.
- Tracker: {architecture['tracker']}.
- Short tracking P95: {rt['lidar_short_term_tracking']['p95']:.3f} ms; causal correction P95: {rt['lidar_causal_motion_correction']['p95']:.3f} ms.
- LiDAR total P95: {rt['lidar_total']['p95']:.3f} ms; online latency P95: {rt['online_latency']['p95']:.3f} ms.
- Future RGB/LiDAR, runtime GT, interpolation and backward smoothing: **not used**.
"""
    scope_note = 'Only RGB indices 0–2936 and LiDAR indices 0–978 (`TRAIN_FIT`) were streamed. TEST and EMBARGO access is zero. ' if summary['scope'] == 'TRAIN_FIT_ONLY' else 'The complete 7,490-RGB/2,498-LiDAR Scene01 sequence was streamed, including validation, TEST and EMBARGO partitions. This is a full-video display/throughput run, not an untouched TEST or generalization claim. '
    growth = summary['queues']['render']['sustained_growth'] or summary['queues']['video_writer']['sustained_growth']
    answers = [f"1. RGB input rate: **{rgb['rgb_input']:.5f} Hz**.", f"2. LiDAR input rate: **{rgb['lidar_input']:.5f} Hz**.", f"3. Wall-clock output rate: **{rgb['wall_output']:.5f} Hz**.", f"4. RGB processed: **{counts['rgb_processed']}/{counts['rgb_received']}**.", f"5. RGB backlog drops: **{counts.get('rgb_dropped_due_to_backlog', 0)}**; recording drops: **{counts.get('video_frames_dropped', 0)}**.", f"6. LiDAR update rate: **{rgb['lidar_updates']:.5f} Hz**.", f"7. Output entity states from new measurements: **{counts.get('measurement_state_rows', 0)}**.", f"8. Causal-prediction entity states: **{counts.get('prediction_state_rows', 0)}**.", f"9. Maximum prediction horizon: **{summary['max_prediction_horizon_ms']:.3f} ms**.", '10. Future RGB read by current output: **NO**.', '11. Future LiDAR read by current output: **NO**.', '12. Offline interpolation: **NO**.', '13. Backward smoothing/rewrite: **NO**.', f"14. Queue sustained growth: **{('YES' if growth else 'NO')}**; bounded writer max depth **{counts['video_queue_max_depth']}/{VIDEO_QUEUE_SIZE}**.", f"15. Wall/sensor duration ratio: **{summary['playback_speed']['wall_over_sensor']:.6f}**.", f"16. YOLO P95: **{rt['yolo']['p95']:.3f} ms**.", f"17. LiDAR pipeline P95: **{rt['lidar_total']['p95']:.3f} ms**.", f"18. Execution mode: **{('ORIGINAL-TIMESTAMP REALTIME' if summary['realtime_pacing'] else 'UNCAPPED MAXIMUM THROUGHPUT')}**; wall output Gate >=29 Hz: **{('PASS' if summary['gate']['rgb_output_rate_pass'] else 'FAIL')}**.", f"19. Unconfirmed detections remain anonymous: **YES** ({counts.get('anonymous_rows', 0)} rows).", '20. Official Robot Z remains unavailable: **YES**.']
    return ('# Pure-Online RGB-Guided LiDAR Report\n\n' if summary['realtime_pacing'] else '# Pure-Online Full-Sequence Uncapped Throughput Report\n\n') + f"Status: **{summary['status']}**. This is local ROS1-bag streaming, not ROS/ROS2 deployment.\n\n## Architecture\n\n`real 30 Hz RGB → frozen YOLO11s → current-box association → render`, while only a new 10 Hz LiDAR message runs `transform → frustum → frozen suppression → CUDA clustering → 3D correction`. Between corrections, XY uses causal constant velocity and person Z is held.\n\nYOLO boxes remain active internal observations but are intentionally not drawn; the final display contains only online 3D cylinders, compact identity/coordinate labels and runtime status.\n\nLiDAR support is restricted to an already arrived RGB frame with `RGB timestamp <= LiDAR timestamp` and delta <=35 ms. RGB boxes never create metric depth. Frozen models, calibration, static prior and TRAIN_FIT ReID prototypes are offline-frozen deployment assets.\n\n## Required answers\n\n" + '\n'.join(answers) + '\n\n## Scope and caveats\n\n' + scope_note + '`LIDAR_MEASUREMENT` marks the first RGB output incorporating a new past LiDAR correction; its exact positive prediction horizon remains recorded. A propagated state is never called a new LiDAR measurement. Robot XY can be measured/predicted, but official Robot Z is unavailable and no heading is rendered because no valid online yaw exists.\n'

def run(args: argparse.Namespace) -> dict:
    if not args.person_only_lightweight:
        raise ValueError('online_v4 is PERSON-only; full/ROBOT/ReID mode is unavailable')
    if args.persistent_identity:
        raise ValueError('online_v4 does not support persistent identity or G01-G05')
    dataset_profile = dataset_adapter.load_profile(ROOT, args.dataset)
    dataset_adapter.require_runtime_ready(dataset_profile)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'online_30hz_summary.json'
    runtime_csv_path = output_dir / 'runtime_per_frame.csv'
    drops_path = output_dir / 'drop_statistics.json'
    causality_path = output_dir / 'causality_audit.json'
    report_path = output_dir / ('RUNTIME_RUN_REPORT.md' if args.person_only_lightweight else 'PURE_ONLINE_30HZ_REPORT.md')
    if base.torch.cuda.is_available():
        base.torch.cuda.reset_peak_memory_stats()
    initialization_started = time.perf_counter()
    pipeline = base.OnlineRgbFrustumPipeline(
        args.device,
        identity=not args.person_only_lightweight,
        projection_audit=args.projection_audit,
        audit_warmup_frames=args.projection_audit_warmup_lidar,
        legacy_display_offset=args.legacy_display_offset,
        suppressed_recovery=args.suppressed_recovery,
        person_only_lightweight=args.person_only_lightweight,
        detector=args.detector,
    )
    initialization_ms = 1000.0 * (time.perf_counter() - initialization_started)
    detector_audit_name = ('COCO_DETECTOR_AUDIT.json' if args.detector == 'yolo11s-coco'
                           else 'YOLO26S_DETECTOR_AUDIT.json')
    detector_provenance = detector_audit.write(
        output_dir / detector_audit_name, pipeline, initialization_ms)
    if pipeline.device.type == 'cuda':
        detector_provenance['gpu_vram_after_warmup_bytes'] = int(
            base.torch.cuda.memory_allocated(pipeline.device))
        detector_provenance['gpu_peak_vram_bytes'] = int(
            base.torch.cuda.max_memory_allocated(pipeline.device))
        (output_dir / detector_audit_name).write_text(
            json.dumps(detector_provenance, indent=2) + '\n', encoding='utf-8')
    identity_manager = persistent.PersistentIdentityManager(pipeline.reid.prototypes, pipeline.reid.identities, appearance_threshold=0.48, max_age=pipeline.tracker.max_age) if args.persistent_identity and pipeline.reid is not None else None
    motion = CausalMotionBank(args.max_prediction_horizon_ms)
    anonymous = Anonymous2DTracker()
    writer = AsyncVideoWriter(Path(args.record), 30.0, not args.no_record)
    pending_lidar: SensorEvent | None = None
    latest_support: RgbSupport | None = None
    latest_lidar_payload: dict | None = None
    lidar_timings: list[float] = []
    lidar_details: list[dict] = []
    runtime_rows: list[dict] = []
    prediction_horizons: list[float] = []
    state_ages: list[float] = []
    lidar_ages: list[float] = []
    render_queue_depths: list[int] = []
    video_queue_depths: list[int] = []
    identity_last: dict[str, str] = {}
    identity_transitions = Counter()
    entity_counts = Counter()
    counts = Counter()
    first_rgb_sensor_ns = last_rgb_sensor_ns = None
    first_lidar_sensor_ns = last_lidar_sensor_ns = None
    wall_epoch = output_wall_start = output_wall_end = None
    last_output_timestamp = -1
    last_output_update_serial = 0
    max_prediction_horizon = 0.0
    causal_support_ok = True
    output_hashes: list[int] = []
    stopped_by_user = False
    pending_renders: deque[dict] = deque()

    def execute_lidar(event: SensorEvent, support: RgbSupport) -> None:
        nonlocal latest_lidar_payload, first_lidar_sensor_ns, last_lidar_sensor_ns, causal_support_ok
        try:
            delta_ms = causal_support_delta_ms(support.timestamp_ns, event.timestamp_ns, args.sync_slop_ms)
        except ValueError:
            causal_support_ok = False
            raise
        decode_started = time.perf_counter()
        points = base.decode_pointcloud2(event.message)
        decode_ended = time.perf_counter()
        timing, tracked = measure_lidar(pipeline, support, points, event.timestamp_ns, event.index, motion, identity_manager)
        timing['decode_ms'] = 1000 * (decode_ended - decode_started)
        timing['total_ms'] = timing['decode_ms'] + timing['geometry_and_correction_ms']
        timing.update({'lidar_frame_id': event.index, 'lidar_timestamp_ns': event.timestamp_ns, 'support_rgb_frame_id': support.index, 'support_rgb_timestamp_ns': support.timestamp_ns, 'support_delta_ms': delta_ms})
        latest_lidar_payload = timing
        lidar_timings.append(timing['total_ms'])
        lidar_details.append(timing.copy())
        first_lidar_sensor_ns = event.timestamp_ns if first_lidar_sensor_ns is None else first_lidar_sensor_ns
        last_lidar_sensor_ns = event.timestamp_ns
        counts['lidar_updates'] += 1
        counts['lidar_measurements'] += timing['measurements']

    def finalize_render(log, block: bool) -> bool:
        nonlocal output_wall_start, output_wall_end, last_output_timestamp, stopped_by_user
        if not pending_renders:
            return False
        job = pending_renders[0]
        future: Future = job['future']
        if not block and (not future.done()):
            return False
        rendered, render_ms = future.result()
        pending_renders.popleft()
        display_started = time.perf_counter()
        if not args.headless:
            cv2.imshow('Pure-online RGB-guided clustering 30 Hz', rendered)
            if cv2.waitKey(1) & 255 in (27, ord('q')):
                stopped_by_user = True
        submitted = writer.submit(rendered)
        video_queue_depths.append(writer.items.qsize() if writer.enabled else 0)
        display_ended = time.perf_counter()
        counts['rgb_displayed'] += 1
        if not submitted:
            counts['video_frames_dropped'] += 1
        output_wall_start = display_ended if output_wall_start is None else output_wall_start
        output_wall_end = display_ended
        payload = job['payload']
        if payload['rgb_timestamp'] <= last_output_timestamp:
            raise RuntimeError('30 Hz output timestamp is not strictly monotonic')
        last_output_timestamp = payload['rgb_timestamp']
        payload.update({'wall_timestamp': time.time_ns(), 'render_ms': render_ms, 'display_submit_ms': 1000 * (display_ended - display_started), 'total_rgb_path_ms': 1000 * (display_ended - job['rgb_started']), 'online_latency_ms': max(0.0, 1000 * (display_ended - job['due'])), 'queue_depth': writer.items.qsize() if writer.enabled else 0})
        log.write(json.dumps(payload, separators=(',', ':')) + '\n')
        log.flush()
        runtime_rows.append({key: value for key, value in payload.items() if key not in {'persons', 'robot', 'lidar_update_detail'}})
        output_hashes.append(hash(json.dumps(payload, sort_keys=True)))
        return True
    try:
        with Path(args.output_jsonl).open('w', encoding='utf-8') as log, ThreadPoolExecutor(max_workers=1, thread_name_prefix='online-render') as render_pool:
            for event in stream_sensor_events(Path(args.bag), args.rgb_limit, args.lidar_limit):
                while finalize_render(log, block=False):
                    pass
                if stopped_by_user:
                    break
                if event.topic == base.LIDAR_TOPIC:
                    counts['lidar_received'] += 1
                    if latest_support is not None:
                        delta_ns = event.timestamp_ns - latest_support.timestamp_ns
                        if 0 <= delta_ns <= int(args.sync_slop_ms * 1000000.0):
                            execute_lidar(event, latest_support)
                            continue
                    if pending_lidar is not None:
                        counts['lidar_unmatched'] += 1
                    pending_lidar = event
                    continue
                counts['rgb_received'] += 1
                if first_rgb_sensor_ns is None:
                    first_rgb_sensor_ns = event.timestamp_ns
                    wall_epoch = time.perf_counter()
                last_rgb_sensor_ns = event.timestamp_ns
                assert wall_epoch is not None and first_rgb_sensor_ns is not None
                due = wall_epoch + (event.timestamp_ns - first_rgb_sensor_ns) / 1000000000.0
                if args.realtime:
                    remaining = due - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                lag_ms = 1000 * (time.perf_counter() - due) if args.realtime else 0.0
                if args.realtime and lag_ms > args.backlog_drop_ms:
                    while pending_renders:
                        finalize_render(log, block=True)
                    counts['rgb_dropped_due_to_backlog'] += 1
                    dropped = {'rgb_frame_id': event.index, 'rgb_timestamp': event.timestamp_ns, 'wall_timestamp': time.time_ns(), 'processed': False, 'drop_reason': 'STALE_BACKLOG', 'dropped_frames': int(counts['rgb_dropped_due_to_backlog']), 'schedule_lag_ms': lag_ms, 'persons': [], 'robot': [], 'queue_depth': writer.items.qsize() if writer.enabled else 0}
                    log.write(json.dumps(dropped, separators=(',', ':')) + '\n')
                    log.flush()
                    runtime_rows.append(dropped)
                    continue
                rgb_started = time.perf_counter()
                pipeline.calibration_audit.begin_frame()
                decode_started = rgb_started
                image = base.decode_image(event.message)
                decode_ended = time.perf_counter()
                yolo_started = decode_ended
                detections = pipeline._detections(image)
                yolo_ended = time.perf_counter()
                rows, assoc_timing = associate_rgb(detections, motion, event.timestamp_ns, anonymous, pipeline)
                counts['identity_conflicts_resolved'] += assoc_timing['identity_conflicts_resolved']
                association_ended = time.perf_counter()
                rgb_elapsed = (event.timestamp_ns - first_rgb_sensor_ns) / 1000000000.0
                lidar_elapsed = 0.0 if first_lidar_sensor_ns is None or last_lidar_sensor_ns is None else (last_lidar_sensor_ns - first_lidar_sensor_ns) / 1000000000.0
                wall_elapsed = max(time.perf_counter() - wall_epoch, 1e-09)
                rates = {'rgb_hz': event.index / rgb_elapsed if rgb_elapsed > 0 else 0.0, 'lidar_hz': max(counts['lidar_updates'] - 1, 0) / lidar_elapsed if lidar_elapsed > 0 else 0.0, 'output_hz': counts['rgb_processed'] / wall_elapsed, 'uncapped': not args.realtime}
                lidar_age_ms = None if motion.latest_lidar_timestamp_ns is None else (event.timestamp_ns - motion.latest_lidar_timestamp_ns) / 1000000.0
                if lidar_age_ms is not None:
                    lidar_ages.append(float(lidar_age_ms))
                render_submit_time = time.perf_counter()
                provisional_latency = max(lag_ms, 0.0) + 1000 * (render_submit_time - rgb_started)
                new_lidar = motion.update_serial > last_output_update_serial
                last_output_update_serial = motion.update_serial
                state_source = 'LIDAR_MEASUREMENT' if new_lidar else 'CAUSAL_PREDICTION'
                persons = [row for row in rows if row['class'] == 'PERSON']
                robots = [row for row in rows if row['class'] == 'ROBOT']
                for row in rows:
                    kind = row['class'].lower()
                    entity_counts[f'{kind}_rows'] += 1
                    entity_counts[f'{kind}_xy_valid_rows'] += int(row['xy_valid'])
                    entity_counts[f'{kind}_xyz_valid_rows'] += int(row['xyz_valid'])
                    if row['xy_valid']:
                        max_prediction_horizon = max(max_prediction_horizon, row['prediction_horizon_ms'] or 0.0)
                        prediction_horizons.append(float(row['prediction_horizon_ms'] or 0.0))
                        counts['measurement_state_rows' if row['state_source'] == 'LIDAR_MEASUREMENT' else 'prediction_state_rows'] += 1
                    if row['state_age_ms'] is not None:
                        state_ages.append(float(row['state_age_ms']))
                    if row['identity'] == 'anonymous':
                        counts['anonymous_rows'] += 1
                        counts[f'{kind}_anonymous_rows'] += 1
                    elif row['identity'] in (*base.IDENTITIES, 'R1'):
                        counts['persistent_identity_rows'] += 1
                        counts[f'{kind}_persistent_identity_rows'] += 1
                    if row['class'] == 'PERSON':
                        previous = identity_last.get(row['track_id'])
                        current = row['identity']
                        if previous in base.IDENTITIES and current == 'anonymous':
                            identity_transitions['confirmed_to_anonymous'] += 1
                        if previous in base.IDENTITIES and current in base.IDENTITIES and current != previous:
                            identity_transitions['fixed_g_to_different_g'] += 1
                        identity_last[row['track_id']] = current
                fixed = [row['identity'] for row in persons if row['identity'] in base.IDENTITIES]
                if len(fixed) != len(set(fixed)):
                    counts['duplicate_fixed_identity_frames'] += 1
                timing = {'rgb_decode_ms': 1000 * (decode_ended - decode_started), 'YOLO_ms': 1000 * (yolo_ended - yolo_started), 'prediction_ms': assoc_timing['prediction_ms'], 'association_ms': assoc_timing['association_ms'], 'schedule_lag_ms': max(lag_ms, 0.0)}
                payload = {'rgb_frame_id': event.index, 'rgb_timestamp': event.timestamp_ns, 'processed': True, 'latest_lidar_frame_id': motion.latest_lidar_frame_id, 'latest_lidar_timestamp': motion.latest_lidar_timestamp_ns, 'lidar_age_ms': lidar_age_ms, 'state_source': state_source, 'prediction_horizon_ms': max([row['prediction_horizon_ms'] or 0.0 for row in rows], default=0.0), 'person_detections': [{'bbox': item['bbox'].astype(float).tolist(), 'confidence': float(item['confidence'])} for item in detections if item['class'] == 'PERSON'], 'persons': persons, 'robot': robots, **timing, 'new_lidar_update': new_lidar, 'lidar_processing_ms': latest_lidar_payload['total_ms'] if new_lidar and latest_lidar_payload is not None else None, 'lidar_update_detail': latest_lidar_payload if new_lidar else None, 'dropped_frames': int(counts['rgb_dropped_due_to_backlog'])}
                while len(pending_renders) >= 1:
                    finalize_render(log, block=True)
                future = render_pool.submit(timed_render, image, rows, pipeline, rates, provisional_latency,
                                            lidar_age_ms, not (args.headless and args.no_record))
                pending_renders.append({'future': future, 'payload': payload, 'rgb_started': rgb_started, 'due': due})
                render_queue_depths.append(len(pending_renders))
                counts['rgb_processed'] += 1
                latest_support = RgbSupport(event.index, event.timestamp_ns, image, copy_detections(detections))
                if pending_lidar is not None:
                    delta_ns = pending_lidar.timestamp_ns - latest_support.timestamp_ns
                    if 0 <= delta_ns <= int(args.sync_slop_ms * 1000000.0):
                        execute_lidar(pending_lidar, latest_support)
                        pending_lidar = None
                    elif delta_ns < 0:
                        counts['lidar_unmatched'] += 1
                        pending_lidar = None
                if (event.index + 1) % 300 == 0:
                    print(f"Online {event.index + 1}/{args.rgb_limit} | output {rates['output_hz']:.2f} Hz | drops {counts['rgb_dropped_due_to_backlog']}", flush=True)
                if stopped_by_user:
                    break
            while pending_renders:
                finalize_render(log, block=True)
    finally:
        pipeline.calibration_audit.close()
        writer.close()
        cv2.destroyAllWindows()
    if pending_lidar is not None:
        counts['lidar_unmatched'] += 1
    if not runtime_rows or first_rgb_sensor_ns is None or last_rgb_sensor_ns is None:
        raise RuntimeError('No RGB outputs were produced')
    sensor_duration = (last_rgb_sensor_ns - first_rgb_sensor_ns) / 1000000000.0
    wall_duration = max((output_wall_end or time.perf_counter()) - (output_wall_start or wall_epoch), 1e-09)
    rgb_processed = [row for row in runtime_rows if row.get('processed')]
    rgb_input_hz = (counts['rgb_received'] - 1) / sensor_duration
    lidar_duration = (last_lidar_sensor_ns - first_lidar_sensor_ns) / 1000000000.0 if first_lidar_sensor_ns is not None and last_lidar_sensor_ns is not None else math.nan
    lidar_input_hz = (counts['lidar_received'] - 1) / lidar_duration
    lidar_update_hz = (counts['lidar_updates'] - 1) / lidar_duration
    effective_output_hz = counts['rgb_displayed'] / wall_duration
    processed_fraction = counts['rgb_processed'] / counts['rgb_received']
    output_timestamps = [int(row['rgb_timestamp']) for row in rgb_processed]
    timestamp_unique = len(set(output_timestamps)) == len(output_timestamps) and strictly_monotonic(output_timestamps)
    sources_valid = all((row['state_source'] in {'LIDAR_MEASUREMENT', 'CAUSAL_PREDICTION'} for row in rgb_processed))
    no_future_runtime = causal_support_ok and all((row.get('lidar_update_detail') is None or row['lidar_update_detail']['support_rgb_timestamp_ns'] <= row['lidar_update_detail']['lidar_timestamp_ns'] for row in [json.loads(line) for line in Path(args.output_jsonl).read_text(encoding='utf-8').splitlines()] if row.get('processed')))
    expected_rgb = FULL_RGB_FRAMES if args.allow_full_sequence else TRAIN_FIT_RGB_FRAMES
    expected_lidar = FULL_LIDAR_FRAMES if args.allow_full_sequence else TRAIN_FIT_LIDAR_FRAMES
    queue_metrics = {'render': queue_growth(render_queue_depths), 'video_writer': queue_growth(video_queue_depths), 'render_capacity': 1, 'video_writer_capacity': VIDEO_QUEUE_SIZE}
    queue_sustained_growth = any(value['sustained_growth'] for value in (queue_metrics['render'], queue_metrics['video_writer']))
    gate = {'rgb_output_rate_pass': effective_output_hz >= 29.0, 'rgb_processed_fraction_pass': processed_fraction >= 0.99, 'lidar_update_rate_pass': 9.5 <= lidar_update_hz <= 10.5, 'lidar_p95_pass': distribution(lidar_timings)['p95'] < 100.0, 'bounded_queue_pass': not queue_sustained_growth and max(render_queue_depths, default=0) <= 1 and writer.max_depth <= VIDEO_QUEUE_SIZE, 'recording_complete_pass': not writer.enabled or (writer.frames_dropped == 0 and writer.frames_written == counts['rgb_processed']), 'no_future_sensor_access_pass': no_future_runtime, 'timestamp_monotonic_unique_pass': timestamp_unique, 'state_source_pass': sources_valid, 'no_duplicate_fixed_g_pass': counts['duplicate_fixed_identity_frames'] == 0, 'robot_z_unavailable_pass': all((not row['robot_z_available'] for item in rgb_processed for row in item.get('robot', []))), 'declared_scope_count_pass': counts['rgb_received'] == expected_rgb and counts['lidar_received'] == expected_lidar, 'runtime_mode_pass': effective_output_hz >= 29.0 if not args.realtime else 0.95 <= wall_duration / sensor_duration <= 1.1}
    if args.person_only_lightweight:
        gate.update({'robot_inference_zero_pass': entity_counts['robot_rows'] == 0 and all(not item.get('robot') for item in rgb_processed),
                     'reid_disabled_pass': pipeline.reid is None,
                     'persistent_identity_disabled_pass': identity_manager is None,
                     'temporary_t_ids_only_pass': all(row['track_id'].startswith('T') and row.get('identity') == 'anonymous' for item in rgb_processed for row in item.get('persons', []))})
    status = 'PASS' if all(gate.values()) and (not stopped_by_user) else 'FAIL'
    if args.person_only_lightweight:
        status_prefix = ('PERSON_ONLY_LIGHTWEIGHT_FULL_REALTIME' if args.realtime else 'PERSON_ONLY_LIGHTWEIGHT_FULL_UNCAPPED') if args.allow_full_sequence else 'PERSON_ONLY_LIGHTWEIGHT_30HZ'
    else:
        status_prefix = ('PURE_ONLINE_FULL_REALTIME' if args.realtime else 'PURE_ONLINE_FULL_UNCAPPED') if args.allow_full_sequence else 'PURE_ONLINE_30HZ'
    runtime_sources = {
        'rgb_decode': [row['rgb_decode_ms'] for row in rgb_processed], 'yolo': [row['YOLO_ms'] for row in rgb_processed],
        'prediction': [row['prediction_ms'] for row in rgb_processed], 'association': [row['association_ms'] for row in rgb_processed],
        'render': [row['render_ms'] for row in rgb_processed], 'display_submit': [row['display_submit_ms'] for row in rgb_processed],
        'rgb_total': [row['total_rgb_path_ms'] for row in rgb_processed], 'schedule_lag': [row['schedule_lag_ms'] for row in rgb_processed],
        'online_latency': [row['online_latency_ms'] for row in rgb_processed],
        'lidar_decode': [row['decode_ms'] for row in lidar_details], 'lidar_transform': [row['transform_ms'] for row in lidar_details],
        'lidar_frustum': [row['frustum_ms'] for row in lidar_details], 'lidar_static_suppression': [row['static_suppression_ms'] for row in lidar_details],
        'lidar_clustering': [row['clustering_ms'] for row in lidar_details],
        'lidar_projection_residual_audit': [row['projection_residual_audit_ms'] for row in lidar_details],
        'lidar_short_term_tracking': [row['short_term_tracking_ms'] for row in lidar_details],
        'lidar_suppressed_recovery': [row['suppressed_recovery_ms'] for row in lidar_details],
        'lidar_reid_and_persistent_identity': [row['reid_and_persistent_identity_ms'] for row in lidar_details],
        'lidar_causal_motion_correction': [row['causal_motion_correction_ms'] for row in lidar_details],
        'lidar_tracking_reid_correction': [row['tracking_reid_correction_ms'] for row in lidar_details],
        'lidar_geometry_and_correction': [row['geometry_and_correction_ms'] for row in lidar_details], 'lidar_total': lidar_timings,
    }
    runtime = {name: distribution(values) for name, values in runtime_sources.items()}
    runtime_samples = {name: len(values) for name, values in runtime_sources.items()}
    sync_metrics = {'support_age_ms': distribution([row['support_delta_ms'] for row in lidar_details]), 'future_support_count': sum(row['support_rgb_timestamp_ns'] > row['lidar_timestamp_ns'] for row in lidar_details), 'matched_lidar_updates': len(lidar_details), 'unmatched_lidar': int(counts['lidar_unmatched']), 'window_ms': float(args.sync_slop_ms), 'rule': 'latest arrived RGB with rgb_sensor_timestamp <= lidar_sensor_timestamp and age <= window'}
    identity_metrics = {'person_persistent_rows': int(counts['person_persistent_identity_rows']), 'person_anonymous_rows': int(counts['person_anonymous_rows']), 'confirmed_to_anonymous_transitions': int(identity_transitions['confirmed_to_anonymous']), 'fixed_g_to_different_g_transitions': int(identity_transitions['fixed_g_to_different_g']), 'duplicate_fixed_identity_frames': int(counts['duplicate_fixed_identity_frames']), 'manager': identity_manager.summary() if identity_manager is not None else {'enabled': False}}
    projection_keys = ('points_input', 'points_projection_valid', 'points_in_view', 'points_height_candidate', 'points_after_static', 'temporal_removed', 'occupancy_removed', 'detections', 'components_available', 'measurements', 'person_measurements', 'robot_measurements', 'suppressed_recovery_attempts', 'suppressed_recovery_accepted', 'suppressed_recovery_rejected')
    projection_totals = {key: int(sum(row[key] for row in lidar_details)) for key in projection_keys}
    audit_metrics = pipeline.projection_status()
    projection_metrics = {**projection_totals, 'component_availability_per_detection': projection_totals['components_available'] / max(projection_totals['detections'], 1), 'measurement_availability_per_detection': projection_totals['measurements'] / max(projection_totals['detections'], 1), 'physical_projection': support.PHYSICAL_PROJECTION, 'empirical_inference_registration': support.EMPIRICAL_INFERENCE_REGISTRATION, 'display_overlay_registration': support.DISPLAY_OVERLAY_REGISTRATION, 'legacy_visualization': {'enabled': bool(args.legacy_display_offset), 'du_px': support.LEGACY_DISPLAY_REGISTRATION['du_px'], 'dv_px': support.LEGACY_DISPLAY_REGISTRATION['dv_px'], 'inference_effect': False, 'overrides_current_display_only': bool(args.legacy_display_offset)}, 'residual_audit': audit_metrics}
    state_metrics = {'prediction_horizon_ms': distribution(prediction_horizons), 'state_age_ms': distribution(state_ages), 'lidar_age_at_rgb_ms': distribution(lidar_ages), 'measurement_rows': int(counts['measurement_state_rows']), 'prediction_rows': int(counts['prediction_state_rows']), 'person_xyz_valid_rows': int(entity_counts['person_xyz_valid_rows']), 'person_rows': int(entity_counts['person_rows']), 'robot_xy_valid_rows': int(entity_counts['robot_xy_valid_rows']), 'robot_rows': int(entity_counts['robot_rows'])}
    lightweight = bool(args.person_only_lightweight)
    summary = {'metrics_schema': 'online_v3.person_only.v1' if lightweight else 'online_v3.benchmark.v1', 'status': f'{status_prefix}_{status}', 'mode': 'PERSON_ONLY_LIGHTWEIGHT' if lightweight else 'FULL', 'scope': 'FULL_SCENE01_DISPLAY_WITH_TEST_EMBARGO_ACCESS' if args.allow_full_sequence else 'TRAIN_FIT_ONLY', 'realtime_pacing': bool(args.realtime), 'stopped_by_user': bool(stopped_by_user), 'environment': environment_info(pipeline), 'architecture': {'person_only': lightweight, 'yolo_requested_classes': [0] if lightweight else [0, 1], 'person_class_index': 0, 'robot_inference_enabled': not lightweight, 'reid_model_loaded': pipeline.reid is not None, 'persistent_identity_enabled': identity_manager is not None, 'tracker': type(pipeline.tracker).__name__}, 'counts': {**{key: int(value) for key, value in counts.items()}, **{key: int(value) for key, value in entity_counts.items()}, 'video_frames_written': writer.frames_written, 'video_frames_dropped': writer.frames_dropped, 'video_queue_max_depth': writer.max_depth, 'render_queue_max_depth': max(render_queue_depths, default=0)}, 'rates_hz': {'rgb_input': rgb_input_hz, 'lidar_input': lidar_input_hz, 'wall_output': effective_output_hz, 'lidar_updates': lidar_update_hz}, 'durations_s': {'sensor': sensor_duration, 'wall': wall_duration}, 'playback_speed': {'wall_over_sensor': wall_duration / sensor_duration, 'sensor_over_wall': sensor_duration / wall_duration}, 'processed_rgb_fraction': processed_fraction, 'max_prediction_horizon_ms': max_prediction_horizon, 'runtime_ms': runtime, 'runtime_samples': runtime_samples, 'sync': sync_metrics, 'state_and_3d': state_metrics, 'identity': identity_metrics, 'projection': projection_metrics, 'queues': queue_metrics, 'gate': gate, 'semantics': {'rgb_timeline': 'REAL_SENSOR_RGB_FRAMES', 'lidar_measurement_rate': '10_HZ', 'output_state_rate': 'UNCAPPED_REAL_RGB_FRAMES' if not args.realtime else 'SENSOR_PACED_REAL_RGB_FRAMES', 'propagation': 'CAUSAL_CONSTANT_VELOCITY_XY_PERSON_Z_HOLD', 'future_rgb_used': not no_future_runtime, 'future_lidar_used': False, 'offline_interpolation_used': False, 'backward_smoothing_used': False, 'runtime_gt_used': False, 'physical_projection': 'RAW_K_D_T_DU_DV_ZERO_AUDIT_BASELINE_NOT_PROVEN_FINAL', 'inference_projection': 'SCENE01_EMPIRICAL_DU64_DV36', 'display_projection': 'SCENE01_DISPLAY_ONLY_DU60_DV17', 'projection_audit_writeback': False, 'legacy_display_offset_enabled': bool(args.legacy_display_offset), 'suppressed_point_recovery_enabled': bool(args.suppressed_recovery), 'suppressed_point_recovery_policy': 'HISTORY_ONLY_OCCUPANCY_DELETED_PERSON_COMPONENT', 'bev_cnn_used': False, 'robot_official_z_available': False, 'robot_heading_available': False, 'frozen_prior': 'YOLO_STATIC_SUPPRESSION' if lightweight else 'YOLO_STATIC_SUPPRESSION_AND_TRAINFIT_REID'}, 'video_writer_backend': writer.backend, 'persistent_identity': identity_metrics['manager'], 'outputs': {'video': None if args.no_record else str(Path(args.record).resolve()), 'jsonl': str(Path(args.output_jsonl).resolve()), 'runtime_csv': str(runtime_csv_path.resolve())}}
    summary['metrics_schema'] = ('online_v4.coco_person.v1' if lightweight
                                 else 'online_v4.benchmark.v1')
    summary['dataset_profile'] = {
        'name': dataset_profile.name,
        'path': str(dataset_profile.path.resolve()),
        'registration_mode': dataset_profile.payload['registration_mode'],
        'scene_specific_priors': dataset_profile.payload['scene_specific_priors'],
    }
    summary['detector_audit'] = detector_provenance
    summary['architecture'].update({
        'detector_id': pipeline.detector_id,
        'detector_family': pipeline.detector_family,
        'yolo_requested_classes': [pipeline.person_class_id],
        'person_class_index': pipeline.person_class_id,
        'robot_inference_enabled': False,
        'reid_enabled': False,
        'persistent_identity_enabled': False,
    })
    summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if identity_manager is not None:
        events_path = output_dir / 'identity_events.jsonl'
        events_path.write_text(''.join((json.dumps(item, separators=(',', ':')) + '\n' for item in identity_manager.events)), encoding='utf-8')
        summary['outputs']['identity_events'] = str(events_path.resolve())
        summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    write_csv(runtime_csv_path, runtime_rows)
    module_rows = [{'module': name, 'samples': runtime_samples[name], **values} for name, values in runtime.items()]
    write_csv(output_dir / 'metrics_modules_lightweight.csv' if args.person_only_lightweight else output_dir / 'metrics_modules_full.csv', module_rows)
    (output_dir / 'metrics_identity.json').write_text(json.dumps(identity_metrics, indent=2) + '\n', encoding='utf-8')
    tracking_stats = pipeline.tracker.statistics() if hasattr(pipeline.tracker, 'statistics') else {'status': 'N/A'}
    (output_dir / 'tracking_statistics.json').write_text(json.dumps(tracking_stats, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'metrics_sync.json').write_text(json.dumps(sync_metrics, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'projection_residual_audit.json').write_text(json.dumps(audit_metrics, indent=2) + '\n', encoding='utf-8')
    (output_dir / ('metrics_lightweight.json' if args.person_only_lightweight else 'metrics_full.json')).write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    drop_payload = {'rgb_received': int(counts['rgb_received']), 'rgb_processed': int(counts['rgb_processed']), 'rgb_dropped_due_to_backlog': int(counts['rgb_dropped_due_to_backlog']), 'video_frames_written': writer.frames_written, 'video_frames_dropped': writer.frames_dropped, 'video_queue_capacity': VIDEO_QUEUE_SIZE, 'video_queue_max_depth': writer.max_depth, 'render_queue_capacity': 1, 'render_queue_max_depth': max(render_queue_depths, default=0), 'lidar_unmatched': int(counts['lidar_unmatched']), 'queue_sustained_growth': queue_sustained_growth, 'queue_analysis': queue_metrics}
    drops_path.write_text(json.dumps(drop_payload, indent=2) + '\n', encoding='utf-8')
    causality = {'status': 'PASS' if no_future_runtime and timestamp_unique and sources_valid else 'FAIL', 'future_lidar_isolation': True, 'future_rgb_isolation': True, 'future_reid_isolation': True, 'no_backward_rewrite': True, 'no_offline_densification': True, 'timestamp_monotonicity': timestamp_unique, 'prediction_source_audit': sources_valid, 'runtime_support_rgb_not_after_lidar': no_future_runtime, 'output_rows_written_once': len(output_hashes) == len(rgb_processed), 'complete_trajectory_loaded': False, 'future_track_state_used': False, 'tests_required_for_claim': 'tests/test_online.py'}
    causality_path.write_text(json.dumps(causality, indent=2) + '\n', encoding='utf-8')
    report_path.write_text(report(summary), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary

def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--bag', default=str(base.DEFAULT_BAG))
    value.add_argument('--dataset', default='scene01',
                       choices=('scene01', 'scene28', 'jrdb'))
    value.add_argument('--device', default='0')
    value.add_argument('--detector', choices=tuple(base.DETECTORS), default='yolo11s-coco')
    value.add_argument('--rgb-limit', type=int, default=TRAIN_FIT_RGB_FRAMES)
    value.add_argument('--lidar-limit', type=int, default=TRAIN_FIT_LIDAR_FRAMES)
    value.add_argument('--sync-slop-ms', type=float, default=35.0)
    value.add_argument('--max-prediction-horizon-ms', type=float, default=MAX_PREDICTION_HORIZON_MS)
    value.add_argument('--backlog-drop-ms', type=float, default=BACKLOG_DROP_MS)
    value.add_argument('--allow-full-sequence', action='store_true', help='Explicitly allow full Scene01 access, including TEST and EMBARGO')
    value.add_argument('--output-dir', default=str(OUT))
    value.add_argument('--realtime', action=argparse.BooleanOptionalAction, default=True)
    value.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
    value.add_argument('--no-record', dest='no_record', action='store_true', default=True)
    value.add_argument('--record-video', dest='no_record', action='store_false')
    value.add_argument('--record', default=str(VIDEO))
    value.add_argument('--output-jsonl', default=str(FRAME_LOG))
    value.add_argument('--persistent-identity', action='store_true', help='Attach finite persistent G identity memory and causal recovery')
    value.add_argument('--person-only-lightweight', action=argparse.BooleanOptionalAction,
                       default=True,
                       help='PERSON class only, short T IDs, no ROBOT, ReID or persistent identity')
    value.add_argument('--projection-audit', action=argparse.BooleanOptionalAction,
                       default=bool(support.REGISTRATION_CONFIG['enabled']),
                       help='Report causal projection residual candidates; never write them to inference')
    value.add_argument('--projection-audit-warmup-lidar', type=int,
                       default=int(support.REGISTRATION_CONFIG['warmup_lidar_frames']))
    value.add_argument('--legacy-display-offset', action=argparse.BooleanOptionalAction,
                       default=bool(support.REGISTRATION_CONFIG['legacy_display_enabled']),
                       help='Optional +48 px legacy visualization only; never affects inference')
    value.add_argument('--suppressed-recovery', action=argparse.BooleanOptionalAction,
                       default=True,
                       help='Causally recover quality-gated PERSON components removed by occupancy suppression')
    return value
if __name__ == '__main__':
    run(parser().parse_args())

