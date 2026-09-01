"""Causal online RGB-guided LiDAR frustum clustering for NavWareSet Scene01.

The executable consumes camera and LiDAR messages in timestamp order.  A LiDAR
message is held for at most ``sync_slop_ms`` of arrival time so the latest
already-arrived RGB frame at or before the LiDAR sensor timestamp can be selected.
Processing after synchronization is single-frame and causal: YOLO,
projection, static suppression, clustering, ReID and tracking never read a
future state or runtime ground truth.

The currently testable transport is ROS1-bag real-time replay.  The processing
engine is transport independent; a hardware ROS adapter can call ``process``
with the same BGR image and PointCloud2 XYZ array.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rosbags.highlevel import AnyReader
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parents[2]
from . import recovery, registration, support, suppression, tracking
offline = cylinder = frustum = support
DEFAULT_BAG = Path('D:\\detection\\01_grs\\1_grs.bag')
DETECTORS = {
    'yolo11s-coco': {
        'path': ROOT / 'assets/models/detector/yolo11s_coco.pt',
        'official_source': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt',
        'family': 'official_ultralytics_yolo11s_coco',
    },
    'yolo26s-coco': {
        'path': ROOT / 'assets/models/detector/yolo26s.pt',
        'official_source': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt',
        'family': 'official_ultralytics_yolo26s_coco',
    },
}
MODEL = DETECTORS['yolo11s-coco']['path']
PROTOTYPES = ROOT / 'assets/priors/trainfit_fixed_identity_prototypes.npz'
REID_WEIGHTS = ROOT / 'assets/models/reid/osnet_x1_0_msmt17_combineall.pth'
GROUND_NORMAL, GROUND_D = (support.GROUND_NORMAL, support.GROUND_D)
OUT = ROOT / 'outputs/runtime'
IMAGE_TOPIC = '/camera/color/image_raw'
LIDAR_TOPIC = '/rslidar_points'
IMAGE_SIZE = (1280, 720)
IDENTITIES = ('G01', 'G02', 'G03', 'G04', 'G05')
COLORS = {'G01': (178, 114, 0), 'G02': (0, 159, 230), 'G03': (115, 158, 0), 'G04': (167, 121, 204), 'G05': (0, 94, 213), 'R1': (146, 77, 15)}
TEMP_COLOR = (210, 210, 210)
_POINT_TYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}

def reid_crop_tensor(image: np.ndarray, box: list[float]) -> tuple[torch.Tensor, np.ndarray]:
    """Frozen OSNet input and color descriptor for one xywh person crop."""
    x, y, w, h = box
    margin_x, margin_y = (0.04 * w, 0.03 * h)
    x1, y1 = (max(0, int(x - margin_x)), max(0, int(y - margin_y)))
    x2 = min(image.shape[1], int(math.ceil(x + w + margin_x)))
    y2 = min(image.shape[0], int(math.ceil(y + h + margin_y)))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((256, 128, 3), np.uint8)
    resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    tensor = (tensor - mean) / std
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histograms = []
    for part in np.array_split(hsv, 3, axis=0):
        hist = cv2.calcHist([part], [0, 1], None, [8, 8], [0, 180, 0, 256]).reshape(-1)
        hist /= max(float(np.linalg.norm(hist)), 1e-08)
        histograms.append(hist)
    return (tensor, np.concatenate(histograms).astype(np.float32))

def timestamp_ns(message: object, fallback: int) -> int:
    header = getattr(message, 'header', None)
    stamp = getattr(header, 'stamp', None)
    if stamp is None:
        return int(fallback)
    return int(stamp.sec) * 1000000000 + int(stamp.nanosec)

def decode_image(message: object) -> np.ndarray:
    width, height, step = (int(message.width), int(message.height), int(message.step))
    encoding = str(message.encoding).lower()
    channels = {'rgb8': 3, 'bgr8': 3, 'mono8': 1}.get(encoding)
    if channels is None:
        raise RuntimeError(f'Unsupported online image encoding: {message.encoding}')
    raw = np.asarray(message.data, dtype=np.uint8)
    expected = step * height
    if raw.size != expected:
        raise RuntimeError(f'Image byte count {raw.size} != {expected}')
    rows = raw.reshape(height, step)[:, :width * channels]
    image = rows.reshape(height, width, channels) if channels > 1 else rows.reshape(height, width)
    if encoding == 'rgb8':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif encoding == 'mono8':
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(image)

def decode_pointcloud2(message: object) -> np.ndarray:
    endian = '>' if bool(message.is_bigendian) else '<'
    names, formats, offsets = ([], [], [])
    for field in message.fields:
        datatype, count = (int(field.datatype), int(field.count))
        if datatype in _POINT_TYPES and count == 1:
            names.append(str(field.name))
            formats.append(endian + _POINT_TYPES[datatype])
            offsets.append(int(field.offset))
    if not {'x', 'y', 'z'}.issubset(names):
        raise RuntimeError('PointCloud2 is missing x/y/z')
    dtype = np.dtype({'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': int(message.point_step)})
    raw = np.asarray(message.data, dtype=np.uint8).tobytes()
    width, height = (int(message.width), int(message.height))
    if len(raw) != int(message.row_step) * height:
        raise RuntimeError('PointCloud2 byte count is inconsistent')
    if int(message.row_step) == width * int(message.point_step):
        values = np.frombuffer(raw, dtype=dtype, count=width * height)
    else:
        values = np.concatenate([np.frombuffer(raw, dtype=dtype, count=width, offset=row * int(message.row_step)) for row in range(height)])
    xyz = np.column_stack([values[axis] for axis in 'xyz']).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]

@dataclass
class SyncedFrame:
    lidar_timestamp_ns: int
    rgb_timestamp_ns: int
    sync_delta_ms: float
    image: np.ndarray
    points_rslidar: np.ndarray

def synchronized_bag_frames(bag: Path, slop_ms: float) -> Iterator[SyncedFrame]:
    """Yield latest-past RGB/LiDAR pairs after a bounded arrival-time wait."""
    slop_ns = int(round(slop_ms * 1000000))
    images: deque[tuple[int, np.ndarray]] = deque()
    lidars: deque[tuple[int, int, object]] = deque()
    with AnyReader([bag]) as reader:
        connections = [item for item in reader.connections if item.topic in {IMAGE_TOPIC, LIDAR_TOPIC}]
        if {item.topic for item in connections} != {IMAGE_TOPIC, LIDAR_TOPIC}:
            raise RuntimeError(f'Bag does not contain both {IMAGE_TOPIC} and {LIDAR_TOPIC}')

        def flush(arrival_watermark_ns: int, final: bool=False) -> Iterator[SyncedFrame]:
            while lidars and (final or arrival_watermark_ns >= lidars[0][1] + slop_ns):
                lidar_stamp, _, lidar_message = lidars.popleft()
                selected = select_causal_rgb(images, lidar_stamp, slop_ns)
                if selected is not None:
                    rgb_stamp, image = selected
                    delta = support.causal_support_delta_ms(rgb_stamp, lidar_stamp, slop_ms)
                    yield SyncedFrame(lidar_stamp, rgb_stamp, delta, image.copy(), decode_pointcloud2(lidar_message))
                while images and images[0][0] < lidar_stamp - slop_ns:
                    images.popleft()
        for connection, bag_ns, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            stamp = timestamp_ns(message, int(bag_ns))
            if connection.topic == IMAGE_TOPIC:
                images.append((stamp, decode_image(message)))
            else:
                lidars.append((stamp, int(bag_ns), message))
            yield from flush(int(bag_ns))
        yield from flush(2 ** 63 - 1, final=True)

def select_causal_rgb(images: Iterator[tuple[int, np.ndarray]], lidar_timestamp_ns: int, slop_ns: int) -> tuple[int, np.ndarray] | None:
    """Select the latest arrived RGB whose sensor timestamp is in [LiDAR-slop, LiDAR]."""
    candidates = [(stamp, image) for stamp, image in images if 0 <= lidar_timestamp_ns - stamp <= slop_ns]
    return max(candidates, key=lambda item: item[0]) if candidates else None

class FrozenReID:

    def __init__(self, device: torch.device) -> None:
        model = support.osnet_x1_0(num_classes=4101, pretrained=False)
        result = model.load_state_dict(torch.load(REID_WEIGHTS, map_location='cpu', weights_only=True), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f'OSNet weight mismatch: {result}')
        self.model = model.eval().to(device)
        self.device = device
        frozen = np.load(PROTOTYPES, allow_pickle=False)
        if str(frozen['fit_scope']) != 'TRAIN_FIT_ONLY':
            raise RuntimeError('Identity prototypes are not TRAIN_FIT-only')
        self.identities = tuple(frozen['identities'].astype(str))
        self.prototypes = np.asarray(frozen['centers'], np.float32)

    def warmup(self) -> None:
        with torch.inference_mode():
            self.model(torch.zeros((1, 3, 256, 128), device=self.device))

    def features(self, image: np.ndarray, boxes_xyxy: list[np.ndarray]) -> np.ndarray:
        if not boxes_xyxy:
            return np.empty((0, self.prototypes.shape[1]), np.float32)
        tensors, colors = ([], [])
        for box in boxes_xyxy:
            x1, y1, x2, y2 = map(float, box)
            tensor, color = reid_crop_tensor(image, [x1, y1, x2 - x1, y2 - y1])
            tensors.append(tensor)
            colors.append(color)
        with torch.inference_mode():
            learned = F.normalize(self.model(torch.stack(tensors).to(self.device)), dim=1).cpu().numpy()
        result = []
        for vector, color in zip(learned, colors, strict=True):
            combined = np.concatenate((0.7 * vector, 1.15 * color)).astype(np.float32)
            combined /= max(float(np.linalg.norm(combined)), 1e-08)
            result.append(combined)
        return np.stack(result)

def iou_xyxy(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_left = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    area_right = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    return intersection / max(area_left + area_right - intersection, 1e-09)

class OnlineTracker:
    """Causal 3D/2D tracker; frozen ReID prototypes are used only for new person IDs."""

    def __init__(self, prototypes: np.ndarray, identities: tuple[str, ...], max_age: int=10) -> None:
        self.prototypes, self.identities, self.max_age = (prototypes, identities, max_age)
        self.tracks: dict[str, dict] = {}
        self.next_temporary = 1
        self.frame = -1

    def _new_person_ids(self, observations: list[dict], features: np.ndarray, indices: list[int]) -> dict[int, str]:
        assigned: dict[int, str] = {}
        unavailable = [i for i, name in enumerate(self.identities) if name not in self.tracks]
        if not indices or not unavailable:
            return assigned
        cost = 1.0 - features[indices] @ self.prototypes[unavailable].T
        rr, cc = linear_sum_assignment(cost)
        for row, column in zip(rr, cc, strict=True):
            best = float(cost[row, column])
            if best <= 0.48:
                assigned[indices[int(row)]] = self.identities[unavailable[int(column)]]
        return assigned

    def update(self, observations: list[dict], features: np.ndarray | None=None, feature_loader=None) -> list[dict]:
        self.frame += 1
        active = [(name, state) for name, state in self.tracks.items() if self.frame - state['last_frame'] <= self.max_age]
        cost = np.full((len(active), len(observations)), np.inf, float)
        for i, (_, state) in enumerate(active):
            for j, observation in enumerate(observations):
                if state['class'] != observation['class']:
                    continue
                overlap = iou_xyxy(state['bbox'], observation['bbox'])
                if observation['xyz'] is not None and state['xyz'] is not None:
                    gap = max(self.frame - state['last_frame'], 1)
                    predicted = state['xyz'] + state['velocity'] * gap
                    distance = float(np.linalg.norm(observation['xyz'][:2] - predicted[:2]))
                    gate = 0.7 + 0.18 * min(gap, 5)
                    if distance <= gate:
                        cost[i, j] = 0.75 * distance / gate + 0.25 * (1.0 - overlap)
                elif overlap >= 0.1:
                    cost[i, j] = 0.7 + 0.3 * (1.0 - overlap)
        matched_observations: dict[int, str] = {}
        if cost.size:
            safe = np.where(np.isfinite(cost), cost, 99.0)
            rr, cc = linear_sum_assignment(safe)
            for row, column in zip(rr, cc, strict=True):
                if safe[row, column] <= 1.0:
                    matched_observations[int(column)] = active[int(row)][0]
        new_people = [index for index, item in enumerate(observations) if index not in matched_observations and item['class'] == 'PERSON']
        missing_identities = any((identity not in self.tracks for identity in self.identities))
        recovered_people = [index for index, track_id in matched_observations.items() if observations[index]['class'] == 'PERSON' and track_id.startswith('T')] if missing_identities else []
        identity_candidates = new_people + recovered_people
        if features is None:
            features = np.zeros((len(observations), self.prototypes.shape[1]), np.float32)
            if identity_candidates and feature_loader is not None:
                features[identity_candidates] = feature_loader(identity_candidates)
        recovered_identities = self._new_person_ids(observations, features, identity_candidates)
        for index, identity in recovered_identities.items():
            previous_identity = matched_observations.get(index)
            if previous_identity is not None and previous_identity.startswith('T'):
                self.tracks[identity] = self.tracks.pop(previous_identity)
            matched_observations[index] = identity
        robot_indices = [index for index, item in enumerate(observations) if item['class'] == 'ROBOT']
        if robot_indices and 'R1' not in matched_observations.values():
            primary_robot = max(robot_indices, key=lambda index: observations[index]['confidence'])
            matched_observations[primary_robot] = 'R1'
        for index, item in enumerate(observations):
            if index not in matched_observations:
                matched_observations[index] = f'T{self.next_temporary:04d}'
                self.next_temporary += 1
        outputs = []
        for index, observation in enumerate(observations):
            track_id = matched_observations[index]
            previous = self.tracks.get(track_id)
            xyz = observation['xyz']
            velocity = np.zeros(3, float)
            if previous is not None and xyz is not None and (previous['xyz'] is not None):
                gap = max(self.frame - previous['last_frame'], 1)
                velocity = 0.72 * previous['velocity'] + 0.28 * (xyz - previous['xyz']) / gap
            elif previous is not None:
                velocity = previous['velocity']
            state = {'class': observation['class'], 'bbox': observation['bbox'].copy(), 'xyz': None if xyz is None else xyz.copy(), 'velocity': velocity, 'last_frame': self.frame}
            self.tracks[track_id] = state
            outputs.append({**observation, 'track_id': track_id})
        self.tracks = {key: value for key, value in self.tracks.items() if self.frame - value['last_frame'] <= self.max_age}
        return outputs

class OnlineRgbFrustumPipeline:

    def __init__(self, device: str='0', identity: bool=True, projection_audit: bool=True,
                 audit_warmup_frames: int | None=None, legacy_display_offset: bool | None=None,
                 suppressed_recovery: bool=False,
                 person_only_lightweight: bool=False,
                 detector: str='yolo11s-coco') -> None:
        if detector not in DETECTORS:
            raise ValueError(f'Unsupported detector {detector!r}; choose from {tuple(DETECTORS)}')
        detector_spec = DETECTORS[detector]
        model_path = Path(detector_spec['path'])
        if not model_path.exists():
            raise RuntimeError(f'Frozen YOLO model is missing: {model_path}')
        self.detector_id = detector
        self.detector_family = str(detector_spec['family'])
        self.detector_official_source = str(detector_spec['official_source'])
        self.model_path = model_path.resolve()
        self.device = torch.device('cuda:0' if device != 'cpu' and torch.cuda.is_available() else 'cpu')
        self.person_only_lightweight = bool(person_only_lightweight)
        if self.person_only_lightweight and identity:
            raise ValueError('Lightweight PERSON-only mode cannot enable ReID')
        self.detector = YOLO(str(self.model_path))
        names = {int(key): str(value) for key, value in self.detector.names.items()}
        person_ids = [key for key, value in names.items()
                      if value.strip().lower() == 'person']
        if len(person_ids) != 1:
            raise RuntimeError(
                f'COCO checkpoint must expose exactly one person class, got {person_ids!r}')
        self.person_class_id = int(person_ids[0])
        self.detector_names = names
        self.transform, self.K, self.D = cylinder.projection_assets()
        self.annotated_from_rslidar = support.T_ANNOTATED_FROM_RSLIDAR.copy()
        audit_config = dict(support.REGISTRATION_CONFIG)
        if legacy_display_offset is not None:
            audit_config['legacy_display_enabled'] = bool(legacy_display_offset)
        self.calibration_audit = registration.ProjectionResidualAudit(
            self.transform @ self.annotated_from_rslidar, self.K, self.D,
            audit_config, enabled=projection_audit,
            warmup_frames=audit_warmup_frames,
        )
        empirical = support.EMPIRICAL_INFERENCE_REGISTRATION
        self.empirical_inference_enabled = bool(empirical['enabled'])
        self.inference_du_px = float(empirical['du_px']) if self.empirical_inference_enabled else 0.0
        self.inference_dv_px = float(empirical['dv_px']) if self.empirical_inference_enabled else 0.0
        self.display_du_px = float(support.DISPLAY_OVERLAY_REGISTRATION['du_px'])
        self.display_dv_px = float(support.DISPLAY_OVERLAY_REGISTRATION['dv_px'])
        self.ground_normal, self.ground_d = (GROUND_NORMAL.copy(), GROUND_D)
        self.suppressed_recovery_enabled = bool(suppressed_recovery)
        self.suppressed_recovery = recovery.CausalSuppressedPointRecovery(False)
        self.static, self.voxel_size, self.occupancy, self.background_receipt = offline.load_frozen_background()
        self.geometry_gpu = None
        if self.device.type == 'cuda':
            self.geometry_gpu = support.GpuFrustumBackend(self.static, self.voxel_size, self.occupancy)
        self.occupancy_tree = None if self.geometry_gpu else cKDTree(self.occupancy)
        self.reid = FrozenReID(self.device) if identity else None
        if self.person_only_lightweight:
            self.tracker = tracking.ShortTermPersonTracker(max_age=10)
        else:
            prototype_matrix = self.reid.prototypes if self.reid else np.empty((0, 0), np.float32)
            prototype_ids = self.reid.identities if self.reid else ()
            self.tracker = OnlineTracker(prototype_matrix, prototype_ids)
        detector_options = dict(source=np.zeros((720, 1280, 3), np.uint8), imgsz=960,
                                conf=0.3, iou=0.6, max_det=12,
                                device=str(self.device), verbose=False)
        if self.person_only_lightweight:
            detector_options['classes'] = [self.person_class_id]
        self.detector.predict(**detector_options)
        if self.reid:
            self.reid.warmup()
        if self.geometry_gpu:
            warm_points = np.asarray([[0.0, 0.0, -1.0], [0.03, 0.0, -1.0], [0.06, 0.0, -1.0]])
            self.geometry_gpu.static_keep(warm_points, np.zeros(len(warm_points)))
            self.geometry_gpu.owners(np.asarray([[2.0, 2.0]]), [np.asarray([0.0, 0.0, 4.0, 4.0])])
            self.geometry_gpu.components(warm_points, np.zeros(len(warm_points), np.int32), 1)
            self.geometry_gpu.synchronize()

    def project_physical(self, points: np.ndarray):
        return cylinder.project_points(points, self.transform, self.K, self.D,
                                       du_px=0.0, dv_px=0.0)

    def project_inference(self, points: np.ndarray):
        """Scene01 empirical registration used by ownership and association."""
        return cylinder.project_points(
            points, self.transform, self.K, self.D,
            du_px=self.inference_du_px, dv_px=self.inference_dv_px)

    def project_display(self, points: np.ndarray):
        if self.calibration_audit.legacy_display_enabled:
            du_px = self.calibration_audit.display_du_px
            dv_px = self.calibration_audit.display_dv_px
        else:
            du_px, dv_px = self.display_du_px, self.display_dv_px
        return cylinder.project_points(
            points, self.transform, self.K, self.D,
            du_px=du_px, dv_px=dv_px,
        )

    def projection_status(self) -> dict:
        return {**self.calibration_audit.snapshot(),
                'empirical_inference': support.EMPIRICAL_INFERENCE_REGISTRATION,
                'display_overlay': support.DISPLAY_OVERLAY_REGISTRATION}

    def _calibration_residual(self, detection: dict, component: np.ndarray) -> tuple[float, float, float] | None:
        config = support.REGISTRATION_CONFIG
        if (float(detection['confidence']) < float(config['minimum_dynamic_confidence']) or
                len(component) < int(config['minimum_dynamic_component_points'])):
            return None
        center = frustum.cluster_center(component)
        raw, valid, depth = self.project_physical(center[None])
        if not valid[0] or depth[0] <= 0:
            return None
        box = np.asarray(detection['bbox'], float)
        target_u = 0.5 * (box[0] + box[2])
        bootstrap_error = abs(raw[0, 0] - target_u)
        if bootstrap_error > max(float(config['maximum_bootstrap_center_error_px']), 0.75 * (box[2] - box[0])):
            return None
        residual = float(target_u - raw[0, 0])
        if not float(config['du_search_min_px']) <= residual <= float(config['du_search_max_px']):
            return None
        return (residual, float(depth[0]), target_u)

    def _static_masks(self, points: np.ndarray, ground_height: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        if self.geometry_gpu:
            return self.geometry_gpu.static_masks(points, ground_height)
        temporal, near = self._static_evidence(points)
        occupancy, policy_counts = suppression.occupancy_keep(
            points, ground_height, near, temporal,
            suppression.Policy(suppression.LEGACY))
        removed = {'temporal_removed': int((~temporal).sum()),
                   'occupancy_removed': policy_counts['occupancy_removed']}
        return (temporal, occupancy, removed)

    def _static_evidence(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return frozen temporal keep and 7 cm occupancy proximity evidence."""
        if self.geometry_gpu:
            return self.geometry_gpu.static_evidence(points)
        values = np.asarray(points, np.float64)
        keys = np.floor(values / self.voxel_size).astype(np.int32)
        temporal = np.fromiter(
            (tuple(map(int, key)) not in self.static for key in keys),
            bool, len(keys))
        if len(values):
            near = self.occupancy_tree.query(values[:, :2], workers=-1)[0] <= 0.07
        else:
            near = np.zeros(0, bool)
        return temporal, near

    def _static_keep(self, points: np.ndarray, ground_height: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        temporal, occupancy, removed = self._static_masks(points, ground_height)
        return (temporal & occupancy, removed)

    def suppressed_candidate(self, components: list[np.ndarray], detection: dict,
                             person_boxes: list[np.ndarray]
                             ) -> recovery.RecoveryCandidate | None:
        ranked = []
        for component in components:
            score, _ = support.component_score(
                component, detection['bbox'], self.transform, self.K, self.D,
                du_px=self.inference_du_px, dv_px=self.inference_dv_px)
            if np.isfinite(score):
                ranked.append((float(score), -len(component), component))
        if not ranked:
            return None
        ranked.sort(key=lambda value: (value[0], value[1]))
        score, _, component = ranked[0]
        center = support.cluster_center(component)
        pixel, valid, _ = support.project_points(
            center[None], self.transform, self.K, self.D,
            du_px=self.inference_du_px, dv_px=self.inference_dv_px)
        inside = bool(valid[0]
                      and detection['bbox'][0] <= pixel[0, 0] <= detection['bbox'][2]
                      and detection['bbox'][1] <= pixel[0, 1] <= detection['bbox'][3])
        containing_people = 0 if not valid[0] else sum(
            box[0] <= pixel[0, 0] <= box[2]
            and box[1] <= pixel[0, 1] <= box[3]
            for box in person_boxes)
        span = np.ptp(component, axis=0)
        horizontal, vertical = float(max(span[0], span[1])), float(span[2])
        second_gap = None if len(ranked) < 2 else float(ranked[1][0] - score)
        return recovery.RecoveryCandidate(
            xyz=center, point_count=len(component), component_score=score,
            projected_center_inside=inside,
            plausible_size=(0.10 <= horizontal <= 1.10
                            and 0.10 <= vertical <= 2.20),
            neighbor_competition=containing_people > 1,
            ambiguous_component=(second_gap is not None and second_gap < 0.05))

    def apply_suppressed_recovery(self, tracked: list[dict],
                                  candidates: list[recovery.RecoveryCandidate | None],
                                  frame_index: int) -> dict[str, int]:
        counts = {'attempts': 0, 'accepted': 0, 'rejected': 0}
        for item, candidate in zip(tracked, candidates, strict=True):
            if item['class'] != 'PERSON':
                continue
            if item['xyz'] is not None:
                self.suppressed_recovery.observe_normal(
                    item['track_id'], item['xyz'], frame_index)
                item['measurement_source'] = 'NORMAL_COMPONENT'
                continue
            if not self.suppressed_recovery_enabled:
                item['measurement_source'] = 'UNAVAILABLE'
                continue
            decision = self.suppressed_recovery.consider(
                item['track_id'], candidate, frame_index)
            item['suppressed_recovery_reason'] = decision['reason']
            if candidate is None:
                item['measurement_source'] = 'UNAVAILABLE'
                continue
            counts['attempts'] += 1
            if not decision['accepted']:
                counts['rejected'] += 1
                item['measurement_source'] = 'UNAVAILABLE'
                continue
            assert candidate is not None
            item['xyz'] = candidate.xyz.copy()
            item['component_points'] = candidate.point_count
            item['component_score'] = candidate.component_score
            item['measurement_source'] = 'RECOVERED_SUPPRESSED_COMPONENT'
            counts['accepted'] += 1
            state = self.tracker.tracks.get(item['track_id'])
            if state is not None:
                state['xyz'] = candidate.xyz.copy()
        return counts

    def _detections(self, image: np.ndarray) -> list[dict]:
        options = dict(source=image, imgsz=960, conf=0.3, iou=0.6, max_det=12,
                       device=str(self.device), verbose=False)
        if self.person_only_lightweight:
            options['classes'] = [self.person_class_id]
        result = self.detector.predict(**options)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        allowed = (self.person_class_id,)
        return [{'class': 'PERSON', 'bbox': box.astype(float), 'confidence': float(score),
                 'xyz': None, 'component_points': 0}
                for box, score, cls in zip(boxes, scores, classes, strict=True)
                if cls in allowed]

    def process(self, image: np.ndarray, points_rslidar: np.ndarray, timestamp: int) -> tuple[np.ndarray, dict]:
        self.calibration_audit.begin_frame()
        started = time.perf_counter()
        detections = self._detections(image)
        after_yolo = time.perf_counter()
        transform = self.annotated_from_rslidar
        annotated = points_rslidar @ transform[:3, :3].T + transform[:3, 3]
        pixels, valid, camera_depth = self.project_inference(annotated)
        ground_height = annotated @ self.ground_normal + self.ground_d
        in_view = valid & (camera_depth > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < IMAGE_SIZE[0]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < IMAGE_SIZE[1])
        broad_height = (ground_height >= 0.03) & (ground_height <= 2.15)
        candidate = np.flatnonzero(in_view & broad_height)
        keep, removed = self._static_keep(annotated[candidate], ground_height[candidate])
        usable = candidate[keep]
        boxes = [item['bbox'] for item in detections]
        if self.geometry_gpu:
            owners = self.geometry_gpu.owners(pixels[usable], boxes)
            component_owners = owners.copy()
            for index, detection in enumerate(detections):
                selected = component_owners == index
                lower = 0.08 if detection['class'] == 'PERSON' else 0.03
                upper = 2.15 if detection['class'] == 'PERSON' else 1.35
                component_owners[selected & ((ground_height[usable] < lower) | (ground_height[usable] > upper))] = -1
            grouped_components = self.geometry_gpu.components(annotated[usable], component_owners, len(boxes))
        else:
            owners = frustum.exclusive_point_owners(pixels[usable], boxes, [None] * len(boxes))
            grouped_components = None
        calibration_residuals: list[tuple[float, float, float]] = []
        for index, detection in enumerate(detections):
            if grouped_components is None:
                point_indices = usable[owners == index]
                height = ground_height[point_indices]
                point_indices = point_indices[(height >= (0.08 if detection['class'] == 'PERSON' else 0.03)) & (height <= (2.15 if detection['class'] == 'PERSON' else 1.35))]
                components = frustum.adaptive_components(annotated[point_indices])
            else:
                components = grouped_components[index]
            component, details = offline.choose_component(components, detection['bbox'], self.transform, self.K, self.D,
                                                          du_px=self.inference_du_px,
                                                          dv_px=self.inference_dv_px)
            if component is not None:
                detection['xyz'] = frustum.cluster_center(component)
                detection['component_points'] = len(component)
                detection['component_score'] = float(details['component_score'])
                residual = self._calibration_residual(detection, component)
                if residual is not None:
                    calibration_residuals.append(residual)
        self.calibration_audit.observe(image, boxes, points_rslidar, calibration_residuals)

        def load_new_identity_features(indices: list[int]) -> np.ndarray:
            if not self.reid:
                return np.empty((len(indices), 0), np.float32)
            return self.reid.features(image, [detections[index]['bbox'] for index in indices])
        if self.person_only_lightweight:
            tracked = self.tracker.update(detections)
        else:
            tracked = self.tracker.update(detections, feature_loader=load_new_identity_features)
        rendered = self.render(image, tracked)
        ended = time.perf_counter()
        payload = {'timestamp_ns': int(timestamp), 'detections': len(detections), 'person_detections': sum((item['class'] == 'PERSON' for item in detections)), 'robot_detections': sum((item['class'] == 'ROBOT' for item in detections)), 'lidar_measurements': sum((item['xyz'] is not None for item in detections)), 'tracks': [self.serializable(item) for item in tracked], 'runtime_ms': {'yolo': 1000 * (after_yolo - started), 'fusion_tracking_render': 1000 * (ended - after_yolo), 'total': 1000 * (ended - started)}, 'static_removed': removed, 'projection_audit': {'state': self.calibration_audit.state, 'raw_physical_du_px': 0.0, 'raw_physical_dv_px': 0.0, 'empirical_inference_du_px': self.inference_du_px, 'empirical_inference_dv_px': self.inference_dv_px, 'audit_candidate_applied': False}, 'safety': {'causal': True, 'runtime_gt_used': False, 'bev_cnn_used': False, 'future_track_state_used': False, 'robot_official_z_available': False, 'raw_physical_projection_du_dv_zero': True, 'empirical_scene01_registration_enabled': self.empirical_inference_enabled, 'geometry_backend': 'CUDA' if self.geometry_gpu else 'CPU'}}
        return (rendered, payload)

    @staticmethod
    def serializable(item: dict) -> dict:
        xyz = item['xyz']
        return {'track_id': item['track_id'], 'class': item['class'], 'confidence': item['confidence'], 'bbox_xyxy': item['bbox'].tolist(), 'xyz': None if xyz is None else xyz.tolist(), 'xyz_semantics': 'PERSON_VISIBLE_COMPONENT_CENTER' if item['class'] == 'PERSON' else 'ROBOT_VISIBLE_SURFACE_COMPONENT_CENTER_NOT_OFFICIAL_ORIGIN', 'component_points': item['component_points']}

    def render(self, image: np.ndarray, tracks: list[dict]) -> np.ndarray:
        canvas = image.copy()
        overlay = canvas.copy()
        cylinder_geometry = []
        for item in tracks:
            if item['xyz'] is not None:
                geometry = self._cylinder_geometry(item, COLORS.get(item['track_id'], TEMP_COLOR))
                if geometry is not None:
                    pixels, depth, color, segments = geometry
                    cylinder_geometry.append(geometry)
                    for index in range(segments):
                        nxt = (index + 1) % segments
                        ids = [index, nxt, segments + nxt, segments + index]
                        cv2.fillConvexPoly(overlay, np.round(pixels[ids]).astype(np.int32), color, cv2.LINE_AA)
        if cylinder_geometry:
            cv2.addWeighted(overlay, 0.14, canvas, 0.86, 0, canvas)
            for pixels, _, color, segments in cylinder_geometry:
                for start in (0, segments):
                    cv2.polylines(canvas, [np.round(pixels[start:start + segments]).astype(np.int32)], True, color, 2, cv2.LINE_AA)
                for index in range(0, segments, 3):
                    cv2.line(canvas, tuple(np.round(pixels[index]).astype(int)), tuple(np.round(pixels[index + segments]).astype(int)), color, 1, cv2.LINE_AA)
        for item in tracks:
            track_id = item['track_id']
            color = COLORS.get(track_id, TEMP_COLOR)
            x1, y1, x2, y2 = np.round(item['bbox']).astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            xyz = item['xyz']
            if xyz is None:
                suffix = '3D --'
            elif item['class'] == 'ROBOT':
                suffix = f'XY {xyz[0]:+.2f},{xyz[1]:+.2f}'
            else:
                suffix = f'XYZ {xyz[0]:+.2f},{xyz[1]:+.2f},{xyz[2]:+.2f}'
            cv2.putText(canvas, f'{track_id} {suffix}', (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
            if xyz is not None:
                pixel, projected, _ = self.project_display(xyz[None])
                if projected[0]:
                    u, v = np.round(pixel[0]).astype(int)
                    cv2.drawMarker(canvas, (u, v), color, cv2.MARKER_CROSS, 18, 3, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (IMAGE_SIZE[0], 34), (14, 18, 22), -1)
        cv2.putText(canvas, 'ONLINE | YOLO11s + RGB frustum + static-suppressed Euclidean + causal 3D tracking', (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (238, 240, 243), 1, cv2.LINE_AA)
        return canvas

    def _cylinder_geometry(self, item: dict, color: tuple[int, int, int]):
        """Render display geometry without changing the tracked 3D measurement."""
        xyz = item['xyz']
        x, y = (float(xyz[0]), float(xyz[1]))
        is_robot = item['class'] == 'ROBOT'
        radius, height, segments = (0.34, 0.8, 12) if is_robot else (0.25, 1.7, 12)
        base = cylinder.ground_z(x, y, self.ground_normal, self.ground_d)
        angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
        bottom = np.column_stack((x + radius * np.cos(angles), y + radius * np.sin(angles), np.full(segments, base)))
        top = bottom.copy()
        top[:, 2] += height
        points = np.vstack((bottom, top))
        pixels, valid, depth = self.project_display(points)
        if not valid.all():
            return None
        return (pixels, depth, color, segments)

def run(args: argparse.Namespace) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline = OnlineRgbFrustumPipeline(
        args.device,
        identity=not args.no_identity,
        projection_audit=args.projection_audit,
        audit_warmup_frames=args.projection_audit_warmup_lidar,
        legacy_display_offset=args.legacy_display_offset,
    )
    log_path = Path(args.output_jsonl) if args.output_jsonl else OUT / 'online_events.jsonl'
    video_path = Path(args.record) if args.record else None
    writer = None
    records = []
    first_stamp = None
    wall_start = time.perf_counter()
    try:
        with log_path.open('w', encoding='utf-8') as log:
            for index, frame in enumerate(synchronized_bag_frames(Path(args.bag), args.sync_slop_ms)):
                if first_stamp is None:
                    first_stamp = frame.lidar_timestamp_ns
                    wall_start = time.perf_counter()
                if args.realtime:
                    due = wall_start + (frame.lidar_timestamp_ns - first_stamp) / 1000000000.0
                    remaining = due - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                rendered, payload = pipeline.process(frame.image, frame.points_rslidar, frame.lidar_timestamp_ns)
                payload.update({'frame_index': index, 'rgb_timestamp_ns': frame.rgb_timestamp_ns, 'sync_delta_ms': frame.sync_delta_ms})
                log.write(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n')
                log.flush()
                records.append(payload)
                if video_path is not None:
                    if writer is None:
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, IMAGE_SIZE)
                        if not writer.isOpened():
                            raise RuntimeError(f'Cannot open video writer: {video_path}')
                    writer.write(rendered)
                if not args.headless:
                    cv2.imshow('RGB-guided clustering online', rendered)
                    if cv2.waitKey(1) & 255 in (27, ord('q')):
                        break
                if (index + 1) % 25 == 0:
                    recent = np.asarray([item['runtime_ms']['total'] for item in records[-25:]])
                    print(f'online {index + 1} frames | mean {recent.mean():.1f} ms | p95 {np.quantile(recent, 0.95):.1f} ms', flush=True)
                if args.limit and index + 1 >= args.limit:
                    break
    finally:
        pipeline.calibration_audit.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    if not records:
        raise RuntimeError('No synchronized frames were processed')
    runtime = np.asarray([item['runtime_ms']['total'] for item in records])
    sync = np.asarray([item['sync_delta_ms'] for item in records])
    summary = {'status': 'PASS_ONLINE_CAUSAL_REPLAY' if float(np.quantile(runtime, 0.95)) <= 100.0 else 'RUNS_BUT_P95_EXCEEDS_10HZ_BUDGET', 'frames': len(records), 'source': str(Path(args.bag).resolve()), 'realtime_pacing': bool(args.realtime), 'runtime_ms': {'mean': float(runtime.mean()), 'p95': float(np.quantile(runtime, 0.95)), 'max': float(runtime.max())}, 'sync_delta_ms': {'mean': float(sync.mean()), 'p95': float(np.quantile(sync, 0.95)), 'max': float(sync.max())}, 'projection_residual_audit': pipeline.projection_status(), 'output_jsonl': str(log_path.resolve()), 'recorded_video': None if video_path is None else str(video_path.resolve()), 'online_semantics': {'physical_projection': 'RAW_K_D_T_DU_DV_ZERO_AUDIT_BASELINE', 'inference_projection': 'SCENE01_EMPIRICAL_DU64_DV36', 'audit_candidate_writeback': False, 'legacy_display_offset': bool(args.legacy_display_offset), 'maximum_sync_wait_ms': args.sync_slop_ms, 'future_track_state_used': False, 'ground_truth_used': False, 'frozen_trainfit_reid_prototypes': not args.no_identity, 'geometry_backend': 'CUDA' if pipeline.geometry_gpu else 'CPU', 'transport_tested': 'ROS1_BAG_REALTIME_REPLAY'}}
    summary_path = OUT / 'online_run_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary

def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--bag', default=str(DEFAULT_BAG))
    value.add_argument('--device', default='0', help='CUDA device index or cpu')
    value.add_argument('--sync-slop-ms', type=float, default=35.0)
    value.add_argument('--limit', type=int, default=979, help='default processes TRAIN_FIT only; 0 explicitly processes the complete stream')
    value.add_argument('--realtime', action=argparse.BooleanOptionalAction, default=True)
    value.add_argument('--headless', action='store_true')
    value.add_argument('--no-identity', action='store_true')
    value.add_argument('--record', default='')
    value.add_argument('--output-jsonl', default='')
    value.add_argument('--projection-audit', action=argparse.BooleanOptionalAction,
                       default=bool(support.REGISTRATION_CONFIG['enabled']))
    value.add_argument('--projection-audit-warmup-lidar', type=int,
                       default=int(support.REGISTRATION_CONFIG['warmup_lidar_frames']))
    value.add_argument('--legacy-display-offset', action=argparse.BooleanOptionalAction,
                       default=bool(support.REGISTRATION_CONFIG['legacy_display_enabled']),
                       help='Optional +48 px legacy visualization only; never affects inference')
    return value
if __name__ == '__main__':
    run(parser().parse_args())
