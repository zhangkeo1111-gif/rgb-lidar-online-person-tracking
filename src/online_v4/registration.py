"""Causal scene-level projection residual audit.

The estimator consumes only an initial warmup prefix.  Automatic RGB/component
center residuals are checked across time, depth, image region and a chronological
holdout; repeated static LiDAR/RGB edges remain non-semantic advisory evidence.
The result is report-only.  Candidate values never modify physical inference;
that path remains raw K/D/T with zero post-projection translation.
"""
from __future__ import annotations

import math
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameEvidence:
    edge_points_rslidar: np.ndarray
    rgb_edge_distance_half: np.ndarray
    dynamic_samples: np.ndarray


def _scan_ring_edges(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float64).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    ranges = np.linalg.norm(points, axis=1)
    points, ranges = points[(ranges >= 1.0) & (ranges <= 18.0)], ranges[(ranges >= 1.0) & (ranges <= 18.0)]
    if len(points) < 20:
        return points.astype(np.float32)
    elevation = np.degrees(np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1])))
    azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    centers = np.arange(-15.0, 15.01, 2.0)
    rings = np.argmin(np.abs(elevation[:, None] - centers[None, :]), axis=1)
    ring_error = np.abs(elevation - centers[rings])
    selected: set[int] = set()
    for ring in range(len(centers)):
        local = np.flatnonzero((rings == ring) & (ring_error < 0.75))
        if len(local) < 4:
            continue
        order = local[np.argsort(azimuth[local])]
        angle_gap = np.diff(azimuth[order])
        range_gap = np.abs(np.diff(ranges[order]))
        threshold = np.maximum(0.25, 0.045 * np.minimum(ranges[order[:-1]], ranges[order[1:]]))
        for hit in np.flatnonzero((angle_gap > 0.01) & (angle_gap < 0.45) & (range_gap > threshold)):
            selected.add(int(order[hit])); selected.add(int(order[hit + 1]))
    if not selected:
        return np.empty((0, 3), np.float32)
    return points[np.fromiter(sorted(selected), dtype=int)].astype(np.float32)


def _edge_distance_half(image: np.ndarray, boxes: list[np.ndarray]) -> np.ndarray:
    small = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0.9)
    edges = cv2.Canny(gray, 55, 130)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    structural = np.zeros_like(edges)
    for contour in contours:
        if cv2.arcLength(contour, False) >= 14.0:
            cv2.drawContours(structural, [contour], -1, 255, 1)
    for box in boxes:
        x1, y1, x2, y2 = np.asarray(box, float) * 0.5
        cv2.rectangle(structural, (max(0, round(x1 - 18)), max(0, round(y1 - 10))),
                      (min(639, round(x2 + 18)), min(359, round(y2 + 10))), 0, -1)
    structural[:6] = 0; structural[-9:] = 0; structural[:, :6] = 0; structural[:, -6:] = 0
    return np.minimum(cv2.distanceTransform(255 - structural, cv2.DIST_L2, 3), 15.0).astype(np.float16)


def _project_raw(points: np.ndarray, optical_from_rslidar: np.ndarray,
                 K: np.ndarray, D: np.ndarray) -> np.ndarray:
    camera = (optical_from_rslidar @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    uv = np.full((len(points), 2), np.nan, np.float64)
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.3) & (camera[:, 2] < 30.0)
    if valid.any():
        uv[valid] = cv2.projectPoints(camera[valid].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    return uv


def _blocks(values: list[float], count: int = 4) -> list[list[float]]:
    return [[values[int(index)] for index in block]
            for block in np.array_split(np.arange(len(values)), count) if len(block)]


class ProjectionResidualAudit:
    """Estimate diagnostic residual candidates without writing them to inference."""

    def __init__(self, optical_from_rslidar: np.ndarray, K: np.ndarray, D: np.ndarray,
                 config: dict, enabled: bool = True, warmup_frames: int | None = None) -> None:
        self.transform = np.asarray(optical_from_rslidar, np.float64)
        self.K, self.D = np.asarray(K, np.float64), np.asarray(D, np.float64)
        self.config = dict(config)
        self.enabled = bool(enabled)
        self.warmup_frames = int(warmup_frames or config['warmup_lidar_frames'])
        self.physical_du_px = float(config['physical_baseline_du_px'])
        self.physical_dv_px = float(config['physical_baseline_dv_px'])
        if self.physical_du_px != 0.0 or self.physical_dv_px != 0.0:
            raise ValueError('physical inference baseline must keep du=dv=0')
        if bool(config['candidate_applied_to_inference']):
            raise ValueError('projection audit candidates must be report-only')
        self.legacy_display_enabled = bool(config['legacy_display_enabled'])
        self.legacy_display_du_px = float(config['legacy_display_du_px'])
        self.legacy_display_dv_px = float(config['legacy_display_dv_px'])
        self.state = 'WARMUP' if self.enabled else 'AUDIT_DISABLED_PHYSICAL_BASELINE'
        self.frames: list[FrameEvidence] = []
        self.result: dict | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future | None = None

    @property
    def du_px(self) -> float:
        """Compatibility accessor: physical inference is always zero-shift."""
        return self.physical_du_px

    @property
    def dv_px(self) -> float:
        return self.physical_dv_px

    @property
    def display_du_px(self) -> float:
        return self.legacy_display_du_px if self.legacy_display_enabled else self.physical_du_px

    @property
    def display_dv_px(self) -> float:
        return self.legacy_display_dv_px if self.legacy_display_enabled else self.physical_dv_px

    def begin_frame(self) -> float:
        """Commit a completed estimate only at a frame boundary."""
        self._poll()
        return self.physical_du_px

    def observe(self, image: np.ndarray, boxes: list[np.ndarray], points_rslidar: np.ndarray,
                dynamic_samples: list[tuple[float, float, float]]) -> None:
        if self.state != 'WARMUP':
            return
        sample_static = len(self.frames) % int(self.config['static_sample_stride']) == 0
        self.frames.append(FrameEvidence(
            _scan_ring_edges(points_rslidar) if sample_static else np.empty((0, 3), np.float32),
            _edge_distance_half(image, boxes) if sample_static else np.empty((0, 0), np.float16),
            np.asarray(dynamic_samples, np.float32).reshape(-1, 3),
        ))
        if len(self.frames) >= self.warmup_frames:
            self.state = 'ESTIMATING'
            frames = tuple(self.frames)
            self.frames.clear()
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='projection-registration')
            self._future = self._executor.submit(self._estimate, frames)

    def _static_uv(self, frames: tuple[FrameEvidence, ...], fit_count: int) -> list[np.ndarray]:
        voxel = float(self.config['voxel_size_m'])
        counts: Counter[tuple[int, int, int]] = Counter()
        for frame in frames[:fit_count]:
            keys = np.unique(np.floor(frame.edge_points_rslidar / voxel).astype(np.int32), axis=0)
            counts.update(map(tuple, keys.tolist()))
        threshold = max(3, math.ceil(float(self.config['static_frequency']) * fit_count))
        result = []
        for frame in frames:
            keys = np.floor(frame.edge_points_rslidar / voxel).astype(np.int32)
            static = np.fromiter((counts.get(tuple(key), 0) >= threshold for key in keys), bool, len(keys))
            result.append(_project_raw(frame.edge_points_rslidar[static], self.transform, self.K, self.D))
        return result

    @staticmethod
    def _frame_score(uv: np.ndarray, distance: np.ndarray, du_px: float) -> float:
        if len(uv) < 8:
            return 30.0
        finite = np.isfinite(uv).all(axis=1) & (np.abs(uv).max(axis=1) < 1000000.0)
        x = np.zeros(len(uv), np.int32)
        y = np.zeros(len(uv), np.int32)
        x[finite] = np.rint((uv[finite, 0] + du_px) * 0.5).astype(np.int32)
        y[finite] = np.rint(uv[finite, 1] * 0.5).astype(np.int32)
        inside = finite & (x >= 0) & (x < 640) & (y >= 0) & (y < 360)
        values = np.full(len(uv), 15.0, np.float32)
        values[inside] = distance[y[inside], x[inside]]
        cap = float(np.percentile(values, 90))
        return 2.0 * float(np.mean(np.minimum(values, cap)))

    def _static_score(self, frames: tuple[FrameEvidence, ...], uv: list[np.ndarray],
                      indices: range | np.ndarray, du_px: float) -> float:
        scores = [self._frame_score(uv[int(index)], frames[int(index)].rgb_edge_distance_half, du_px)
                  for index in indices if frames[int(index)].rgb_edge_distance_half.size]
        return float(np.mean(scores)) if scores else math.inf

    def _group_medians(self, samples: np.ndarray, column: int, boundaries: tuple[float, float]) -> list[float]:
        minimum = int(self.config['minimum_samples_per_dynamic_bin'])
        groups = [samples[samples[:, column] < boundaries[0]],
                  samples[(samples[:, column] >= boundaries[0]) & (samples[:, column] < boundaries[1])],
                  samples[samples[:, column] >= boundaries[1]]]
        return [float(np.median(group[:, 0])) for group in groups if len(group) >= minimum]

    def _estimate(self, frames: tuple[FrameEvidence, ...]) -> dict:
        fit_count = int(math.floor(float(self.config['fit_fraction']) * len(frames)))
        fit, hold = range(fit_count), range(fit_count, len(frames))
        grid = np.arange(float(self.config['du_search_min_px']),
                         float(self.config['du_search_max_px']) + 0.5 * float(self.config['du_search_step_px']),
                         float(self.config['du_search_step_px']))
        uv = self._static_uv(frames, fit_count)
        static_scores = np.asarray([self._static_score(frames, uv, fit, float(value)) for value in grid])
        static_du = float(grid[int(np.argmin(static_scores))])
        static_blocks = []
        for block in np.array_split(np.arange(fit_count), 4):
            scores = [self._static_score(frames, uv, block, float(value)) for value in grid]
            static_blocks.append(float(grid[int(np.argmin(scores))]))

        dynamic_by_frame = [frame.dynamic_samples[np.isfinite(frame.dynamic_samples).all(axis=1)] for frame in frames]
        fit_groups = [dynamic_by_frame[index] for index in fit if len(dynamic_by_frame[index])]
        hold_groups = [dynamic_by_frame[index] for index in hold if len(dynamic_by_frame[index])]
        dynamic_fit_samples = np.vstack(fit_groups) if fit_groups else np.empty((0, 3), np.float32)
        dynamic_hold_samples = np.vstack(hold_groups) if hold_groups else np.empty((0, 3), np.float32)
        dynamic_fit = dynamic_fit_samples[:, 0]
        dynamic_hold = dynamic_hold_samples[:, 0]
        dynamic_frames = [float(np.median(values[:, 0])) for values in dynamic_by_frame[:fit_count] if len(values)]
        dynamic_blocks = [float(np.median(block)) for block in _blocks(dynamic_frames) if block]
        dynamic_depth_groups = self._group_medians(dynamic_fit_samples, 1, (4.0, 8.0))
        dynamic_image_groups = self._group_medians(dynamic_fit_samples, 2, (1280.0 / 3.0, 2560.0 / 3.0))
        dynamic_du = float(np.median(dynamic_fit)) if len(dynamic_fit) else self.physical_du_px
        step = float(self.config['du_search_step_px'])
        candidate = float(np.clip(step * round(dynamic_du / step), grid[0], grid[-1]))

        static_hold_baseline = self._static_score(frames, uv, hold, self.physical_du_px)
        static_hold_candidate = self._static_score(frames, uv, hold, candidate)
        dynamic_hold_baseline = float(np.median(np.abs(dynamic_hold - self.physical_du_px))) if len(dynamic_hold) else math.inf
        dynamic_hold_candidate = float(np.median(np.abs(dynamic_hold - candidate))) if len(dynamic_hold) else math.inf
        dynamic_improvement = 0.0 if not math.isfinite(dynamic_hold_baseline) else 1.0 - dynamic_hold_candidate / max(dynamic_hold_baseline, 1e-6)
        static_counts = [int(np.isfinite(value).all(axis=1).sum()) for frame, value in zip(frames, uv, strict=True)
                         if frame.rgb_edge_distance_half.size]
        checks = {
            'enough_dynamic_frames': sum(bool(len(values)) for values in dynamic_by_frame) >= int(self.config['minimum_dynamic_frames']),
            'dynamic_block_stable': bool(dynamic_blocks) and max(dynamic_blocks) - min(dynamic_blocks) <= float(self.config['maximum_dynamic_block_spread_px']),
            'dynamic_depth_consistent': len(dynamic_depth_groups) >= 2 and max(dynamic_depth_groups) - min(dynamic_depth_groups) <= float(self.config['maximum_dynamic_depth_spread_px']),
            'dynamic_image_region_consistent': len(dynamic_image_groups) >= 2 and max(dynamic_image_groups) - min(dynamic_image_groups) <= float(self.config['maximum_dynamic_image_region_spread_px']),
            'dynamic_holdout_improved': dynamic_improvement >= float(self.config['minimum_dynamic_holdout_improvement_fraction']),
        }
        advisory = {
            'enough_static_points': len(static_counts) >= int(self.config['minimum_static_frames']) and min(static_counts, default=0) >= int(self.config['minimum_static_points_per_frame']),
            'static_block_stable': bool(static_blocks) and max(static_blocks) - min(static_blocks) <= float(self.config['maximum_static_block_spread_px']),
            'static_dynamic_agree': abs(static_du - dynamic_du) <= float(self.config['maximum_static_dynamic_disagreement_px']),
            'static_holdout_not_regressed': static_hold_candidate <= static_hold_baseline * (1.0 + float(self.config['maximum_static_holdout_regression_fraction'])),
        }
        return {
            'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': {key: bool(value) for key, value in checks.items()},
            'advisory_static_edge_checks': {key: bool(value) for key, value in advisory.items()},
            'diagnostic_candidate_du_px': candidate,
            'physical_baseline_du_px': self.physical_du_px,
            'candidate_applied_to_inference': False,
            'static_du_px': static_du, 'dynamic_du_px': dynamic_du,
            'static_block_du_px': static_blocks, 'dynamic_block_du_px': dynamic_blocks,
            'dynamic_depth_bin_du_px': dynamic_depth_groups,
            'dynamic_image_region_du_px': dynamic_image_groups,
            'static_holdout_score_px': {'candidate': static_hold_candidate, 'physical_baseline': static_hold_baseline},
            'dynamic_holdout_median_abs_px': {'candidate': dynamic_hold_candidate, 'physical_baseline': dynamic_hold_baseline},
            'dynamic_holdout_improvement_fraction': dynamic_improvement,
            'warmup_frames': len(frames), 'fit_frames': fit_count, 'holdout_frames': len(frames) - fit_count,
            'manual_correspondences_used': False, 'ground_truth_used': False,
            'future_frames_used_after_freeze': False, 'per_frame_correction': False,
        }

    def _poll(self, wait: bool = False) -> None:
        if self._future is None or (not wait and not self._future.done()):
            return
        try:
            result = self._future.result()
        except Exception as error:  # registration failure must not stop the online detector
            result = {
                'status': 'FAIL',
                'checks': {'estimator_completed_without_exception': False},
                'error': f'{type(error).__name__}: {error}',
                'diagnostic_candidate_du_px': None,
                'physical_baseline_du_px': self.physical_du_px,
                'candidate_applied_to_inference': False,
                'manual_correspondences_used': False,
                'ground_truth_used': False,
                'future_frames_used_after_freeze': False,
                'per_frame_correction': False,
            }
        self.result = result
        self.state = 'AUDIT_CANDIDATE_PASS_REPORT_ONLY' if result['status'] == 'PASS' else 'AUDIT_CANDIDATE_FAIL_REPORT_ONLY'
        if self._executor is not None:
            self._executor.shutdown(wait=False)
        self._executor = None; self._future = None

    def close(self) -> None:
        self._poll(wait=True)

    def snapshot(self) -> dict:
        self._poll()
        return {
            'mode': self.config['mode'], 'state': self.state,
            'physical_inference': {
                'projection': 'RAW_K_D_T', 'du_px': self.physical_du_px, 'dv_px': self.physical_dv_px,
                'calibration_status': 'RAW_PHYSICAL_CALIBRATION_BASELINE_NOT_PROVEN_FINAL',
                'audit_candidate_applied': False,
            },
            'legacy_visualization': {
                'enabled': self.legacy_display_enabled, 'du_px': self.legacy_display_du_px,
                'dv_px': self.legacy_display_dv_px, 'inference_effect': False,
            },
            'warmup_frames_required': self.warmup_frames,
            'warmup_frames_collected': self.warmup_frames if self.state != 'WARMUP' else len(self.frames),
            'result': self.result, 'provenance': self.config['provenance'],
        }
