"""Report-only causal recovery for PERSON points removed by occupancy suppression.

This module does not alter projection, static priors or the production runtime.
It only decides whether an already-extracted, quality-checked candidate is
compatible with past metric state. Ground truth and future frames are never
accepted as inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ComponentSignature:
    horizontal_span_m: float
    vertical_span_m: float
    point_count: int
    center_u_norm: float
    center_v_norm: float


@dataclass(frozen=True)
class FrozenConsistencyThresholds:
    innovation_m: float
    horizontal_log_ratio: float
    vertical_log_ratio: float
    point_count_log_ratio: float
    center_u_norm_delta: float
    center_v_norm_delta: float
    source: str = "NORMAL_TRAINFIT_COMPONENTS_ONLY"


@dataclass(frozen=True)
class RecoveryCandidate:
    xyz: np.ndarray
    point_count: int
    component_score: float
    projected_center_inside: bool
    plausible_size: bool
    neighbor_competition: bool
    ambiguous_component: bool
    signature: ComponentSignature | None = None


class CausalSuppressedPointRecovery:
    """Finite causal state for report-only suppressed-component recovery."""

    def __init__(self, allow_causal_seed: bool, max_age_frames: int=10,
                 seed_frames: int=3,
                 consistency_thresholds: FrozenConsistencyThresholds | None=None) -> None:
        self.allow_causal_seed = bool(allow_causal_seed)
        self.max_age_frames = int(max_age_frames)
        self.seed_frames = int(seed_frames)
        self.consistency_thresholds = consistency_thresholds
        self.history: dict[str, dict] = {}
        self.seeds: dict[str, dict] = {}

    @staticmethod
    def motion_gate_m(gap_frames: int) -> float:
        """The existing OnlineTracker gate; no recovery-specific wider gate."""
        return 0.7 + 0.18 * min(max(int(gap_frames), 1), 5)

    @staticmethod
    def quality_reason(candidate: RecoveryCandidate | None) -> str | None:
        if candidate is None:
            return "NO_OCCUPANCY_DELETED_COMPONENT"
        if candidate.point_count < 3:
            return "TOO_FEW_POINTS"
        if not candidate.projected_center_inside:
            return "CENTER_OUTSIDE_ORIGINAL_BOX"
        if not candidate.plausible_size:
            return "IMPLAUSIBLE_3D_SIZE"
        if candidate.component_score > 1.15:
            return "COMPONENT_SCORE_ABOVE_EXISTING_ASSIGNMENT_LIMIT"
        if candidate.neighbor_competition:
            return "NEIGHBOR_PERSON_COMPETITION"
        if candidate.ambiguous_component:
            return "AMBIGUOUS_COMPONENT_RANKING"
        return None

    @staticmethod
    def _signature_deltas(left: ComponentSignature,
                          right: ComponentSignature) -> dict[str, float]:
        safe = lambda value: max(float(value), 1e-06)
        return {
            "horizontal_log_ratio": abs(float(np.log(safe(left.horizontal_span_m)
                                                       / safe(right.horizontal_span_m)))),
            "vertical_log_ratio": abs(float(np.log(safe(left.vertical_span_m)
                                                     / safe(right.vertical_span_m)))),
            "point_count_log_ratio": abs(float(np.log(safe(left.point_count)
                                                        / safe(right.point_count)))),
            "center_u_norm_delta": abs(float(left.center_u_norm - right.center_u_norm)),
            "center_v_norm_delta": abs(float(left.center_v_norm - right.center_v_norm)),
        }

    def _observe(self, track_id: str, xyz: np.ndarray, frame_index: int,
                 source: str, signature: ComponentSignature | None=None) -> None:
        xyz = np.asarray(xyz, np.float64).reshape(3)
        previous = self.history.get(track_id)
        velocity = np.zeros(3, np.float64)
        if previous is not None:
            gap = int(frame_index) - int(previous["frame_index"])
            if gap > 0:
                instant = (xyz - previous["xyz"]) / gap
                velocity = 0.72 * previous["velocity"] + 0.28 * instant
        velocity[2] = 0.0
        normal_signature = (signature if source == "NORMAL_COMPONENT"
                            else None if previous is None
                            else previous.get("normal_signature"))
        self.history[track_id] = {
            "xyz": xyz.copy(), "velocity": velocity,
            "frame_index": int(frame_index), "source": source,
            "normal_signature": normal_signature,
        }

    def observe_normal(self, track_id: str, xyz: np.ndarray,
                       frame_index: int,
                       signature: ComponentSignature | None=None) -> None:
        self._observe(track_id, xyz, frame_index, "NORMAL_COMPONENT", signature)
        self.seeds.pop(track_id, None)

    def consider(self, track_id: str, candidate: RecoveryCandidate | None,
                 frame_index: int) -> dict:
        reason = self.quality_reason(candidate)
        if reason is not None:
            self.seeds.pop(track_id, None)
            return {"accepted": False, "reason": reason,
                    "motion_distance_m": None, "motion_gate_m": None,
                    "history_available": False, "seed_count": 0}
        assert candidate is not None
        if self.consistency_thresholds is None and not self.allow_causal_seed:
            # A plain HISTORY_ONLY manager is intentionally allowed. A strict
            # manager is created only after its frozen thresholds are assigned.
            pass
        previous = self.history.get(track_id)
        if previous is not None:
            gap = int(frame_index) - int(previous["frame_index"])
            if 0 < gap <= self.max_age_frames:
                prediction = previous["xyz"] + previous["velocity"] * gap
                distance = float(np.linalg.norm(candidate.xyz[:2] - prediction[:2]))
                gate = self.motion_gate_m(gap)
                signature_deltas = None
                thresholds = self.consistency_thresholds
                if thresholds is not None:
                    gate = min(gate, float(thresholds.innovation_m))
                    normal_signature = previous.get("normal_signature")
                    if candidate.signature is None or normal_signature is None:
                        return {"accepted": False,
                                "reason": "NORMAL_SIGNATURE_UNAVAILABLE",
                                "motion_distance_m": distance,
                                "motion_gate_m": gate,
                                "history_available": True, "seed_count": 0}
                    signature_deltas = self._signature_deltas(
                        candidate.signature, normal_signature)
                    limits = {
                        "horizontal_log_ratio": thresholds.horizontal_log_ratio,
                        "vertical_log_ratio": thresholds.vertical_log_ratio,
                        "point_count_log_ratio": thresholds.point_count_log_ratio,
                        "center_u_norm_delta": thresholds.center_u_norm_delta,
                        "center_v_norm_delta": thresholds.center_v_norm_delta,
                    }
                    failed = [key for key, value in signature_deltas.items()
                              if value > limits[key]]
                    if failed:
                        return {"accepted": False,
                                "reason": "HISTORY_COMPONENT_SIGNATURE_REJECT",
                                "signature_failed_fields": ";".join(failed),
                                **signature_deltas,
                                "motion_distance_m": distance,
                                "motion_gate_m": gate,
                                "history_available": True, "seed_count": 0}
                if distance <= gate:
                    self._observe(track_id, candidate.xyz, frame_index,
                                  "RECOVERED_SUPPRESSED_COMPONENT")
                    self.seeds.pop(track_id, None)
                    return {"accepted": True,
                            "reason": "HISTORY_MOTION_COMPATIBLE",
                            "motion_distance_m": distance,
                            "motion_gate_m": gate,
                            "history_available": True, "seed_count": 0,
                            **({} if signature_deltas is None else signature_deltas)}
                return {"accepted": False,
                        "reason": "HISTORY_MOTION_GATE_REJECT",
                        "motion_distance_m": distance,
                        "motion_gate_m": gate,
                        "history_available": True, "seed_count": 0}
        if not self.allow_causal_seed:
            return {"accepted": False, "reason": "NO_METRIC_HISTORY",
                    "motion_distance_m": None, "motion_gate_m": None,
                    "history_available": False, "seed_count": 0}
        seed = self.seeds.get(track_id)
        consecutive = 1
        if seed is not None and int(frame_index) == int(seed["frame_index"]) + 1:
            distance = float(np.linalg.norm(candidate.xyz[:2] - seed["xyz"][:2]))
            if distance <= self.motion_gate_m(1):
                consecutive = int(seed["count"]) + 1
        self.seeds[track_id] = {"xyz": candidate.xyz.copy(),
                                "frame_index": int(frame_index),
                                "count": consecutive}
        if consecutive < self.seed_frames:
            return {"accepted": False, "reason": "CAUSAL_SEED_PENDING",
                    "motion_distance_m": None,
                    "motion_gate_m": self.motion_gate_m(1),
                    "history_available": False, "seed_count": consecutive}
        self._observe(track_id, candidate.xyz, frame_index,
                      "RECOVERED_SUPPRESSED_COMPONENT")
        self.seeds.pop(track_id, None)
        return {"accepted": True, "reason": "CAUSAL_SEED_CONFIRMED",
                "motion_distance_m": None,
                "motion_gate_m": self.motion_gate_m(1),
                "history_available": False, "seed_count": consecutive}
