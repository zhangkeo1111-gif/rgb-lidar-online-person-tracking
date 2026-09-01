"""Causal persistent-identity state for the RGB-guided online tracker.

This module never changes detections, boxes, LiDAR components or XYZ.  It only
attaches a persistent G identity to the raw causal track returned by the frozen
tracker.  Recovery uses past identity memory plus the current observation and
frozen ReID embedding; future data and GT are not accepted by the API.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
IDENTITY_STATES = {'ANONYMOUS', 'CANDIDATE', 'CONFIRMED', 'TEMPORARILY_LOST', 'RECOVERED'}
IDENTITY_SOURCES = {'INITIAL_REID_CONFIRMATION', 'TRACK_INHERITANCE', 'SHORT_OCCLUSION_HOLD', 'CAUSAL_RECOVERY', 'ANONYMOUS'}

@dataclass
class IdentityMemory:
    identity: str
    owner_raw_track_id: str
    identity_state: str
    identity_confirmed_timestamp_ns: int
    last_identity_support_timestamp_ns: int
    last_seen_timestamp_ns: int
    last_xyz_timestamp_ns: int | None
    last_xyz: np.ndarray | None
    velocity_mps: np.ndarray
    last_reid_embedding: np.ndarray | None
    reid_support_count: int
    motion_support_count: int

class PersistentIdentityManager:
    """Finite identity lock and motion+appearance causal recovery."""

    def __init__(self, prototypes: np.ndarray, identities: tuple[str, ...], appearance_threshold: float=0.48, max_age: int=10, nominal_lidar_hz: float=10.0) -> None:
        self.prototypes = np.asarray(prototypes, np.float32)
        self.identities = tuple(identities)
        self.prototype_index = {name: index for index, name in enumerate(self.identities)}
        self.appearance_threshold = float(appearance_threshold)
        lifecycle_s = max_age / nominal_lidar_hz
        self.reservation_ns = int(round((lifecycle_s + 0.2) * 1000000000.0))
        self.recovery_memory_ns = int(round(2.0 * lifecycle_s * 1000000000.0))
        self.memories: dict[str, IdentityMemory] = {}
        self.raw_to_identity: dict[str, str] = {}
        self.candidate_support: Counter[tuple[str, str]] = Counter()
        self.events: list[dict] = []
        self.counters: Counter[str] = Counter()
        self.last_timestamp_ns: int | None = None

    @staticmethod
    def _motion_gate_m(gap_s: float) -> float:
        return min(3.0, 0.75 + 1.5 * max(gap_s, 0.0))

    def _evidence(self, memory: IdentityMemory, xyz: np.ndarray | None, embedding: np.ndarray | None, timestamp_ns: int) -> dict:
        reference_ns = memory.last_xyz_timestamp_ns if memory.last_xyz_timestamp_ns is not None else memory.last_seen_timestamp_ns
        gap_s = max((timestamp_ns - reference_ns) / 1000000000.0, 0.0)
        motion_distance = float('inf')
        motion_gate = self._motion_gate_m(gap_s)
        if xyz is not None and memory.last_xyz is not None:
            predicted = memory.last_xyz.copy()
            predicted[:2] += memory.velocity_mps[:2] * gap_s
            motion_distance = float(np.linalg.norm(np.asarray(xyz)[:2] - predicted[:2]))
        reid_distance = float('inf')
        if embedding is not None:
            prototype = self.prototypes[self.prototype_index[memory.identity]]
            reid_distance = float(1.0 - np.dot(embedding, prototype))
        return {'time_gap_ms': gap_s * 1000.0, 'motion_distance_m': motion_distance, 'motion_gate_m': motion_gate, 'reid_distance': reid_distance, 'appearance_threshold': self.appearance_threshold, 'motion_compatible': motion_distance <= motion_gate, 'appearance_compatible': reid_distance <= self.appearance_threshold}

    def _event(self, timestamp_ns: int, raw_track_id: str | None, old_identity: str | None, new_identity: str | None, reason: str, old_state: str, new_state: str, evidence: dict | None=None, conflict_status: str='NONE') -> None:
        values = evidence or {}

        def finite_or_none(name: str):
            value = values.get(name)
            return value if value is None or np.isfinite(value) else None
        self.events.append({'timestamp_ns': int(timestamp_ns), 'raw_track_id': raw_track_id, 'old_identity': old_identity, 'new_identity': new_identity, 'old_state': old_state, 'new_state': new_state, 'reason': reason, 'reid_distance': finite_or_none('reid_distance'), 'motion_distance_m': finite_or_none('motion_distance_m'), 'motion_gate_m': finite_or_none('motion_gate_m'), 'time_gap_ms': finite_or_none('time_gap_ms'), 'conflict_status': conflict_status})

    def _update_memory(self, identity: str, raw_track_id: str, xyz: np.ndarray | None, embedding: np.ndarray | None, timestamp_ns: int, state: str) -> None:
        previous = self.memories.get(identity)
        velocity = np.zeros(3, np.float64)
        motion_support = 0
        reid_support = int(embedding is not None)
        confirmed_at = timestamp_ns
        last_xyz = None if xyz is None else np.asarray(xyz, np.float64).copy()
        last_xyz_timestamp_ns = int(timestamp_ns) if xyz is not None else None
        last_embedding = None if embedding is None else np.asarray(embedding, np.float32).copy()
        if previous is not None:
            velocity = previous.velocity_mps.copy()
            motion_support = previous.motion_support_count
            reid_support += previous.reid_support_count
            confirmed_at = previous.identity_confirmed_timestamp_ns
            if xyz is None:
                last_xyz = None if previous.last_xyz is None else previous.last_xyz.copy()
                last_xyz_timestamp_ns = previous.last_xyz_timestamp_ns
            if embedding is None:
                last_embedding = None if previous.last_reid_embedding is None else previous.last_reid_embedding.copy()
            if xyz is not None and previous.last_xyz is not None:
                reference_ns = previous.last_xyz_timestamp_ns if previous.last_xyz_timestamp_ns is not None else previous.last_seen_timestamp_ns
                dt = (timestamp_ns - reference_ns) / 1000000000.0
                if dt > 1e-06:
                    instant = (np.asarray(xyz, np.float64) - previous.last_xyz) / dt
                    instant[2] = 0.0
                    velocity = 0.72 * velocity + 0.28 * instant
                    motion_support += 1
        velocity[2] = 0.0
        self.memories[identity] = IdentityMemory(identity=identity, owner_raw_track_id=raw_track_id, identity_state=state, identity_confirmed_timestamp_ns=int(confirmed_at), last_identity_support_timestamp_ns=int(timestamp_ns), last_seen_timestamp_ns=int(timestamp_ns), last_xyz_timestamp_ns=last_xyz_timestamp_ns, last_xyz=last_xyz, velocity_mps=velocity, last_reid_embedding=last_embedding, reid_support_count=reid_support, motion_support_count=motion_support)

    def _bind(self, raw_track_id: str, identity: str) -> None:
        for raw, assigned in list(self.raw_to_identity.items()):
            if assigned == identity:
                del self.raw_to_identity[raw]
        self.raw_to_identity[raw_track_id] = identity

    def resolve(self, tracks: list[dict], embeddings: dict[int, np.ndarray], timestamp_ns: int) -> list[dict]:
        """Attach persistent_identity without changing raw track IDs or geometry."""
        if self.last_timestamp_ns is not None and timestamp_ns <= self.last_timestamp_ns:
            raise ValueError('identity updates must be strictly causal and monotonic')
        self.last_timestamp_ns = int(timestamp_ns)
        outputs = [{**item, 'base_track_id': item['track_id']} for item in tracks]
        assigned: dict[int, tuple[str, str, str, dict]] = {}
        used_identities: set[str] = set()
        person_indices = [index for index, item in enumerate(outputs) if item['class'] == 'PERSON']
        current_raw_ids = {outputs[index]['track_id'] for index in person_indices}
        ordered = sorted(person_indices, key=lambda index: (self.raw_to_identity.get(outputs[index]['track_id']) is None, outputs[index]['track_id']))
        for index in ordered:
            item = outputs[index]
            raw = item['track_id']
            identity = self.raw_to_identity.get(raw)
            memory = self.memories.get(identity) if identity else None
            if memory is None or identity in used_identities:
                continue
            identity_gap_ns = timestamp_ns - memory.last_seen_timestamp_ns
            if memory.identity_state == 'TEMPORARILY_LOST' and identity_gap_ns > self.reservation_ns:
                del self.raw_to_identity[raw]
                continue
            evidence = self._evidence(memory, item.get('xyz'), embeddings.get(index), timestamp_ns)
            hard_contradiction = item.get('xyz') is not None and memory.last_xyz is not None and (not evidence['motion_compatible'])
            if hard_contradiction:
                self.counters['identity_teleportation_rejected'] += 1
                self.counters['confirmed_identity_dropped'] += 1
                self._event(timestamp_ns, raw, identity, None, 'HARD_PHYSICAL_CONTRADICTION', memory.identity_state, 'ANONYMOUS', evidence)
                del self.raw_to_identity[raw]
                continue
            short_hold = memory.identity_state == 'TEMPORARILY_LOST'
            source = 'SHORT_OCCLUSION_HOLD' if short_hold else 'TRACK_INHERITANCE'
            assigned[index] = (identity, 'CONFIRMED', source, evidence)
            used_identities.add(identity)
            self.counters['short_occlusion_retained_rows' if short_hold else 'track_inheritance_measurement_rows'] += 1
            if short_hold:
                self._event(timestamp_ns, raw, identity, identity, 'OWNER_TRACK_RETURNED_WITHIN_RESERVATION', 'TEMPORARILY_LOST', 'CONFIRMED', evidence)
        for index in person_indices:
            if index in assigned:
                continue
            item = outputs[index]
            raw = item['track_id']
            if raw not in self.identities or raw in self.memories or raw in used_identities:
                continue
            embedding = embeddings.get(index)
            distance = None
            if embedding is not None:
                distance = float(1.0 - np.dot(embedding, self.prototypes[self.prototype_index[raw]]))
            evidence = {'reid_distance': distance, 'motion_distance_m': None, 'motion_gate_m': None, 'time_gap_ms': 0.0}
            assigned[index] = (raw, 'CONFIRMED', 'INITIAL_REID_CONFIRMATION', evidence)
            used_identities.add(raw)
            self._bind(raw, raw)
            self.counters['initial_confirmations'] += 1
            self._event(timestamp_ns, raw, None, raw, 'FROZEN_BASE_REID_ACCEPTED', 'ANONYMOUS', 'CONFIRMED', evidence)
        for identity, memory in list(self.memories.items()):
            if identity in used_identities:
                continue
            gap_ns = timestamp_ns - memory.last_seen_timestamp_ns
            if gap_ns > self.recovery_memory_ns:
                self.counters['identity_memory_expired'] += 1
                self._event(timestamp_ns, memory.owner_raw_track_id, identity, None, 'FINITE_IDENTITY_MEMORY_EXPIRED', memory.identity_state, 'ANONYMOUS', {'time_gap_ms': gap_ns / 1000000.0})
                del self.memories[identity]
                for raw, value in list(self.raw_to_identity.items()):
                    if value == identity:
                        del self.raw_to_identity[raw]
            elif memory.identity_state != 'TEMPORARILY_LOST':
                memory.identity_state = 'TEMPORARILY_LOST'
                self.counters['temporarily_lost_events'] += 1
                self._event(timestamp_ns, memory.owner_raw_track_id, identity, identity, 'OWNER_TRACK_NOT_OBSERVED', 'CONFIRMED', 'TEMPORARILY_LOST', {'time_gap_ms': gap_ns / 1000000.0})
        candidates = [index for index in person_indices if index not in assigned]
        recoverable = [identity for identity, memory in self.memories.items() if identity not in used_identities and timestamp_ns - memory.last_seen_timestamp_ns <= self.recovery_memory_ns]
        evidence_by_pair: dict[tuple[int, int], dict] = {}
        cost = np.full((len(candidates), len(recoverable)), 99.0, np.float64)
        for row, index in enumerate(candidates):
            for column, identity in enumerate(recoverable):
                evidence = self._evidence(self.memories[identity], outputs[index].get('xyz'), embeddings.get(index), timestamp_ns)
                evidence_by_pair[row, column] = evidence
                if evidence['motion_compatible'] and evidence['appearance_compatible']:
                    cost[row, column] = 0.6 * evidence['reid_distance'] / self.appearance_threshold + 0.4 * evidence['motion_distance_m'] / evidence['motion_gate_m']
        accepted_rows: set[int] = set()
        if cost.size:
            rr, cc = linear_sum_assignment(cost)
            for row, column in zip(rr, cc, strict=True):
                if cost[row, column] > 1.0:
                    continue
                index = candidates[int(row)]
                identity = recoverable[int(column)]
                raw = outputs[index]['track_id']
                evidence = evidence_by_pair[int(row), int(column)]
                assigned[index] = (identity, 'RECOVERED', 'CAUSAL_RECOVERY', evidence)
                used_identities.add(identity)
                accepted_rows.add(int(row))
                self._bind(raw, identity)
                self.counters['recovery_attempts'] += 1
                self.counters['recovery_accepted'] += 1
                self._event(timestamp_ns, raw, None, identity, 'MOTION_APPEARANCE_AVAILABLE_GLOBAL_MATCH', 'ANONYMOUS', 'RECOVERED', evidence)
        for row, index in enumerate(candidates):
            if row in accepted_rows or not recoverable:
                continue
            self.counters['recovery_attempts'] += 1
            self.counters['recovery_rejected'] += 1
            best_column = min(range(len(recoverable)), key=lambda column: cost[row, column])
            evidence = evidence_by_pair[row, best_column]
            reason = 'MOTION_INCOMPATIBLE' if not evidence['motion_compatible'] else 'APPEARANCE_INCOMPATIBLE' if not evidence['appearance_compatible'] else 'GLOBAL_ONE_TO_ONE_CONFLICT'
            self._event(timestamp_ns, outputs[index]['track_id'], None, None, f'RECOVERY_REJECTED_{reason}', 'ANONYMOUS', 'ANONYMOUS', evidence, 'IDENTITY_RESERVED_OR_ASSIGNED' if reason.endswith('CONFLICT') else 'NONE')
        for index, item in enumerate(outputs):
            if item['class'] == 'ROBOT':
                identity = 'R1' if item['track_id'] == 'R1' else None
                item.update({'persistent_identity': identity, 'identity_state': 'CONFIRMED' if identity else 'ANONYMOUS', 'identity_source': 'TRACK_INHERITANCE' if identity else 'ANONYMOUS', 'identity_evidence': {}})
                continue
            if index not in assigned:
                item.update({'persistent_identity': None, 'identity_state': 'ANONYMOUS', 'identity_source': 'ANONYMOUS', 'identity_evidence': {}})
                self.counters['anonymous_measurement_rows'] += 1
                continue
            identity, state, source, evidence = assigned[index]
            embedding = embeddings.get(index)
            self._update_memory(identity, item['track_id'], item.get('xyz'), embedding, timestamp_ns, state)
            item.update({'persistent_identity': identity, 'identity_state': state, 'identity_source': source, 'identity_evidence': evidence})
            self.counters['confirmed_measurement_rows'] += 1
        fixed = [item.get('persistent_identity') for item in outputs if item.get('persistent_identity') in self.identities]
        if len(fixed) != len(set(fixed)):
            raise RuntimeError('persistent identity manager produced a same-frame duplicate G')
        return outputs

    def summary(self) -> dict:
        return {**{key: int(value) for key, value in self.counters.items()}, 'active_identity_memories': len(self.memories), 'appearance_threshold': self.appearance_threshold, 'reservation_ms': self.reservation_ns / 1000000.0, 'recovery_memory_ms': self.recovery_memory_ns / 1000000.0, 'future_sensor_or_reid_access': False, 'runtime_gt_identity_used': False, 'forced_five_identity_assignment': False}
