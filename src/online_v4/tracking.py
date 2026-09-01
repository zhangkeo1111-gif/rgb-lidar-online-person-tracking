"""Lightweight causal short-term PERSON tracking.

This module intentionally contains no detector, ROBOT, ReID, fixed identity,
prototype, or long-term recovery logic.  Its geometry and lifecycle constants
match the frozen PERSON branch of the online_v3 tracker.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_xyxy(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_left = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    area_right = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    return intersection / max(area_left + area_right - intersection, 1e-9)


class ShortTermPersonTracker:
    """Hungarian PERSON tracker using only causal 3D geometry and 2D IoU."""

    def __init__(self, max_age: int = 10) -> None:
        self.max_age = int(max_age)
        self.tracks: dict[str, dict] = {}
        self.next_id = 1
        self.frame = -1
        self.births = 0
        self.deaths = 0
        self.matches = 0
        self.unmatched_observations = 0
        self.short_gap_continuations = 0
        self.completed_lifetimes: list[int] = []

    def update(self, observations: list[dict]) -> list[dict]:
        if any(item.get('class') != 'PERSON' for item in observations):
            raise ValueError('ShortTermPersonTracker accepts PERSON observations only')
        self.frame += 1
        active = [(key, state) for key, state in self.tracks.items()
                  if self.frame - state['last_frame'] <= self.max_age]
        cost = np.full((len(active), len(observations)), np.inf, np.float64)
        for row, (_, state) in enumerate(active):
            for column, observation in enumerate(observations):
                overlap = iou_xyxy(state['bbox'], observation['bbox'])
                xyz = observation.get('xyz')
                if xyz is not None and state['xyz'] is not None:
                    gap = max(self.frame - state['last_frame'], 1)
                    predicted = state['xyz'] + state['velocity'] * gap
                    distance = float(np.linalg.norm(np.asarray(xyz)[:2] - predicted[:2]))
                    gate = 0.7 + 0.18 * min(gap, 5)
                    if distance <= gate:
                        cost[row, column] = 0.75 * distance / gate + 0.25 * (1.0 - overlap)
                elif overlap >= 0.1:
                    cost[row, column] = 0.7 + 0.3 * (1.0 - overlap)

        assignments: dict[int, str] = {}
        matched_gaps: dict[int, int] = {}
        if cost.size:
            safe = np.where(np.isfinite(cost), cost, 99.0)
            rows, columns = linear_sum_assignment(safe)
            for row, column in zip(rows, columns, strict=True):
                if safe[row, column] <= 1.0:
                    track_id, state = active[int(row)]
                    assignments[int(column)] = track_id
                    matched_gaps[int(column)] = max(self.frame - state['last_frame'], 1)
                    self.matches += 1

        outputs: list[dict] = []
        for index, observation in enumerate(observations):
            track_id = assignments.get(index)
            if track_id is None:
                track_id = f'T{self.next_id:04d}'
                self.next_id += 1
                self.births += 1
                self.unmatched_observations += 1
                birth_frame = self.frame
            else:
                birth_frame = self.tracks[track_id]['birth_frame']
                if matched_gaps[index] > 1:
                    self.short_gap_continuations += 1
            previous = self.tracks.get(track_id)
            xyz = observation.get('xyz')
            xyz_array = None if xyz is None else np.asarray(xyz, np.float64)
            velocity = np.zeros(3, np.float64)
            if previous is not None and xyz_array is not None and previous['xyz'] is not None:
                gap = max(self.frame - previous['last_frame'], 1)
                velocity = 0.72 * previous['velocity'] + 0.28 * (xyz_array - previous['xyz']) / gap
            elif previous is not None:
                velocity = previous['velocity'].copy()
            velocity[2] = 0.0
            self.tracks[track_id] = {
                'class': 'PERSON',
                'bbox': np.asarray(observation['bbox'], np.float64).copy(),
                'xyz': None if xyz_array is None else xyz_array.copy(),
                'velocity': velocity,
                'last_frame': self.frame,
                'birth_frame': birth_frame,
            }
            outputs.append({**observation, 'track_id': track_id,
                            'track_age_lidar_frames': self.frame - birth_frame + 1})

        expired = [key for key, state in self.tracks.items()
                   if self.frame - state['last_frame'] > self.max_age]
        for key in expired:
            state = self.tracks.pop(key)
            self.deaths += 1
            self.completed_lifetimes.append(state['last_frame'] - state['birth_frame'] + 1)
        return outputs

    def statistics(self) -> dict:
        active_lifetimes = [state['last_frame'] - state['birth_frame'] + 1
                            for state in self.tracks.values()]
        lifetimes = self.completed_lifetimes + active_lifetimes
        return {
            'max_age_lidar_frames': self.max_age,
            'active_tracks': len(self.tracks),
            'track_births': self.births,
            'track_deaths': self.deaths,
            'matched_observations': self.matches,
            'unmatched_observations': self.unmatched_observations,
            'short_gap_continuations': self.short_gap_continuations,
            'average_lifetime_lidar_frames': None if not lifetimes else float(np.mean(lifetimes)),
            'maximum_lifetime_lidar_frames': max(lifetimes, default=0),
        }

