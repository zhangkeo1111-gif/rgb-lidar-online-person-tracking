from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from online_v4 import pipeline
from online_v4.runtime import CausalMotionBank, parser
from online_v4.tracking import ShortTermPersonTracker


def observation(x: float, bbox=(10.0, 10.0, 30.0, 50.0)) -> dict:
    return {'class': 'PERSON', 'bbox': np.asarray(bbox, np.float64),
            'confidence': 0.9, 'xyz': np.asarray([x, 0.0, -1.0]),
            'component_points': 20}


def test_official_coco_detector_class_mapping_is_verified() -> None:
    # The PERSON index is resolved from real checkpoint metadata.
    from ultralytics import YOLO
    names = YOLO(str(pipeline.MODEL)).names
    assert len(names) == 80
    assert [key for key, value in names.items() if value == 'person'] == [0]


def test_short_tracker_rejects_robot() -> None:
    tracker = ShortTermPersonTracker()
    value = observation(1.0)
    value['class'] = 'ROBOT'
    with pytest.raises(ValueError):
        tracker.update([value])


def test_t_id_continuity_and_short_gap() -> None:
    tracker = ShortTermPersonTracker(max_age=10)
    first = tracker.update([observation(1.0)])[0]
    tracker.update([])
    third = tracker.update([observation(1.05)])[0]
    assert first['track_id'] == 'T0001'
    assert third['track_id'] == first['track_id']
    assert tracker.statistics()['short_gap_continuations'] == 1


def test_long_disappearance_creates_new_t_id() -> None:
    tracker = ShortTermPersonTracker(max_age=10)
    old = tracker.update([observation(1.0)])[0]['track_id']
    for _ in range(11):
        tracker.update([])
    new = tracker.update([observation(1.0)])[0]['track_id']
    assert old == 'T0001'
    assert new == 'T0002'
    assert tracker.statistics()['track_deaths'] == 1


def test_causal_motion_exports_velocity_and_prediction() -> None:
    bank = CausalMotionBank(max_horizon_ms=500.0)
    one = {**observation(1.0), 'track_id': 'T0001', 'track_age_lidar_frames': 1}
    two = {**observation(1.1), 'track_id': 'T0001', 'track_age_lidar_frames': 2}
    bank.correct([one], 1_000_000_000, 0)
    bank.correct([two], 1_100_000_000, 1)
    state = bank.snapshot(1_150_000_000)['T0001']
    assert state['state_source'] == 'LIDAR_MEASUREMENT'
    assert state['velocity_mps'][0] == pytest.approx(0.28)
    assert state['xyz'][0] == pytest.approx(1.114)
    assert state['xyz'][2] == pytest.approx(-1.0)


class _Boxes:
    xyxy = torch.tensor([[1.0, 2.0, 30.0, 60.0], [5.0, 6.0, 25.0, 45.0]])
    conf = torch.tensor([0.9, 0.8])
    cls = torch.tensor([0.0, 1.0])


class _Result:
    boxes = _Boxes()


class _Detector:
    def __init__(self) -> None:
        self.options = None

    def predict(self, **options):
        self.options = options
        return [_Result()]


def test_lightweight_yolo_requests_person_class_and_emits_no_robot() -> None:
    value = object.__new__(pipeline.OnlineRgbFrustumPipeline)
    value.person_only_lightweight = True
    value.person_class_id = 0
    value.device = torch.device('cpu')
    value.detector = _Detector()
    detections = value._detections(np.zeros((720, 1280, 3), np.uint8))
    assert value.detector.options['classes'] == [0]
    assert [item['class'] for item in detections] == ['PERSON']


def test_lightweight_cli_defaults_disable_display_and_recording() -> None:
    args: argparse.Namespace = parser().parse_args([])
    assert args.person_only_lightweight is True
    assert args.headless is True
    assert args.no_record is True


def test_v2_source_tree_still_exists() -> None:
    v2 = ROOT.parent / '03_online_v2'
    assert (v2 / 'src' / 'online_v2' / 'runtime.py').is_file()
    assert (v2 / 'assets' / 'models' / 'reid' / 'osnet_x1_0_msmt17_combineall.pth').is_file()

