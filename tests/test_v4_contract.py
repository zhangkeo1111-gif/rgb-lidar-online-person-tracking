from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT.parent / 'versions' / '06_online_v3'
sys.path.insert(0, str(ROOT / 'src'))

from online_v4 import __version__, dataset_adapter, pipeline
from online_v4.runtime import CausalMotionBank, parser
from online_v4.tracking import ShortTermPersonTracker


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_version_and_official_checkpoint_metadata() -> None:
    assert __version__ == '4.0.0'
    assert pipeline.MODEL.name == 'yolo11s_coco.pt'
    model = YOLO(str(pipeline.MODEL))
    names = {int(key): str(value) for key, value in model.names.items()}
    assert len(names) == 80
    person = [key for key, value in names.items() if value.lower() == 'person']
    assert person == [0]


@pytest.mark.parametrize('detector_id,filename', [
    ('yolo11s-coco', 'yolo11s_coco.pt'),
    ('yolo26s-coco', 'yolo26s.pt'),
])
def test_official_coco_detector_registry(detector_id: str, filename: str) -> None:
    spec = pipeline.DETECTORS[detector_id]
    checkpoint = Path(spec['path'])
    assert checkpoint.name == filename
    assert checkpoint.is_file()
    model = YOLO(str(checkpoint))
    names = {int(key): str(value) for key, value in model.names.items()}
    assert model.task == 'detect'
    assert len(names) == 80
    assert [key for key, value in names.items() if value.lower() == 'person'] == [0]


def test_v4_never_references_scene01_detector() -> None:
    source = (ROOT / 'src' / 'online_v4' / 'pipeline.py').read_text(encoding='utf-8')
    assert 'best_detector_fasttrack.pt' not in source
    assert "detector_options['classes'] = [self.person_class_id]" in source
    assert "options['classes'] = [self.person_class_id]" in source


def test_person_only_defaults_disable_robot_reid_and_identity() -> None:
    args = parser().parse_args([])
    assert args.detector == 'yolo11s-coco'
    assert args.person_only_lightweight is True
    assert args.persistent_identity is False
    source = (ROOT / 'src' / 'online_v4' / 'runtime.py').read_text(encoding='utf-8')
    assert 'online_v4 is PERSON-only' in source
    assert 'online_v4 does not support persistent identity' in source


def test_detector_cli_choices() -> None:
    assert parser().parse_args(['--detector', 'yolo11s-coco']).detector == 'yolo11s-coco'
    assert parser().parse_args(['--detector', 'yolo26s-coco']).detector == 'yolo26s-coco'
    with pytest.raises(SystemExit):
        parser().parse_args(['--detector', 'not-official'])


def test_dataset_adapter_prevents_scene01_prior_leakage() -> None:
    scene01 = dataset_adapter.load_profile(ROOT, 'scene01')
    dataset_adapter.require_runtime_ready(scene01)
    for name in ('scene28', 'jrdb'):
        profile = dataset_adapter.load_profile(ROOT, name)
        assert not any(profile.payload['scene_specific_priors'].values())
        with pytest.raises(RuntimeError):
            dataset_adapter.require_runtime_ready(profile)


def test_short_tracker_source_is_byte_identical_to_v3() -> None:
    if not V3.is_dir():
        pytest.skip('requires the sibling Online v3 archive')
    assert digest(ROOT / 'src' / 'online_v4' / 'tracking.py') == digest(
        V3 / 'src' / 'online_v3' / 'tracking.py')


def test_short_tracker_and_causal_motion() -> None:
    tracker = ShortTermPersonTracker(max_age=10)
    first = tracker.update([{'class': 'PERSON', 'bbox': np.array([0., 0., 10., 20.]),
                             'xyz': np.array([1., 0., 0.])}])[0]
    second = tracker.update([{'class': 'PERSON', 'bbox': np.array([1., 0., 11., 20.]),
                              'xyz': np.array([1.1, 0., 0.]), 'confidence': 0.9}])[0]
    assert first['track_id'] == second['track_id'] == 'T0001'
    bank = CausalMotionBank(180.0)
    bank.correct([second], 1_000_000_000, 0)
    state = bank.snapshot(1_100_000_000)['T0001']
    assert state['prediction_horizon_ms'] == pytest.approx(100.0)
    assert state['fresh'] is True
    assert bank.snapshot(1_200_000_000)['T0001']['fresh'] is False


def test_v3_baseline_still_present_and_versioned() -> None:
    assert (V3 / 'assets' / 'models' / 'detector' / 'best_detector_fasttrack.pt').is_file()
    assert "__version__ = '3.0.0'" in (
        V3 / 'src' / 'online_v3' / '__init__.py').read_text(encoding='utf-8')
