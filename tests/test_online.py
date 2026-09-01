from __future__ import annotations
import copy
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = ROOT.parent / '06_online_v3'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from online_v4.identity import PersistentIdentityManager
import audit_a2_correspondence as a2_audit
PROTOTYPES = np.asarray([[1.0, 0.0], [0.0, 1.0]], np.float32)
IDENTITIES = ('G01', 'G02')

def person(track_id: str, x: float | None, y: float=0.0) -> dict:
    xyz = None if x is None else np.asarray([x, y, -1.0], np.float64)
    return {'track_id': track_id, 'class': 'PERSON', 'xyz': xyz, 'bbox': np.asarray([10.0, 10.0, 30.0, 70.0]), 'confidence': 0.9}

def robot(track_id: str) -> dict:
    return {'track_id': track_id, 'class': 'ROBOT', 'xyz': np.asarray([2.0, 0.0, 0.0]), 'bbox': np.asarray([10.0, 10.0, 30.0, 30.0]), 'confidence': 0.9}

def manager() -> PersistentIdentityManager:
    return PersistentIdentityManager(PROTOTYPES, IDENTITIES, max_age=10, nominal_lidar_hz=10.0)

def test_confirmed_track_ignores_single_bad_reid() -> None:
    value = manager()
    first = value.resolve([person('G01', 1.0)], {0: PROTOTYPES[0]}, 1000000000)
    second = value.resolve([person('G01', 1.05)], {0: PROTOTYPES[1]}, 1100000000)
    assert first[0]['persistent_identity'] == 'G01'
    assert second[0]['persistent_identity'] == 'G01'
    assert second[0]['identity_source'] == 'TRACK_INHERITANCE'

def test_short_occlusion_keeps_same_raw_track_lock() -> None:
    value = manager()
    value.resolve([person('G01', 1.0)], {0: PROTOTYPES[0]}, 1000000000)
    value.resolve([], {}, 1100000000)
    recovered = value.resolve([person('G01', 1.1)], {}, 1200000000)
    assert recovered[0]['persistent_identity'] == 'G01'
    assert recovered[0]['identity_source'] == 'SHORT_OCCLUSION_HOLD'

def test_longer_track_break_recovers_only_with_motion_and_appearance() -> None:
    value = manager()
    value.resolve([person('G01', 1.0)], {0: PROTOTYPES[0]}, 1000000000)
    value.resolve([], {}, 1500000000)
    recovered = value.resolve([person('T0001', 1.2)], {0: PROTOTYPES[0]}, 2000000000)
    assert recovered[0]['persistent_identity'] == 'G01'
    assert recovered[0]['identity_state'] == 'RECOVERED'
    assert recovered[0]['identity_source'] == 'CAUSAL_RECOVERY'

def test_recovery_rejects_appearance_only_teleport() -> None:
    value = manager()
    value.resolve([person('G01', 1.0)], {0: PROTOTYPES[0]}, 1000000000)
    value.resolve([], {}, 1100000000)
    rejected = value.resolve([person('T0001', 8.0)], {0: PROTOTYPES[0]}, 1200000000)
    assert rejected[0]['persistent_identity'] is None
    assert value.summary()['recovery_rejected'] == 1

def test_global_assignment_never_duplicates_fixed_identity() -> None:
    value = manager()
    value.resolve([person('G01', 1.0), person('G02', 4.0)], {0: PROTOTYPES[0], 1: PROTOTYPES[1]}, 1000000000)
    value.resolve([], {}, 1100000000)
    rows = value.resolve([person('T0001', 1.1), person('T0002', 1.2)], {0: PROTOTYPES[0], 1: PROTOTYPES[0]}, 1200000000)
    fixed = [row['persistent_identity'] for row in rows if row['persistent_identity'] in IDENTITIES]
    assert fixed == ['G01']

def test_robot_is_separate_and_extra_robot_stays_anonymous() -> None:
    rows = manager().resolve([robot('R1'), robot('T0001')], {}, 1000000000)
    assert rows[0]['persistent_identity'] == 'R1'
    assert rows[1]['persistent_identity'] is None

def test_identity_layer_does_not_modify_raw_track_or_xyz() -> None:
    value = manager()
    source = person('G01', 1.25, -0.4)
    expected_xyz = source['xyz'].copy()
    row = value.resolve([source], {0: PROTOTYPES[0]}, 1000000000)[0]
    assert row['track_id'] == 'G01'
    assert row['base_track_id'] == 'G01'
    np.testing.assert_array_equal(row['xyz'], expected_xyz)

def test_future_reid_or_lidar_cannot_rewrite_past_output() -> None:
    value = manager()
    past = value.resolve([person('G01', 1.0)], {0: PROTOTYPES[0]}, 1000000000)
    frozen_past = copy.deepcopy(past)
    value.resolve([], {}, 1100000000)
    value.resolve([person('T0001', 7.0)], {0: PROTOTYPES[1]}, 1200000000)
    assert past[0]['persistent_identity'] == frozen_past[0]['persistent_identity']
    np.testing.assert_array_equal(past[0]['xyz'], frozen_past[0]['xyz'])
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from online_v4 import pipeline as MODULE, registration, support, suppression

def pipeline_observation(entity_class: str, box: tuple[float, float, float, float], xyz: tuple[float, float, float]):
    return {'class': entity_class, 'bbox': np.asarray(box, float), 'confidence': 0.9, 'xyz': np.asarray(xyz, float), 'component_points': 12}

def test_image_decoder_rgb8_is_bgr():

    class Message:
        width, height, step, encoding = (1, 1, 3, 'rgb8')
        data = np.asarray([255, 0, 0], np.uint8)
    image = MODULE.decode_image(Message())
    assert image.shape == (1, 1, 3)
    assert image[0, 0].tolist() == [0, 0, 255]

def test_online_tracker_keeps_id_without_future_state():
    prototypes = np.eye(2, dtype=np.float32)
    tracker = MODULE.OnlineTracker(prototypes, ('G01', 'G02'), max_age=3)
    first = tracker.update([pipeline_observation('PERSON', (10, 10, 50, 100), (2.0, 0.1, -1.2))], np.asarray([[1.0, 0.0]], np.float32))
    second = tracker.update([pipeline_observation('PERSON', (12, 10, 52, 100), (2.1, 0.1, -1.2))], np.asarray([[0.0, 1.0]], np.float32))
    assert first[0]['track_id'] == 'G01'
    assert second[0]['track_id'] == 'G01'

def test_fixed_ids_are_unique_per_online_frame():
    prototypes = np.eye(2, dtype=np.float32)
    tracker = MODULE.OnlineTracker(prototypes, ('G01', 'G02'))
    rows = tracker.update([pipeline_observation('PERSON', (10, 10, 50, 100), (2.0, 0.0, -1.2)), pipeline_observation('PERSON', (80, 10, 120, 100), (3.0, 0.0, -1.2))], np.eye(2, dtype=np.float32))
    assert {row['track_id'] for row in rows} == {'G01', 'G02'}

def test_global_identity_assignment_resolves_low_local_margin():
    prototypes = np.eye(2, dtype=np.float32)
    tracker = MODULE.OnlineTracker(prototypes, ('G01', 'G02'))
    rows = tracker.update([pipeline_observation('PERSON', (10, 10, 50, 100), (2.0, 0.0, -1.2)), pipeline_observation('PERSON', (80, 10, 120, 100), (3.0, 0.0, -1.2))], np.asarray([[0.71, 0.7], [0.05, 0.98]], np.float32))
    assert [row['track_id'] for row in rows] == ['G01', 'G02']

def test_robot_visible_surface_is_not_official_z():
    row = {'track_id': 'R1', **pipeline_observation('ROBOT', (10, 10, 50, 60), (4.0, 0.0, -1.6))}
    result = MODULE.OnlineRgbFrustumPipeline.serializable(row)
    assert result['xyz_semantics'] == 'ROBOT_VISIBLE_SURFACE_COMPONENT_CENTER_NOT_OFFICIAL_ORIGIN'
    assert 'official' not in result['xyz_semantics'].lower() or 'not_official' in result['xyz_semantics'].lower()

def test_robot_keeps_r1_after_out_of_gate_measurement():
    tracker = MODULE.OnlineTracker(np.empty((0, 0), np.float32), ())
    first = tracker.update([pipeline_observation('ROBOT', (10, 10, 50, 60), (2.0, 0.0, -1.6))])
    second = tracker.update([pipeline_observation('ROBOT', (200, 200, 260, 280), (5.0, 2.0, -1.6))])
    assert first[0]['track_id'] == 'R1'
    assert second[0]['track_id'] == 'R1'

def test_duplicate_robot_detection_has_only_one_r1():
    tracker = MODULE.OnlineTracker(np.empty((0, 0), np.float32), ())
    first = pipeline_observation('ROBOT', (10, 10, 50, 60), (2.0, 0.0, -1.6))
    second = pipeline_observation('ROBOT', (80, 10, 120, 60), (4.0, 0.0, -1.6))
    second['confidence'] = 0.7
    result = tracker.update([first, second])
    assert [item['track_id'] for item in result].count('R1') == 1

def test_anonymous_person_recovers_available_frozen_identity():
    tracker = MODULE.OnlineTracker(np.eye(1, dtype=np.float32), ('G01',), max_age=1)
    feature = np.asarray([[1.0]], np.float32)
    original = pipeline_observation('PERSON', (10, 10, 50, 100), (2.0, 0.0, -1.2))
    returned = pipeline_observation('PERSON', (200, 10, 250, 100), (5.0, 0.0, -1.2))
    assert tracker.update([original], feature)[0]['track_id'] == 'G01'
    assert tracker.update([returned], feature)[0]['track_id'].startswith('T')
    assert tracker.update([returned], feature)[0]['track_id'].startswith('T')
    assert tracker.update([returned], feature)[0]['track_id'] == 'G01'

def test_iou_xyxy_identity_and_disjoint():
    box = np.asarray([1, 2, 5, 8], float)
    assert MODULE.iou_xyxy(box, box) == 1.0
    assert MODULE.iou_xyxy(box, np.asarray([10, 10, 20, 20], float)) == 0.0

def test_online_default_stops_before_embargo():
    assert MODULE.parser().parse_args([]).limit == 979

def test_compatibility_sync_selects_latest_past_rgb() -> None:
    images = [(80_000_000, np.zeros((1, 1, 3), np.uint8)), (105_000_000, np.full((1, 1, 3), 2, np.uint8))]
    selected = MODULE.select_causal_rgb(images, 100_000_000, 35_000_000)
    assert selected is not None and selected[0] == 80_000_000

def test_compatibility_sync_rejects_future_only_rgb() -> None:
    assert MODULE.select_causal_rgb([(105_000_000, np.zeros((1, 1, 3), np.uint8))], 100_000_000, 35_000_000) is None

def test_shared_sync_boundary_rejects_36ms_and_accepts_35ms() -> None:
    with pytest.raises(ValueError, match='synchronization'):
        support.causal_support_delta_ms(64_000_000, 100_000_000, 35.0)
    assert support.causal_support_delta_ms(65_000_000, 100_000_000, 35.0) == pytest.approx(35.0)
    assert online30.causal_support_delta_ms(65_000_000, 100_000_000, 35.0) == pytest.approx(35.0)

def test_physical_projection_is_zero_shift_and_legacy_shift_is_explicit() -> None:
    points = np.asarray([[0.2, -0.1, 4.0], [0.0, 0.0, 2.0]], np.float64)
    transform = np.eye(4, dtype=np.float64)
    K = np.asarray([[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]])
    D = np.zeros(5)
    actual, valid, _ = support.project_points(points, transform, K, D)
    expected = np.column_stack((600.0 * points[:, 0] / points[:, 2] + 640.0, 600.0 * points[:, 1] / points[:, 2] + 360.0))
    assert valid.all()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    legacy, _, _ = support.project_points(points, transform, K, D, du_px=48.0)
    np.testing.assert_allclose(legacy[:, 0] - actual[:, 0], 48.0, rtol=0, atol=1e-10)
    assert support.PHYSICAL_PROJECTION['du_px'] == 0.0
    assert support.PHYSICAL_PROJECTION['dv_px'] == 0.0
    assert support.PHYSICAL_PROJECTION['calibration_status'] == 'NOT_PROVEN_FINAL'
    assert support.LEGACY_DISPLAY_REGISTRATION['du_px'] == 48.0
    assert support.LEGACY_DISPLAY_REGISTRATION['inference_effect'] is False
    assert support.EMPIRICAL_INFERENCE_REGISTRATION['enabled'] is True
    assert support.EMPIRICAL_INFERENCE_REGISTRATION['du_px'] == 64.0
    assert support.EMPIRICAL_INFERENCE_REGISTRATION['dv_px'] == 36.0
    assert support.EMPIRICAL_INFERENCE_REGISTRATION['physical_calibration_update'] is False
    assert support.DISPLAY_OVERLAY_REGISTRATION['du_px'] == 60.0
    assert support.DISPLAY_OVERLAY_REGISTRATION['dv_px'] == 17.0
    assert support.DISPLAY_OVERLAY_REGISTRATION['inference_effect'] is False
    assert support.REGISTRATION_CONFIG['mode'] == 'PROJECTION_RESIDUAL_AUDIT_REPORT_ONLY'

def test_source_has_no_hidden_per_frame_or_identity_pixel_shift() -> None:
    source_root = ROOT / 'src' / 'online_v4'
    source = '\n'.join(path.read_text(encoding='utf-8') for path in source_root.glob('*.py'))
    forbidden = ('PROJECTION_DU_PX', 'PROJECTION_DV_PX', 'per_id_pixel', 'per_frame_pixel', 'ground_truth_pixel', 'pixels[valid, 0] +=', 'pixels[valid, 1] +=')
    assert all(token not in source for token in forbidden)
    config = support.REGISTRATION_CONFIG
    assert config['physical_baseline_du_px'] == 0.0
    assert config['physical_baseline_dv_px'] == 0.0
    assert config['candidate_applied_to_inference'] is False
    assert config['empirical_inference_du_px'] == 64.0
    assert config['empirical_inference_dv_px'] == 36.0
    assert config['empirical_inference_scope'] == 'SCENE01_ONLY'
    assert config['legacy_display_du_px'] == 48.0
    assert config['legacy_display_enabled'] is False
    assert config['fit_fraction'] < 1.0

def test_projection_residual_audit_never_writes_candidate_to_inference() -> None:
    config = dict(support.REGISTRATION_CONFIG)
    transform = np.eye(4, dtype=np.float64)
    K = np.asarray([[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]])
    disabled = registration.ProjectionResidualAudit(transform, K, np.zeros(5), config, enabled=False)
    assert disabled.snapshot()['state'] == 'AUDIT_DISABLED_PHYSICAL_BASELINE'
    assert disabled.du_px == pytest.approx(0.0)

    audit = registration.ProjectionResidualAudit(transform, K, np.zeros(5), config, warmup_frames=1)
    audit._estimate = lambda frames: {
        'status': 'PASS', 'diagnostic_candidate_du_px': 36.0,
        'physical_baseline_du_px': 0.0, 'candidate_applied_to_inference': False,
        'checks': {'synthetic_gate': True}, 'manual_correspondences_used': False,
        'ground_truth_used': False, 'future_frames_used_after_freeze': False,
        'per_frame_correction': False,
    }
    audit.observe(np.zeros((720, 1280, 3), np.uint8), [], np.empty((0, 3)), [])
    assert audit.du_px == pytest.approx(0.0)
    audit.close()
    snapshot = audit.snapshot()
    assert snapshot['state'] == 'AUDIT_CANDIDATE_PASS_REPORT_ONLY'
    assert snapshot['result']['diagnostic_candidate_du_px'] == pytest.approx(36.0)
    assert snapshot['physical_inference']['du_px'] == pytest.approx(0.0)
    assert snapshot['physical_inference']['audit_candidate_applied'] is False
    audit.observe(np.ones((720, 1280, 3), np.uint8), [], np.empty((0, 3)), [])
    assert audit.du_px == pytest.approx(0.0)

def test_runtime_source_separates_physical_inference_and_display_projection() -> None:
    pipeline_source = (ROOT / 'src' / 'online_v4' / 'pipeline.py').read_text(encoding='utf-8')
    runtime_source = (ROOT / 'src' / 'online_v4' / 'runtime.py').read_text(encoding='utf-8')
    assert 'def project_physical' in pipeline_source
    assert 'def project_inference' in pipeline_source
    assert 'def project_display' in pipeline_source
    assert "du_px=0.0" in pipeline_source
    assert "pipeline.project_inference(points)" in runtime_source
    assert "candidate_applied_to_inference': False" in (ROOT / 'src' / 'online_v4' / 'registration.py').read_text(encoding='utf-8')


def test_empirical_inference_shift_is_exact_and_does_not_change_raw_projection() -> None:
    from types import SimpleNamespace
    value = object.__new__(MODULE.OnlineRgbFrustumPipeline)
    value.transform = np.eye(4, dtype=np.float64)
    value.K = np.asarray([[600.0, 0.0, 640.0],
                          [0.0, 600.0, 360.0],
                          [0.0, 0.0, 1.0]])
    value.D = np.zeros(5)
    value.inference_du_px = 64.0
    value.inference_dv_px = 36.0
    value.display_du_px = 60.0
    value.display_dv_px = 17.0
    value.calibration_audit = SimpleNamespace(legacy_display_enabled=False)
    points = np.asarray([[0.2, -0.1, 4.0], [0.0, 0.0, 2.0]])
    raw, raw_valid, _ = value.project_physical(points)
    inferred, inferred_valid, _ = value.project_inference(points)
    displayed, displayed_valid, _ = value.project_display(points)
    assert raw_valid.all() and inferred_valid.all() and displayed_valid.all()
    np.testing.assert_allclose(
        inferred - raw, np.tile([64.0, 36.0], (len(points), 1)),
        rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        displayed - raw, np.tile([60.0, 17.0], (len(points), 1)),
        rtol=0, atol=1e-12)
    value.calibration_audit = SimpleNamespace(
        legacy_display_enabled=True, display_du_px=48.0, display_dv_px=0.0)
    legacy_display, _, _ = value.project_display(points)
    np.testing.assert_allclose(
        legacy_display - raw, np.tile([48.0, 0.0], (len(points), 1)),
        rtol=0, atol=1e-12)


def test_empirical_offset_audit_passed_and_is_integrated_scene01_only() -> None:
    summary = json.loads((V3_ROOT / 'outputs' / 'empirical_pixel_offset_audit' /
                          'audit_summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'PASS'
    assert summary['train_fit_selected_candidate'] == {'du_px': 64.0, 'dv_px': 36.0}
    assert summary['selected_empirical_offset'] == {'du_px': 64.0, 'dv_px': 36.0}
    assert summary['candidate_applied_to_online'] is True
    assert summary['gates']['raw_baseline_fingerprint_exact'] is True
    assert summary['test_access'] == 0 and summary['embargo_access'] == 0
import inspect
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from online_v4 import runtime as online30

def motion_observation(x: float, track_id: str='G01', entity_class: str='PERSON') -> dict:
    return {'track_id': track_id, 'class': entity_class, 'xyz': np.asarray([x, 0.0, 0.8], dtype=np.float64), 'bbox': np.asarray([10.0, 20.0, 30.0, 80.0], dtype=np.float64), 'confidence': 0.9}

def test_future_lidar_isolation_and_no_backward_rewrite() -> None:
    bank = online30.CausalMotionBank(max_horizon_ms=180.0)
    bank.correct([motion_observation(1.0)], 1000000000, 0)
    emitted = bank.snapshot(1033000000)['G01']
    emitted_xyz = emitted['xyz'].copy()
    bank.correct([motion_observation(8.0)], 1100000000, 1)
    np.testing.assert_array_equal(emitted['xyz'], emitted_xyz)
    assert emitted['measurement_timestamp_ns'] == 1000000000

def test_future_rgb_support_is_rejected() -> None:
    with pytest.raises(ValueError, match='future RGB'):
        online30.causal_support_delta_ms(1001000000, 1000000000, 35.0)

def test_old_rgb_support_outside_sync_window_is_rejected() -> None:
    with pytest.raises(ValueError, match='synchronization'):
        online30.causal_support_delta_ms(900000000, 1000000000, 35.0)

def test_causal_prediction_source_and_person_z_hold() -> None:
    bank = online30.CausalMotionBank(max_horizon_ms=180.0)
    bank.correct([motion_observation(1.0)], 1000000000, 0)
    measured = bank.snapshot(1000000000)['G01']
    assert measured['state_source'] == 'LIDAR_MEASUREMENT'
    bank.mark_reported(['G01'])
    bank.correct([motion_observation(1.1)], 1100000000, 1)
    bank.mark_reported(['G01'])
    predicted = bank.snapshot(1133000000)['G01']
    assert predicted['state_source'] == 'CAUSAL_PREDICTION'
    assert predicted['xyz'][0] > 1.1
    assert predicted['xyz'][2] == pytest.approx(0.8)
    assert predicted['prediction_horizon_ms'] == pytest.approx(33.0)

def test_prediction_becomes_stale_after_bounded_horizon() -> None:
    bank = online30.CausalMotionBank(max_horizon_ms=180.0)
    bank.correct([motion_observation(1.0)], 1000000000, 0)
    assert bank.snapshot(1180000000)['G01']['fresh'] is True
    assert bank.snapshot(1181000000)['G01']['fresh'] is False

def test_streaming_reader_is_generator_and_no_offline_densification() -> None:
    assert inspect.isgeneratorfunction(online30.stream_sensor_events)
    source = Path(online30.__file__).read_text(encoding='utf-8').lower()
    forbidden = ('person_tracking_30hz', 'rgb_to_canonical_anchor_map', 'np.interp(', 'interpolate_track', 'read_csv(')
    assert all((token not in source for token in forbidden))

def test_timestamp_monotonicity_audit() -> None:
    assert online30.strictly_monotonic([10, 20, 30])
    assert not online30.strictly_monotonic([10, 10, 30])
    assert not online30.strictly_monotonic([10, 9, 30])

def test_queue_growth_is_measured_not_hardcoded() -> None:
    stable = online30.queue_growth([0, 1] * 20)
    rising = online30.queue_growth([0] * 10 + list(range(1, 21)))
    assert stable['sustained_growth'] is False
    assert rising['late_mean'] > rising['early_mean']
    assert rising['slope_per_sample'] > 0
    assert rising['sustained_growth'] is True

def test_frozen_train_fit_limits_and_realtime_defaults() -> None:
    args = online30.parser().parse_args([])
    assert args.rgb_limit == 2937
    assert args.lidar_limit == 979
    assert args.realtime is True
    assert args.max_prediction_horizon_ms == pytest.approx(180.0)
    assert args.backlog_drop_ms == pytest.approx(180.0)
    assert args.suppressed_recovery is True

def test_robot_state_has_no_z_velocity() -> None:
    bank = online30.CausalMotionBank(max_horizon_ms=180.0)
    bank.correct([motion_observation(1.0, 'R1', 'ROBOT')], 1000000000, 0)
    bank.correct([motion_observation(1.1, 'R1', 'ROBOT')], 1100000000, 1)
    assert bank.states['R1']['velocity_mps'][2] == 0.0


def test_suppressed_recovery_uses_existing_motion_gate_and_is_causal() -> None:
    from online_v4.recovery import CausalSuppressedPointRecovery, RecoveryCandidate

    manager = CausalSuppressedPointRecovery(allow_causal_seed=False)
    candidate = RecoveryCandidate(
        np.asarray([1.1, 2.0, -1.2]), 8, 0.4, True, True, False, False)
    assert manager.consider('T0001', candidate, 0)['reason'] == 'NO_METRIC_HISTORY'
    manager.observe_normal('T0001', np.asarray([1.0, 2.0, -1.2]), 0)
    accepted = manager.consider('T0001', candidate, 1)
    assert accepted['accepted']
    assert accepted['motion_gate_m'] == pytest.approx(0.88)
    far = RecoveryCandidate(
        np.asarray([4.0, 2.0, -1.2]), 8, 0.4, True, True, False, False)
    assert manager.consider('T0001', far, 2)['reason'] == 'HISTORY_MOTION_GATE_REJECT'


def test_online_pipeline_applies_only_history_backed_suppressed_recovery() -> None:
    from types import SimpleNamespace
    from online_v4.recovery import CausalSuppressedPointRecovery, RecoveryCandidate

    value = object.__new__(MODULE.OnlineRgbFrustumPipeline)
    value.suppressed_recovery_enabled = True
    value.suppressed_recovery = CausalSuppressedPointRecovery(False)
    value.tracker = SimpleNamespace(tracks={'T0001': {'xyz': None}})
    normal = person('T0001', 1.0)
    value.apply_suppressed_recovery([normal], [None], 0)

    missing = person('T0001', None)
    candidate = RecoveryCandidate(
        np.asarray([1.1, 0.0, -1.0]), 8, 0.4, True, True, False, False)
    counts = value.apply_suppressed_recovery([missing], [candidate], 1)
    assert counts == {'attempts': 1, 'accepted': 1, 'rejected': 0}
    assert missing['measurement_source'] == 'RECOVERED_SUPPRESSED_COMPONENT'
    np.testing.assert_allclose(missing['xyz'], candidate.xyz)
    np.testing.assert_allclose(value.tracker.tracks['T0001']['xyz'], candidate.xyz)


def test_suppressed_recovery_seed_never_backfills_and_rejects_neighbor() -> None:
    from online_v4.recovery import CausalSuppressedPointRecovery, RecoveryCandidate

    manager = CausalSuppressedPointRecovery(allow_causal_seed=True)
    good = RecoveryCandidate(
        np.asarray([1.0, 2.0, -1.2]), 8, 0.4, True, True, False, False)
    assert manager.consider('T0001', good, 0)['reason'] == 'CAUSAL_SEED_PENDING'
    assert manager.consider('T0001', good, 1)['reason'] == 'CAUSAL_SEED_PENDING'
    assert manager.consider('T0001', good, 2)['reason'] == 'CAUSAL_SEED_CONFIRMED'
    competing = RecoveryCandidate(
        np.asarray([1.0, 2.0, -1.2]), 8, 0.4, True, True, True, False)
    assert manager.consider('T0002', competing, 0)['reason'] == 'NEIGHBOR_PERSON_COMPETITION'


def test_strict_recovery_requires_frozen_normal_component_consistency() -> None:
    from online_v4.recovery import (
        CausalSuppressedPointRecovery, ComponentSignature,
        FrozenConsistencyThresholds, RecoveryCandidate)

    thresholds = FrozenConsistencyThresholds(
        innovation_m=0.25, horizontal_log_ratio=0.2,
        vertical_log_ratio=0.2, point_count_log_ratio=0.3,
        center_u_norm_delta=0.1, center_v_norm_delta=0.1)
    signature = ComponentSignature(0.5, 1.5, 20, 0.0, 0.0)
    manager = CausalSuppressedPointRecovery(
        allow_causal_seed=False, consistency_thresholds=thresholds)
    manager.observe_normal('T0001', np.asarray([1.0, 2.0, -1.2]), 0, signature)
    good = RecoveryCandidate(
        np.asarray([1.1, 2.0, -1.2]), 18, 0.4, True, True, False, False,
        ComponentSignature(0.52, 1.48, 18, 0.03, -0.02))
    assert manager.consider('T0001', good, 1)['accepted']

    bad_shape = RecoveryCandidate(
        np.asarray([1.15, 2.0, -1.2]), 5, 0.4, True, True, False, False,
        ComponentSignature(0.15, 0.3, 5, 0.4, 0.4))
    result = manager.consider('T0001', bad_shape, 2)
    assert result['reason'] == 'HISTORY_COMPONENT_SIGNATURE_REJECT'


def test_ground_height_is_signed_metric_plane_distance() -> None:
    normal = np.asarray([0.0, 0.0, 2.0])
    points = np.asarray([[2.0, -3.0, -1.0], [2.0, -3.0, 0.0]])
    height = suppression.ground_height(points, normal, 2.0)
    np.testing.assert_allclose(height, [0.0, 1.0], atol=1e-12)


def test_person_middle_height_is_not_occupancy_hard_deleted() -> None:
    points = np.asarray([[2.0, 0.0, -1.0]])
    height = np.asarray([1.0])
    temporal = np.asarray([True])
    near = np.asarray([True])
    legacy, _ = suppression.occupancy_keep(
        points, height, near, temporal,
        suppression.Policy(suppression.LEGACY))
    physical, _ = suppression.occupancy_keep(
        points, height, near, temporal,
        suppression.Policy(suppression.GROUND_LIMITED, 0.25))
    assert not legacy[0]
    assert physical[0]


def test_static_ground_is_suppressed_and_temporal_remains_primary() -> None:
    points = np.asarray([[2.0, 0.0, -2.0], [2.1, 0.0, -1.0]])
    height = np.asarray([0.10, 1.0])
    temporal = np.asarray([True, False])
    near = np.asarray([True, False])
    occupancy, _ = suppression.occupancy_keep(
        points, height, near, temporal,
        suppression.Policy(suppression.GROUND_LIMITED, 0.25))
    np.testing.assert_array_equal(occupancy, [False, True])
    np.testing.assert_array_equal(
        suppression.combined_keep(temporal, occupancy), [False, False])


def test_person_conditional_protection_is_exact_owner_only() -> None:
    points = np.asarray([[2.0, 0.0, -2.0], [2.0, 0.1, -2.0],
                         [2.0, 0.2, -2.0]])
    height = np.asarray([0.10, 0.10, 0.10])
    temporal = np.asarray([True, False, True])
    near = np.asarray([True, True, True])
    owners = np.asarray([0, 0, 1], np.int32)
    keep, counts = suppression.occupancy_keep(
        points, height, near, temporal,
        suppression.Policy(suppression.PERSON_CONDITIONAL),
        owners, ('PERSON', 'ROBOT'))
    np.testing.assert_array_equal(keep, [True, False, False])
    assert counts['person_box_restored_points'] == 1


def test_exclusive_ownership_still_assigns_each_point_at_most_once() -> None:
    pixels = np.asarray([[15.0, 15.0], [25.0, 15.0], [80.0, 80.0]])
    boxes = [np.asarray([0.0, 0.0, 30.0, 30.0]),
             np.asarray([10.0, 0.0, 40.0, 30.0])]
    owners = support.exclusive_point_owners(pixels, boxes, [None, None])
    assert owners.shape == (3,)
    assert set(owners.tolist()) <= {-1, 0, 1}


def test_unaffected_normal_mask_is_unchanged() -> None:
    points = np.asarray([[2.0, 0.0, -1.0]])
    height = np.asarray([1.0])
    temporal = np.asarray([True])
    near = np.asarray([False])
    legacy, _ = suppression.occupancy_keep(
        points, height, near, temporal,
        suppression.Policy(suppression.LEGACY))
    redesigned, _ = suppression.occupancy_keep(
        points, height, near, temporal,
        suppression.Policy(suppression.GROUND_LIMITED, 0.25))
    np.testing.assert_array_equal(legacy, redesigned)


def test_suppression_module_has_no_future_or_gt_dependency() -> None:
    parameters = set(inspect.signature(suppression.occupancy_keep).parameters)
    assert parameters == {
        'points', 'height_m', 'occupancy_near', 'temporal_keep', 'policy',
        'owners', 'owner_classes'}


def test_cpu_cuda_suppression_and_geometry_equivalence() -> None:
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from scipy.spatial import cKDTree

    static = {(0, 0, 0), (10, 0, 0)}
    occupancy = np.asarray([[0.02, 0.02], [1.02, 0.02]], np.float64)
    points = np.asarray([[0.02, 0.02, 0.02], [0.03, 0.02, 0.03],
                         [0.04, 0.02, 0.04], [1.02, 0.02, 0.15],
                         [1.03, 0.02, 0.16], [1.04, 0.02, 0.17]], np.float64)
    value = object.__new__(MODULE.OnlineRgbFrustumPipeline)
    value.geometry_gpu = None
    value.static = static
    value.voxel_size = 0.10
    value.occupancy_tree = cKDTree(occupancy)
    cpu_temporal, cpu_near = value._static_evidence(points)

    gpu = support.GpuFrustumBackend(static, 0.10, occupancy)
    gpu_temporal, gpu_near = gpu.static_evidence(points)
    np.testing.assert_array_equal(gpu_temporal, cpu_temporal)
    np.testing.assert_array_equal(gpu_near, cpu_near)
    height = np.asarray([0.05, 0.06, 0.07, 0.25, 0.26, 0.27])
    cpu_legacy_temporal, cpu_legacy_keep, cpu_removed = value._static_masks(
        points, height)
    gpu_legacy_temporal, gpu_legacy_keep, gpu_removed = gpu.static_masks(
        points, height)
    np.testing.assert_array_equal(gpu_legacy_temporal, cpu_legacy_temporal)
    np.testing.assert_array_equal(
        gpu_legacy_temporal & gpu_legacy_keep,
        cpu_legacy_temporal & cpu_legacy_keep)
    assert gpu_removed == cpu_removed
    policy = suppression.Policy(suppression.GROUND_LIMITED, 0.25)
    cpu_keep, _ = suppression.occupancy_keep(
        points, height, cpu_near, cpu_temporal, policy)
    gpu_keep, _ = suppression.occupancy_keep(
        points, height, gpu_near, gpu_temporal, policy)
    np.testing.assert_array_equal(gpu_keep, cpu_keep)

    pixels = np.asarray([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0],
                         [60.0, 60.0], [61.0, 61.0], [62.0, 62.0]])
    boxes = [np.asarray([0.0, 0.0, 30.0, 30.0]),
             np.asarray([50.0, 50.0, 80.0, 80.0])]
    cpu_owners = support.exclusive_point_owners(pixels, boxes, [None, None])
    gpu_owners = gpu.owners(pixels, boxes)
    np.testing.assert_array_equal(gpu_owners, cpu_owners)
    cpu_groups = [support.adaptive_components(points[cpu_owners == index])
                  for index in range(2)]
    gpu_groups = gpu.components(points, gpu_owners, 2)
    for cpu_components, gpu_components in zip(cpu_groups, gpu_groups, strict=True):
        assert len(cpu_components) == len(gpu_components)
        for left, right in zip(cpu_components, gpu_components, strict=True):
            np.testing.assert_allclose(
                support.cluster_center(left), support.cluster_center(right),
                rtol=0, atol=1e-12)


def test_fn_root_cause_audit_is_exclusive_complete_and_frozen() -> None:
    output = V3_ROOT / 'outputs' / 'fn_root_cause_audit'
    summary = json.loads((output / 'audit_summary.json').read_text(encoding='utf-8'))
    with (output / 'fn_case_details.csv').open(encoding='utf-8-sig', newline='') as stream:
        cases = list(csv.DictReader(stream))
    assert len(cases) == 909
    assert all(row['primary_root_cause'] for row in cases)
    assert sum(summary['taxonomy']['V1_by_primary'].values()) == 909
    assert summary['metrics']['V0']['fn'] == 976
    assert summary['metrics']['V1']['fn'] == 909
    assert summary['old_880_plus_96_taxonomy_belongs_to'] == 'V0_LEGACY_BASELINE'


def test_fn_root_cause_audit_gt_and_holdout_isolation() -> None:
    output = V3_ROOT / 'outputs' / 'fn_root_cause_audit'
    summary = json.loads((output / 'audit_summary.json').read_text(encoding='utf-8'))
    assert summary['safety']['gt_loaded_after_predictions'] is True
    assert summary['safety']['instrumentation_predictions_exact'] is True
    assert summary['safety']['test_access'] == 0
    assert summary['safety']['embargo_access'] == 0
    assert summary['projection'] == 'RAW_PHYSICAL_K_D_T_DU_DV_ZERO_BASELINE'


def test_a2_cuboid_proxy_uses_all_eight_projected_corners() -> None:
    class IdentityProjection:
        @staticmethod
        def project_physical(points):
            values = np.asarray(points, np.float64)
            return values[:, :2], np.ones(len(values), bool), values[:, 2]

    target = {
        'xyz': np.asarray([10.0, 20.0, 30.0]),
        'size': np.asarray([2.0, 4.0, 6.0]),
        'yaw': 0.0,
    }
    result = a2_audit.proxy_projection(target, IdentityProjection())
    assert result['corner_pixels'].shape == (8, 2)
    np.testing.assert_allclose(result['proxy'], [9.0, 18.0, 11.0, 22.0])
    np.testing.assert_allclose([result['center_u'], result['center_v']], [10.0, 20.0])


def test_a2_audit_is_complete_exclusive_and_frozen() -> None:
    output = V3_ROOT / 'outputs' / 'a2_correspondence_decomposition'
    summary = json.loads((output / 'audit_summary.json').read_text(encoding='utf-8'))
    prior = json.loads((V3_ROOT / 'outputs' / 'fn_root_cause_audit' /
                        'audit_summary.json').read_text(encoding='utf-8'))
    counts = summary['subtype_counts']
    assert set(counts) == set(a2_audit.SUBTYPES)
    assert sum(counts.values()) == 770
    assert summary['frozen_gate']['V1_FN'] == 909
    assert summary['frozen_gate']['A2'] == 770
    assert summary['frozen_gate']['prediction_fingerprints'] == prior['fingerprints']
    assert summary['rules']['source'] == 'TRAIN_FIT_ONLY_BEFORE_VALIDATION_CLASSIFICATION'
    assert summary['rules']['validation_updates'] == 0
    assert summary['tests']['taxonomy_exclusive'] is True
    assert summary['tests']['taxonomy_complete_770'] is True


def test_a2_audit_has_no_gt_runtime_or_calibration_writeback() -> None:
    output = V3_ROOT / 'outputs' / 'a2_correspondence_decomposition'
    summary = json.loads((output / 'audit_summary.json').read_text(encoding='utf-8'))
    receipt = json.loads((output / 'freeze_receipt_before_gt.json').read_text(encoding='utf-8'))
    assert receipt['gt_loaded'] is False
    assert receipt['physical_du_px'] == 0.0
    assert receipt['candidate_applied_to_inference'] is False
    assert summary['candidate_applied_to_inference'] is False
    assert summary['scope']['TEST_access'] == 0
    assert summary['scope']['EMBARGO_access'] == 0
    assert summary['tests']['instrumentation_equivalent'] is True
    assert summary['tests']['no_calibration_writeback'] is True
