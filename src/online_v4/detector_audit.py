"""Checkpoint provenance audit for the official COCO PERSON detector."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ultralytics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def payload(pipeline, initialization_ms: float) -> dict:
    names = {str(key): value for key, value in sorted(pipeline.detector_names.items())}
    checkpoint = pipeline.detector.ckpt if isinstance(pipeline.detector.ckpt, dict) else {}
    train_args = checkpoint.get('train_args', {})
    return {
        'detector_id': pipeline.detector_id,
        'detector_family': pipeline.detector_family,
        'official_source': pipeline.detector_official_source,
        'checkpoint_path': str(pipeline.model_path),
        'checkpoint_sha256': sha256(pipeline.model_path),
        'checkpoint_filename': pipeline.model_path.name,
        'model_names': names,
        'person_class_id': int(pipeline.person_class_id),
        'number_of_classes': len(names),
        'ultralytics_version': ultralytics.__version__,
        'task': pipeline.detector.task,
        'model_end2end': bool(getattr(pipeline.detector.model, 'end2end', False)),
        'parameter_count': int(sum(value.numel() for value in pipeline.detector.model.parameters())),
        'checkpoint_metadata': {
            'task': train_args.get('task'),
            'data': train_args.get('data'),
            'model': train_args.get('model'),
            'end2end': train_args.get('end2end'),
        },
        'input_size': 960,
        'inference_parameters': {
            'classes': [int(pipeline.person_class_id)],
            'confidence': 0.30,
            'iou': 0.60,
            'max_det': 12,
        },
        'scene01_detector_fine_tuning': False,
        'model_initialization_ms': float(initialization_ms),
    }


def write(path: Path, pipeline, initialization_ms: float) -> dict:
    result = payload(pipeline, initialization_ms)
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result
