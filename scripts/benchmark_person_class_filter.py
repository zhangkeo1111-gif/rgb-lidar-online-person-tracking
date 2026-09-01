"""Paired detector audit for full classes versus Ultralytics classes=[0]."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from online_v4 import pipeline, runtime


def stats(values: list[float]) -> dict:
    array = np.asarray(values, np.float64)
    return {name: float(value) for name, value in {
        'mean': array.mean(), 'p95': np.percentile(array, 95),
        'p99': np.percentile(array, 99), 'max': array.max()}.items()}


def person_rows(result) -> np.ndarray:
    if result.boxes is None:
        return np.empty((0, 6), np.float64)
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidence = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    selected = classes == 0
    return np.column_stack((boxes[selected], confidence[selected], classes[selected]))


def predict(model, image: np.ndarray, classes: list[int] | None):
    options = dict(source=image, imgsz=960, conf=0.3, iou=0.6, max_det=12,
                   device='cuda:0' if torch.cuda.is_available() else 'cpu', verbose=False)
    if classes is not None:
        options['classes'] = classes
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = model.predict(**options)[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, 1000.0 * (time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--output', default=str(ROOT / 'outputs/person_only_lightweight/detector_class_filter_audit.json'))
    args = parser.parse_args()
    model = pipeline.YOLO(str(pipeline.MODEL))
    if model.names != {0: 'PERSON', 1: 'ROBOT'}:
        raise RuntimeError(f'Unexpected frozen class mapping: {model.names}')
    warm = np.zeros((720, 1280, 3), np.uint8)
    predict(model, warm, None)
    predict(model, warm, [0])
    times = {'full': [], 'person_only': []}
    exact = changed = full_count = person_count = 0
    for event in runtime.stream_sensor_events(pipeline.DEFAULT_BAG, args.frames, 0):
        image = pipeline.decode_image(event.message)
        order = (('full', None), ('person_only', [0])) if event.index % 2 == 0 else (('person_only', [0]), ('full', None))
        results = {}
        for name, classes in order:
            results[name], elapsed = predict(model, image, classes)
            times[name].append(elapsed)
        full = person_rows(results['full'])
        person = person_rows(results['person_only'])
        full_count += len(full)
        person_count += len(person)
        if full.shape == person.shape and np.array_equal(full, person):
            exact += 1
        else:
            changed += 1
    payload = {'frames': args.frames, 'person_class_index': 0,
               'person_detection_counts': {'full': full_count, 'person_only': person_count},
               'exact_frames': exact, 'changed_frames': changed,
               'runtime_ms': {name: stats(values) for name, values in times.items()}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
