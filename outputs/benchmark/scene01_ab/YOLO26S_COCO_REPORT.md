# Online v4 — Official COCO YOLO11s vs YOLO26s Scene01 A/B

Historical frozen Scene01 comparison. Later cross-scene experiments are outside
this archive; statements below describe the state at this comparison's date.

## Decision

- `BEST_RECALL = YOLO11S_COCO`
- `BEST_SPEED = YOLO26S_COCO`
- `BEST_SPEED_ACCURACY_TRADEOFF = YOLO26S_COCO`
- `GENERALIZATION_PRIMARY_CANDIDATE = YOLO26S_COCO`
- `NEAR_PERSON_RECALL_RISK = YES`

YOLO26s is the stronger primary candidate for a controlled Scene28/JRDB
generalization experiment because it has substantially fewer 3D false
positives, higher F1, lower detector latency, and higher throughput. YOLO11s
must remain the recall-control baseline because YOLO26s loses 0–3 m recall.
No Scene28 or JRDB run was started.

## Official checkpoint audit

The installed Ultralytics `8.4.115` resolved the official detection checkpoint
as `yolo26s.pt` and downloaded it from the Ultralytics assets `v8.4.0` release.
The loaded checkpoint reports:

- task: `detect`;
- 80 COCO classes;
- metadata class `person`: ID `0`, resolved programmatically;
- SHA256: `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`;
- end-to-end model: `true`;
- no Scene01 fine-tuning.

Primary official references:

- <https://docs.ultralytics.com/models/yolo26/>
- <https://github.com/ultralytics/assets/releases/tag/v8.4.0>

The same runtime arguments were passed to both branches:
`imgsz=960`, `conf=0.30`, `iou=0.60`, `max_det=12`, and
`classes=[metadata-resolved person ID]`. YOLO26s is intrinsically end-to-end
and NMS-free, so the common `iou=0.60` argument is recorded but does not create
an NMS stage for its default one-to-one head. No alternate YOLO26 head or
Scene01-specific parameter was selected.

## Frozen experiment controls

Only the detector checkpoint/architecture changed. Both branches used the same:

- Scene01 RGB and LiDAR messages and timestamps;
- PERSON-only mode; ROBOT, OSNet/ReID, G01–G05 disabled;
- K/D/T and Scene01 inference registration;
- RGB–LiDAR synchronization;
- static suppression, CUDA Euclidean clustering and component scoring;
- XYZ definition, `ShortTermPersonTracker`, and `CausalMotionBank`;
- 30 Hz causal propagation;
- frozen VALIDATION split, official 3D GT, one-to-one 3D Hungarian assignment,
  and 1.5 m gate.

Predictions were generated without runtime GT. Both complete prediction logs
were loaded and hashed before the post-run evaluator read official 3D GT.
There is no independent 2D PERSON GT in this project; therefore box-only A/B
differences are not reported as 2D precision/recall.

## Scene01 results

| Metric | YOLO11s COCO | YOLO26s COCO | Change (26s−11s) |
| --- | ---: | ---: | ---: |
| Full RGB PERSON detections | 36,478 | 35,534 | −944 |
| Full propagated PERSON rows | 35,573 | 34,348 | −1,225 |
| VALIDATION predictions | 3,129 | 2,827 | −302 |
| TP | 2,833 | 2,766 | −67 |
| FP | 296 | 61 | **−235** |
| FN | 87 | 154 | +67 |
| Precision | 0.905401 | **0.978422** | +0.073021 |
| Recall | **0.970205** | 0.947260 | −0.022945 |
| F1 | 0.936684 | **0.962589** | +0.025905 |
| 0–3 m Recall | **0.951009** | 0.904899 | −0.046110 |
| 3–6 m Recall | 0.974026 | **0.987013** | +0.012987 |
| 6–9 m Recall | 0.985714 | **1.000000** | +0.014286 |
| >9 m Recall | **0.988448** | 0.984838 | −0.003610 |
| Component availability | 0.997864 | **0.998057** | +0.000192 |
| LiDAR XYZ availability | 0.997864 | **0.998057** | +0.000192 |
| XY RMSE | 0.245134 m | **0.241699 m** | −0.003435 m |
| XY P95 | 0.567313 m | **0.566421 m** | −0.000891 m |

Interpretation:

- YOLO26s reduces 3D FP by 79.4% and raises F1 by 2.59 percentage points.
- Overall recall falls by 2.29 points. The main regression is 0–3 m recall,
  down 4.61 points. Combined 0–6 m recall is approximately 0.9522 for YOLO11s
  and 0.9092 for YOLO26s.
- Component/XYZ availability and matched XY error are effectively stable. The
  accuracy difference is dominated by which people enter the pipeline, not a
  change to 3D component geometry.
- YOLO26s does reduce the YOLO11s FP burden, but it does not preserve total or
  near-range recall.

## Runtime

Detector and LiDAR timings below come from separate complete uncapped runs.
Online latency, sensor-paced FPS, and drops come from separate complete
sensor-paced runs.

| Metric | YOLO11s COCO | YOLO26s COCO | Change (26s−11s) |
| --- | ---: | ---: | ---: |
| YOLO mean | 25.753 ms | **18.357 ms** | −7.395 ms |
| YOLO P95 | 36.878 ms | **29.100 ms** | −7.778 ms |
| YOLO P99 | 44.997 ms | **37.755 ms** | −7.242 ms |
| YOLO max | 194.671 ms | **131.973 ms** | −62.698 ms |
| LiDAR P95 | 33.831 ms | **24.604 ms** | −9.226 ms |
| Sensor-paced online latency P95 | 225.247 ms | **219.532 ms** | −5.715 ms |
| Uncapped FPS | 25.651 | **35.250** | +9.599 |
| Sensor-paced FPS | 26.501 | **27.474** | +0.973 |
| Sensor-paced backlog drops | 869 | **624** | −245 |

YOLO26s lowers detector P95 by 21.1% and raises uncapped throughput by 37.4%.
The sensor-paced run also improves, but still fails the zero-drop 30 Hz gate:
624 of 7,490 RGB frames were dropped. The lower LiDAR P95 is a measured
whole-system scheduling/resource-contention effect; the LiDAR algorithm itself
was frozen and must not be described as an algorithmic LiDAR improvement.

## Required answers

1. Preserve or improve total person Recall? **NO** (`0.970205 → 0.947260`).
2. Preserve near 0–6 m Recall? **NO**, due to the 0–3 m regression.
3. Reduce YOLO11s FP? **YES** (`296 → 61`).
4. Lower YOLO P95? **YES** (`36.878 → 29.100 ms`).
5. Improve overall FPS? **YES** in both uncapped and sensor-paced runs.
6. Ready as zero-drop 30 Hz production detector? **NO**.
7. Better speed–accuracy trade-off on this evaluation? **YOLO26s**, if F1 and
   FP burden are the target. For recall-critical near-person safety, YOLO11s
   remains preferable.

## Artifacts

- `YOLO11S_VS_YOLO26S_COCO_COMPARISON.csv`: exact metric table.
- `YOLO11S_VS_YOLO26S_COCO_METRICS.json`: protocol, hashes, 3D metrics and depth recall.
- `YOLO26S_DETECTOR_AUDIT.json`: checkpoint metadata/provenance.
- `YOLO11S_ONLY_CONTACT_SHEET.jpg`: box-level YOLO11s-only cases.
- `YOLO26S_ONLY_CONTACT_SHEET.jpg`: box-level YOLO26s-only cases.
- `NEAR_PERSON_MISSES_CONTACT_SHEET.jpg`: YOLO26s near-range 3D FN frames.
- `FP_HEAVY_CASES_CONTACT_SHEET.jpg`: frozen VALIDATION 3D FP-heavy frames.

Box-only sheets are qualitative detector A/B evidence, not independent 2D GT.
