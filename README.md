# Archived Online v4: COCO-Person Pure-Online RGB-Guided LiDAR 3D Tracking

Status: **ARCHIVED / FROZEN 2026-08-29**.

Online v4 contains two explicitly separated detector branches that share the
same 3D core:

| Branch | Detector | Entry | Output |
| --- | --- | --- | --- |
| A | official COCO YOLO11s PERSON | `scripts\run_yolo11s_coco.ps1` | `outputs\online_v4_coco_person` |
| B | official COCO YOLO26s PERSON | `scripts\run_yolo26s_coco.ps1` | `outputs\online_v4_yolo26s_coco_person` |

The A/B comparison is stored separately under
`outputs\yolo11s_vs_yolo26s_coco`. See `ARCHIVE_MANIFEST.json` for frozen
checkpoint hashes and branch ownership.

The repository follows the same top-level layout as Online v2:
`assets/`, `configs/`, `scripts/`, `src/`, `tests/`, `outputs/`, and the single
root entry point `run_online.py`. A/B remain logical detector branches selected
through `--detector`; they are not separate top-level code copies.

`online_v4` is an independent detector-replacement experiment derived from
`D:\detection\versions\06_online_v3`.

## Official COCO detector branches

| Version | RGB PERSON detector |
| --- | --- |
| online_v3 | Scene01-specific `best_detector_fasttrack.pt` |
| online_v4 / `yolo11s-coco` | official Ultralytics COCO-pretrained `yolo11s_coco.pt` |
| online_v4 / `yolo26s-coco` | official Ultralytics COCO-pretrained `yolo26s.pt` |

The checkpoint metadata is read at runtime. The class whose metadata name is
exactly `person` is resolved programmatically and passed to Ultralytics through
`classes=[person_class_id]`. No Scene01, Scene28, or JRDB detector fine-tuning is
performed.

## Frozen shared pipeline

The following are unchanged from online_v3:

- PERSON-only processing; ROBOT disabled;
- no OSNet, ReID prototypes, persistent identity, or G01-G05;
- Scene01 K/D/T and empirical inference registration `(+64,+36) px`;
- RGB-LiDAR synchronization, point ownership, height gates, temporal and
  occupancy suppression, adaptive Euclidean clustering, component scoring and
  visible-component center;
- `ShortTermPersonTracker` with temporary `Txxxx` IDs;
- `CausalMotionBank` and causal RGB-timeline propagation;
- cylinder geometry and 3D Hungarian evaluation with a 1.5 m gate.

The Scene01 empirical registration, temporal background, and occupancy map are
**Scene01-specific assets**. They are not transferable to Scene28 or JRDB.
Non-Scene01 profiles are intentionally blocked until their official calibration,
schema, registration, and background-prior policies have been independently
audited.

## Dataset profiles

- `configs/datasets/scene01.json`: runnable Scene01 adapter.
- `configs/datasets/scene28.json`: non-runnable adapter stub.
- `configs/datasets/jrdb.json`: non-runnable adapter stub; no parameters are
  filled from memory.

## Run

```powershell
cd online_v4

# Select either official detector; yolo11s-coco remains the default.
python run_online.py --detector yolo11s-coco
python run_online.py --detector yolo26s-coco

# Optional PowerShell wrappers under the shared scripts directory
.\scripts\run_yolo11s_coco.ps1
.\scripts\run_yolo26s_coco.ps1

# Full sequence, maximum throughput, no display or video
D:\navwareset_scene01_clean\.venv\Scripts\python.exe run_online.py `
  --detector yolo11s-coco `
  --dataset scene01 --allow-full-sequence --rgb-limit 7490 --lidar-limit 2498 `
  --headless --no-realtime --no-record `
  --output-dir outputs\online_v4_coco_person\uncapped `
  --output-jsonl outputs\online_v4_coco_person\uncapped\online_frames.jsonl

# Full sequence, original sensor-time pacing
D:\navwareset_scene01_clean\.venv\Scripts\python.exe run_online.py `
  --detector yolo11s-coco `
  --dataset scene01 --allow-full-sequence --rgb-limit 7490 --lidar-limit 2498 `
  --headless --realtime --no-record `
  --output-dir outputs\online_v4_coco_person\realtime `
  --output-jsonl outputs\online_v4_coco_person\realtime\online_frames.jsonl
```

Default execution is `headless + no-record`. Prediction is produced without
runtime GT; official 3D GT is read only by post-run evaluation.

## Outputs

Branch A artifacts are isolated under `outputs/online_v4_coco_person/`.
Branch B artifacts are isolated under `outputs/online_v4_yolo26s_coco_person/`.
Cross-branch metrics and contact sheets are isolated under
`outputs/yolo11s_vs_yolo26s_coco/`.
