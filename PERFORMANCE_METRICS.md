# Performance Metrics

## Scope

This is a runtime benchmark, not an accuracy evaluation. Both canonical runs used only the default `TRAIN_FIT` range (2,937 RGB / 979 LiDAR); TEST and EMBARGO were not read.

```powershell
python run_online.py --headless --no-realtime
python run_online.py --headless
```

The measured code is clean commit `490742ce1667baf2186e09c48fcdbf8379a501f8`, with the user-requested global frozen empirical correction `du=48`, `dv=0`. Headless mode still executes rendering; window display and video encoding are disabled.

## Environment

| Item | Value |
| --- | --- |
| OS | Windows 10.0.26200 (`Windows-10-10.0.26200-SP0`) |
| CPU | Intel64 Family 6 Model 186 Stepping 2; 20 logical CPUs |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU; 8,585,216,000 bytes; driver 566.24 |
| Python | 3.11.6 |
| Runtime | PyTorch 2.10.0+cu128; CUDA 12.8; CUDA geometry backend |
| Libraries | Ultralytics 8.4.115; NumPy 2.4.6; OpenCV 5.0.0; SciPy 1.17.1 |
| Sensor duration | 97.896918 s |
| CPU/RAM/GPU utilization | N/A — no bundled per-process GPU sampler with verified attribution; static hardware metadata only |

The canonical uncapped run started on 2026-08-27 at 07:09:30 +08:00. The canonical realtime run started on 2026-08-28 at 01:14:53 +08:00. An intervening realtime attempt was suspended with the desktop task for about 18 hours; it is archived locally as interrupted and excluded from all tracked metrics.

## End-to-end results

| Metric | Uncapped | Original-timestamp realtime |
| --- | ---: | ---: |
| Status | `PURE_ONLINE_30HZ_FAIL` | `PURE_ONLINE_30HZ_FAIL` |
| Wall output rate | **26.046 FPS** | **26.979 FPS** |
| RGB processed / received | 2,937 / 2,937 (100%) | 2,639 / 2,937 (89.854%) |
| Backlog drops | 0 | 298 |
| LiDAR updates / received | 979 / 979 | 943 / 979 |
| LiDAR update rate | 10.000 Hz | 9.632 Hz |
| Wall duration | 112.762 s | 97.816 s |
| Wall / sensor duration | 1.15184 | 0.99918 |
| Online latency mean / P95 / max | 6,759.685 / 13,831.546 / 14,984.613 ms* | 115.535 / 236.434 / 419.584 ms |
| Prediction horizon P95 / max | 92.847 / 179.087 ms | 95.351 / 179.087 ms |

\* In uncapped mode, `online_latency_ms` is relative to the original sensor-time schedule even though pacing is intentionally disabled. Because this run was slower than realtime, that value accumulates and is not an interactive-latency claim. Use wall throughput and `rgb_total` for uncapped performance.

The current device state did not meet 29 FPS. Realtime additionally failed the 99% completeness gate. The result must not be described as stable 30 FPS.

## Module timings

Values are mean / P95 / max in milliseconds. Mean, median, P90, P95, P99 and maximum for both runs are in `outputs/benchmark/metrics_modules.csv`.

| Module | Uncapped | Realtime |
| --- | ---: | ---: |
| RGB decode | 1.118 / 1.476 / 13.144 | 0.840 / 1.241 / 49.595 |
| YOLO | 22.795 / 27.391 / 207.137 | 19.358 / 39.831 / 188.479 |
| Motion prediction | 0.125 / 0.210 / 1.261 | 0.086 / 0.131 / 0.493 |
| RGB association | 0.671 / 0.935 / 4.070 | 0.520 / 0.736 / 110.979 |
| Render | 8.471 / 10.690 / 66.485 | 5.921 / 9.731 / 95.880 |
| Complete RGB path | 55.126 / 76.817 / 259.984 | 48.877 / 82.423 / 240.765 |
| LiDAR decode | 1.346 / 1.586 / 6.605 | 0.952 / 1.437 / 3.658 |
| Transform | 0.463 / 0.632 / 1.612 | 0.338 / 0.552 / 1.430 |
| Projection/frustum gates | 7.292 / 8.435 / 15.891 | 5.615 / 8.971 / 20.367 |
| Static suppression | 2.990 / 4.707 / 9.331 | 3.540 / 12.190 / 59.468 |
| Clustering/component selection | 6.131 / 8.196 / 22.401 | 4.856 / 8.429 / 64.125 |
| Tracking/ReID/identity correction | 10.488 / 37.861 / 145.499 | 8.989 / 37.108 / 105.861 |
| Complete LiDAR path | 28.712 / 58.099 / 175.693 | 24.290 / 58.703 / 148.477 |

Complete-path timings include interleaving and downstream work and are not the arithmetic sum of displayed submodules. Initialization/model loading is outside the per-frame distributions.

## Synchronization and causality

| Check | Uncapped | Realtime |
| --- | ---: | ---: |
| Matched LiDAR updates | 979 | 943 |
| Unmatched LiDAR | 0 | 36 |
| RGB support age mean / P95 / max | 18.490 / 33.180 / 34.994 ms | 18.563 / 33.244 / 34.994 ms |
| Future-RGB support count | **0** | **0** |
| Future LiDAR / interpolation / backward rewrite | 0 / 0 / 0 | 0 / 0 / 0 |

Both runtime paths use the same rule: within the bounded arrival wait, select the latest already-arrived RGB satisfying `rgb_sensor_timestamp <= lidar_sensor_timestamp` and age `<=35 ms`.

## Queues, state, identity, and 3D availability

Render-queue early/late means were 1.0/1.0 in both runs, slope was approximately zero, maximum depth was 1/1, and measured sustained growth was false. Video output was disabled, so writer depth stayed 0/12; recording throughput is not measured.

Canonical uncapped counts:

- `LIDAR_MEASUREMENT`: 5,680 rows; `CAUSAL_PREDICTION`: 11,135 rows.
- Person XYZ available: 14,023 / 14,529 rows; Robot XY available: 2,792 / 2,854 rows. Robot Z remains unavailable.
- Person persistent-identity rows: 13,086; anonymous person rows: 1,443.
- Confirmed→anonymous transitions: 9; fixed-G→different-G transitions: 1; duplicate fixed-G frames: 0.

These are availability/transition counts, not identity accuracy; this benchmark has no independent GT identity evaluation.

## Projection comparison and retention decision

The provenance audit found that `u+48` is empirical and post-test-curated rather than official calibration. The user explicitly requested that its removal be rolled back, so the final code retains exactly one global frozen correction and labels it accordingly. See `PROJECTION_OFFSET_AUDIT.md`.

For transparency, the earlier TRAIN_FIT-only `du=0` diagnostic and final `du=48` uncapped run produced:

| Projection/runtime statistic | Diagnostic `du=0` | Final retained `du=48` |
| --- | ---: | ---: |
| Input points | 27,136,104 | 27,136,104 |
| Projection-valid points | 13,329,144 | 13,329,144 |
| In-view points | 7,083,006 | 7,073,331 |
| Height-gated points | 4,454,263 | 4,462,006 |
| After static suppression | 1,116,075 | 1,095,739 |
| Detections | 5,791 | 5,791 |
| Selected components / measurements | 4,241 (73.234%) | 5,700 (98.429%) |
| Wall output rate | 39.123 FPS | 26.046 FPS |

The point/measurement difference is a real functional effect of projected pixel placement. The speed difference is **not** attributable to a constant 48-pixel addition: the runs occurred under materially different device performance states, and module timings changed broadly. Neither column is an accuracy result because no independent untouched GT was used; the diagnostic is retained only as an engineering comparison, not as the final configuration.

## Machine-readable artifacts

- `outputs/benchmark/metrics_uncapped.json`
- `outputs/benchmark/metrics_realtime.json`
- `outputs/benchmark/metrics_modules.csv`
- `outputs/benchmark/metrics_identity.json`
- `outputs/benchmark/metrics_sync.json`
