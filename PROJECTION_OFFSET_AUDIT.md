# Projection Offset Audit

## Current decision

The runtime now separates three independent chains:

```text
Physical audit baseline
raw K + D + T, du=dv=0
-> diagnostics / reproducibility reference only

Scene01 empirical inference
raw K + D + T -> (+64,+36) px
-> image/frustum gate -> point ownership -> component selection
-> 3D–2D association -> XYZ / tracking

Projection residual audit
raw physical projection -> causal residual diagnostics
-> chronological holdout -> REPORT ONLY

legacy visualization -> optional (+48,0) px -> DISPLAY/DEBUG ONLY
```

`du=0` is a **raw physical-calibration baseline**, not a claim that the bundled
calibration is proven final. Audit candidates never modify inference. The
legacy `+48 px` display option is disabled by default and cannot affect point
ownership, cluster selection, association, XYZ or tracking.

## Provenance of `+48 px`

The earliest located record is the 2026-08-13 empirical Scene01 audit under
`D:\navwareset_scene01_clean\outputs\projection_horizontal_offset_rootcause`.
The available record used 247 manually curated identity boxes and full-scene
post-test display material. It reported correct-box inclusion changing from
`72/247` to `243/247` after shifting projections right by 48 pixels.

This evidence is useful for display diagnosis but is not an independent sensor
calibration:

- it used manual, GT-like 2D references;
- independence from TEST/EMBARGO inspection is not established;
- it was not shown to be frozen before formal evaluation;
- no physical derivation from CameraInfo, ROI/crop or frame transforms was found.

Therefore `+48 px` is retained only as a named legacy visualization option.
The production Scene01 value was re-searched in two dimensions using
TRAIN_FIT only and frozen before VALIDATION.  The selected `(du,dv)=(64,36)`
passed the independent gate: F1 `0.9538→0.9697`, FN `165→116`, FP `102→59`,
and XY RMSE `0.259→0.245 m` relative to legacy `(48,0)`.

## Residual evidence and its limit

The earlier Scene01 prefix diagnostic found approximately `+58 px` from
person-component to RGB-box centers, but left/right image regions disagreed
(`+57.04` versus `+85.95 px`). Person-component centers and RGB-box centers are
not calibration correspondences; visibility, truncation, pose, occlusion and
wrong component selection can all enter this residual.

Static unlabeled image/LiDAR edges are also only advisory: nearby edges are not
guaranteed to represent the same physical point. Consequently the current
automatic audit may report a horizontal diagnostic candidate, but it cannot
promote that value to principal-point or SE(3) calibration.

A physical update requires independently established static 3D–2D
correspondences and chronological holdout evidence that reduces both median and
P95 errors consistently across image regions, depth ranges and time blocks.

## Enforcement

- `support.project_points(...)` defaults to `du=dv=0`.
- `pipeline.project_physical(...)` remains the raw zero-shift audit baseline.
- `pipeline.project_inference(...)` is the Scene01 inference projection and
  applies the frozen `(64,36)` empirical registration.
- `pipeline.project_display(...)` is used only by rendering.
- `registration.ProjectionResidualAudit` always reports
  `candidate_applied_to_inference=false`.
- `configs/projection_residual_audit.json` enables Scene01 empirical inference
  and keeps legacy display disabled by default.
- The regression suite verifies the zero-shift default, explicit display-only
  shift and absence of audit writeback.

Runtime evidence is written to `projection_residual_audit.json`. Historical
`+48` runs remain comparison artifacts only; they are not the current physical
inference policy.
