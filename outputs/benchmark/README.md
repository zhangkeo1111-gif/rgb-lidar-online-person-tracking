# Benchmark outputs

This directory mirrors the Online v2 repository layout while keeping generated
runtime artifacts out of Git.

- Branch A (`yolo11s-coco`) writes to `outputs/online_v4_coco_person/`.
- Branch B (`yolo26s-coco`) writes to `outputs/online_v4_yolo26s_coco_person/`.
- Cross-detector comparisons write to `outputs/yolo11s_vs_yolo26s_coco/`.

The frozen archive metrics remain documented in the root reports and
`ARCHIVE_MANIFEST.json`; reruns should not overwrite those records silently.
