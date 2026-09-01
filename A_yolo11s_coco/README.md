# Online v4A — Official COCO YOLO11s PERSON

This is the frozen YOLO11s COCO baseline branch of archived Online v4.

- detector ID: `yolo11s-coco`
- checkpoint: `..\assets\models\detector\yolo11s_coco.pt`
- shared core: `..\src\online_v4`
- outputs: `..\outputs\online_v4_coco_person`
- VALIDATION Precision / Recall / F1: `0.905401 / 0.970205 / 0.936684`

Run from this directory with `run.ps1`. Extra arguments are forwarded to the
shared `run_online.py` entry point.
