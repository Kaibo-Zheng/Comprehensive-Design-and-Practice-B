# model/

YOLO person-detection models for the tracker. Binaries are **not committed**;
this file records naming, source, and the export/conversion pipeline so any of
them can be regenerated.

## Naming convention

| File | Format | Runtime | Used by |
| --- | --- | --- | --- |
| `yolo11n.onnx` | ONNX | onnxruntime (CPU) | `--detector yolo_onnx` |
| `yolo11n.rknn` | RKNN | rknnlite2 (RK3588 NPU) | `--detector yolo_rknn` |

Default expected input size is 640×640, single image (batch=1), COCO classes
(class id 0 = `person`).

## Source and export pipeline

```text
YOLO .pt  (Ultralytics, e.g. yolov8n.pt / yolov5nu.pt)
  -> ONNX   : exported with opset 12, fixed 640x640, batch=1
  -> RKNN   : converted on an x86 host (or board) with rknn-toolkit2, target rk3588
  -> model/yolo11n.rknn
```

### 1. Export ONNX (example, Ultralytics)

```bash
# from an env that has ultralytics + torch
yolo export model=yolov8n.pt format=onnx opset=12 imgsz=640 simplify=True
mv yolov8n.onnx model/yolo11n.onnx
```

Use an environment that has Ultralytics and torch installed to produce the
`.pt` and ONNX export. Keep large exported files out of git.

### 2. Convert ONNX -> RKNN

Use the helper script (see `scripts/convert_yolo_rknn.sh`):

```bash
bash scripts/convert_yolo_rknn.sh \
  --onnx model/yolo11n.onnx \
  --output model/yolo11n.rknn \
  --target rk3588
```

`rknn-toolkit2` (the full conversion toolkit) is normally run on an x86 host.
The board only needs `rknn_toolkit_lite2` (runtime) to *run* the `.rknn`.
See `scripts/check_rknn_env.sh` and `result/logs/rknn_env.txt` for the board's
current RKNN/NPU status.

## Notes

- If you only have the runtime (`rknnlite2`) on the board, convert the model on
  another machine and copy the `.rknn` here.
- Output decoding (YOLOv5 vs YOLOv8 layout) is handled in
  `inference/postprocess.py`, shared by both ONNX and RKNN detectors.

## Board RKNN status (measured 2026-06-13)

| Component | Status |
| --- | --- |
| NPU hardware (`/proc/device-tree/npu@fdab0000/status`) | `okay` |
| RKNPU kernel driver (`/sys/kernel/debug/rknpu/version`) | `v0.9.6` |
| System runtime `/usr/lib/librknnrt.so` | present (2023-05-23) |
| `/usr/bin/rknn_server` | present |
| `/dev/dri/renderD128`, `renderD129` | present |
| Python `rknn_toolkit_lite2` / `rknnlite` (env `b`) | installed (`rknnlite` import OK) |
| `onnxruntime` (CPU) | 1.23.2 — `CPUExecutionProvider` available |

**Current working state:** the board-side lite runtime is available and the
converted model is available at `model/yolo11n.rknn`. The RKNN detector
runs on the NPU via:

```bash
/root/miniconda3/envs/b/bin/python3 -m app \
  --camera /dev/video0 \
  --detector yolo_rknn \
  --model model/yolo11n.rknn \
  --target-class person \
  --mock-servo --no-display --max-frames 20 \
  --log result/logs/p7_yolo_rknn.csv
```

Observed on-board result (2026-06-13): runtime/library/toolkit version `2.3.2`,
driver `0.9.6`, `avg_inference_ms ≈ 71.4`, `average_fps ≈ 7.0` on
`result/logs/p7_yolo_rknn.csv` (20-frame mock run, 20/20 frames detected
`person`).
