#!/usr/bin/env bash
# YOLO RKNN (RK3588 NPU) person-tracking demo.
#
#   bash scripts/run_rknn.sh --mock
#   MODEL=model/yolo11n.rknn bash scripts/run_rknn.sh --mock
#
# Requires rknn_toolkit_lite2 + a converted .rknn model on the board. If either
# is missing this exits with a clear error and does NOT fall back to CPU.
# Env overrides: CAMERA, MODEL, TARGET_CLASS, CONF, IOU, INPUT_SIZE, LOG, MAX_FRAMES, PYTHON, MOCK=1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON=${PYTHON:-/root/miniconda3/envs/b/bin/python3}
CAMERA=${CAMERA:-/dev/video0}
MODEL=${MODEL:-model/yolo11n.rknn}
TARGET_CLASS=${TARGET_CLASS:-person}
LOG=${LOG:-result/logs/rknn_demo.csv}

MOCK_MODE=0
if [[ "${1:-}" == "--mock" ]]; then MOCK_MODE=1; shift; fi
if [[ "${MOCK:-0}" == "1" ]]; then MOCK_MODE=1; fi

CMD=("$PYTHON" -m tracker.app --camera "$CAMERA" --detector yolo_rknn \
  --model "$MODEL" --target-class "$TARGET_CLASS" --log "$LOG")
if [[ -n "${CONF:-}" ]]; then CMD+=(--conf-thres "$CONF"); fi
if [[ -n "${IOU:-}" ]]; then CMD+=(--iou-thres "$IOU"); fi
if [[ -n "${INPUT_SIZE:-}" ]]; then CMD+=(--input-size "$INPUT_SIZE"); fi
if [[ "$MOCK_MODE" == "1" ]]; then
  CMD+=(--mock-servo --no-display --max-frames "${MAX_FRAMES:-30}")
fi
CMD+=("$@")

printf 'running:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
