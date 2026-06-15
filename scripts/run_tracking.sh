#!/usr/bin/env bash
# RKNN + camera + gimbal tracking with the same tuning defaults as scripts/run.sh,
# but without the web monitor and without voice playback.
#
#   bash scripts/run_tracking.sh           # real servo, board hardware
#   bash scripts/run_tracking.sh --mock    # finite headless mock smoke
#
# Env overrides: CAMERA, MODEL, TARGET_CLASS, WIDTH, HEIGHT, FPS, PAN_CHANNEL,
# TILT_CHANNEL, PAN_RANGE, TILT_RANGE, PAN_CENTER, TILT_CENTER, PAN_SIGN,
# TILT_SIGN, PAN_GAIN, TILT_GAIN, DEAD_ZONE, MAX_STEP, STARTUP_CENTER, LOG,
# MAX_FRAMES, PYTHON, MOCK=1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON=${PYTHON:-/root/miniconda3/envs/b/bin/python3}
CAMERA=${CAMERA:-/dev/video0}
MODEL=${MODEL:-model/yolo11n.rknn}
TARGET_CLASS=${TARGET_CLASS:-person}
WIDTH=${WIDTH:-640}
HEIGHT=${HEIGHT:-480}
FPS=${FPS:-30}
PAN_CHANNEL=${PAN_CHANNEL:-0}
TILT_CHANNEL=${TILT_CHANNEL:-1}
PAN_RANGE=${PAN_RANGE:-45,135}
TILT_RANGE=${TILT_RANGE:-0,90}
PAN_CENTER=${PAN_CENTER:-90}
TILT_CENTER=${TILT_CENTER:-45}
PAN_SIGN=${PAN_SIGN:--1}
TILT_SIGN=${TILT_SIGN:-1}
PAN_GAIN=${PAN_GAIN:-0.015}
TILT_GAIN=${TILT_GAIN:-0.015}
DEAD_ZONE=${DEAD_ZONE:-25}
MAX_STEP=${MAX_STEP:-0.8}
STARTUP_CENTER=${STARTUP_CENTER:-pan}
LOG=${LOG:-result/logs/rknn_tracking_demo.csv}

MOCK_MODE=0
if [[ "${1:-}" == "--mock" ]]; then MOCK_MODE=1; shift; fi
if [[ "${MOCK:-0}" == "1" ]]; then MOCK_MODE=1; fi

CMD=(
  "$PYTHON" -m tracker.app
  --camera "$CAMERA"
  --width "$WIDTH"
  --height "$HEIGHT"
  --fps "$FPS"
  --detector yolo_rknn
  --model "$MODEL"
  --target-class "$TARGET_CLASS"
  --pan-channel "$PAN_CHANNEL"
  --tilt-channel "$TILT_CHANNEL"
  --pan-range "$PAN_RANGE"
  --tilt-range "$TILT_RANGE"
  --pan-center "$PAN_CENTER"
  --tilt-center "$TILT_CENTER"
  --pan-sign "$PAN_SIGN"
  --tilt-sign "$TILT_SIGN"
  --pan-gain "$PAN_GAIN"
  --tilt-gain "$TILT_GAIN"
  --dead-zone "$DEAD_ZONE"
  --max-step "$MAX_STEP"
  --startup-center "$STARTUP_CENTER"
  --no-display
  --log "$LOG"
)
if [[ -n "${CONF:-}" ]]; then CMD+=(--conf-thres "$CONF"); fi
if [[ -n "${IOU:-}" ]]; then CMD+=(--iou-thres "$IOU"); fi
if [[ -n "${INPUT_SIZE:-}" ]]; then CMD+=(--input-size "$INPUT_SIZE"); fi
if [[ "$MOCK_MODE" == "1" ]]; then
  CMD+=(--mock-servo --max-frames "${MAX_FRAMES:-30}")
fi
CMD+=("$@")

printf 'running:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
