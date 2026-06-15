#!/usr/bin/env bash
# Camera-only MJPEG stream. This verifies capture and USB camera settings without
# detector, gimbal, RKNN, or voice.
#
#   bash scripts/run_camera.sh
#   CAMERA=/dev/video0 WEB_PORT=8081 bash scripts/run_camera.sh
#
# Env overrides: CAMERA, WIDTH, HEIGHT, FPS, WEB_HOST, WEB_PORT, WEB_QUALITY,
# FOURCC, BACKEND, PYTHON
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON=${PYTHON:-/root/miniconda3/envs/b/bin/python3}
CAMERA=${CAMERA:-/dev/video0}
WIDTH=${WIDTH:-640}
HEIGHT=${HEIGHT:-480}
FPS=${FPS:-30}
WEB_HOST=${WEB_HOST:-0.0.0.0}
WEB_PORT=${WEB_PORT:-8080}
WEB_QUALITY=${WEB_QUALITY:-80}
FOURCC=${FOURCC:-MJPG}
BACKEND=${BACKEND:-v4l2}

CMD=("$PYTHON" -m web.stream_webcam
  --camera "$CAMERA"
  --host "$WEB_HOST"
  --port "$WEB_PORT"
  --width "$WIDTH"
  --height "$HEIGHT"
  --fps "$FPS"
  --quality "$WEB_QUALITY"
  --fourcc "$FOURCC"
  --backend "$BACKEND")
CMD+=("$@")

printf 'camera stream: http://%s:%s/\n' "$WEB_HOST" "$WEB_PORT"
printf 'running:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
