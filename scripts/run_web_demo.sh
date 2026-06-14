#!/usr/bin/env bash
# Tracking + integrated web monitor. Open http://<board-ip>:<WEB_PORT>/ in a browser.
#
#   bash scripts/run_web_demo.sh           # live; runs until Ctrl-C
#   bash scripts/run_web_demo.sh --mock    # finite headless mock smoke (frees the port)
#
# Env overrides: CAMERA, DETECTOR, MODEL, TARGET_CLASS, WEB_HOST, WEB_PORT,
# WEB_QUALITY, WEB_MAX_FPS, LOG, MAX_FRAMES, PYTHON, MOCK=1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON=${PYTHON:-/root/miniconda3/envs/b/bin/python3}
CAMERA=${CAMERA:-/dev/video0}
DETECTOR=${DETECTOR:-face}
MODEL=${MODEL:-}
TARGET_CLASS=${TARGET_CLASS:-person}
WEB_HOST=${WEB_HOST:-0.0.0.0}
WEB_PORT=${WEB_PORT:-8080}
LOG=${LOG:-result/logs/web_demo.csv}

MOCK_MODE=0
if [[ "${1:-}" == "--mock" ]]; then MOCK_MODE=1; shift; fi
if [[ "${MOCK:-0}" == "1" ]]; then MOCK_MODE=1; fi

CMD=("$PYTHON" -m app --camera "$CAMERA" --detector "$DETECTOR" \
  --web-host "$WEB_HOST" --web-port "$WEB_PORT" --no-display --log "$LOG")
if [[ -n "$MODEL" ]]; then CMD+=(--model "$MODEL"); fi
if [[ "$DETECTOR" == yolo_* ]]; then CMD+=(--target-class "$TARGET_CLASS"); fi
if [[ -n "${WEB_QUALITY:-}" ]]; then CMD+=(--web-quality "$WEB_QUALITY"); fi
if [[ -n "${WEB_MAX_FPS:-}" ]]; then CMD+=(--web-max-fps "$WEB_MAX_FPS"); fi
if [[ "$MOCK_MODE" == "1" ]]; then
  CMD+=(--mock-servo --max-frames "${MAX_FRAMES:-30}")
fi
CMD+=("$@")

printf 'web monitor: http://%s:%s/\n' "$WEB_HOST" "$WEB_PORT"
printf 'running:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
