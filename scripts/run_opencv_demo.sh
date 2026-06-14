#!/usr/bin/env bash
# Baseline OpenCV face/upperbody tracking demo (the fallback "保底" demo).
#
#   bash scripts/run_opencv_demo.sh           # real servo (run on the board)
#   bash scripts/run_opencv_demo.sh --mock    # safe headless mock-servo smoke
#
# Env overrides: CAMERA, DETECTOR, LOG, MAX_FRAMES, PYTHON, MOCK=1
# Extra args after the script are passed straight through to app.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON=${PYTHON:-/root/miniconda3/envs/b/bin/python3}
CAMERA=${CAMERA:-/dev/video0}
DETECTOR=${DETECTOR:-face}
LOG=${LOG:-result/logs/opencv_demo.csv}

MOCK_MODE=0
if [[ "${1:-}" == "--mock" ]]; then MOCK_MODE=1; shift; fi
if [[ "${MOCK:-0}" == "1" ]]; then MOCK_MODE=1; fi

CMD=("$PYTHON" -m app --camera "$CAMERA" --detector "$DETECTOR" --log "$LOG")
if [[ "$MOCK_MODE" == "1" ]]; then
  CMD+=(--mock-servo --no-display --max-frames "${MAX_FRAMES:-30}")
fi
CMD+=("$@")

printf 'running:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
