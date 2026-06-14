#!/usr/bin/env bash
# Run the project benchmarks.
#
# By default only the lightweight 5x5 convolution benchmark runs (safe on the
# 4GB board): pure Python, pymp row-parallel Python, and the C extension.
# Set WITH_YOLO=1 to also time a short YOLO ONNX inference run, but only if a
# model exists and >=1.0GB memory is available (4GB memory guard).
#
#   bash scripts/benchmark_all.sh
#   WITH_YOLO=1 bash scripts/benchmark_all.sh
#
# Env overrides: HEIGHT, WIDTH, ITERS, PYMP_THREADS, PYTHON, WITH_YOLO, MODEL, CAMERA, MAX_FRAMES
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON=${PYTHON:-/root/miniconda3/envs/b/bin/python3}
HEIGHT=${HEIGHT:-120}
WIDTH=${WIDTH:-160}
ITERS=${ITERS:-3}
PYMP_THREADS=${PYMP_THREADS:-4}
OUT=result/logs/cpp_conv_benchmark.txt
mkdir -p result/logs

echo "=== 5x5 convolution benchmark (${WIDTH}x${HEIGHT}, ${ITERS} iters, pymp=${PYMP_THREADS}) ==="
CMD=("$PYTHON" acceleration/benchmark.py --height "$HEIGHT" --width "$WIDTH" --iterations "$ITERS" --pymp-threads "$PYMP_THREADS")
printf 'running:'; printf ' %q' "${CMD[@]}"; printf '\n'
"${CMD[@]}" | tee "$OUT"
echo "saved: $OUT"

if [[ "${WITH_YOLO:-0}" == "1" ]]; then
  MODEL=${MODEL:-model/yolo11n.onnx}
  CAMERA=${CAMERA:-/dev/video0}
  YOUT=result/logs/yolo_onnx_benchmark.csv
  echo
  echo "=== optional YOLO ONNX CPU benchmark (WITH_YOLO=1) ==="
  if [[ ! -e "$MODEL" ]]; then
    echo "skip: model not found: $MODEL" >&2
  else
    avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    echo "MemAvailable: $((avail_kb / 1024)) MB"
    if (( avail_kb < 1048576 )); then
      echo "skip: <1.0GB available; not starting ONNX inference (4GB memory guard)" >&2
    else
      YCMD=("$PYTHON" -m app --camera "$CAMERA" --detector yolo_onnx \
        --model "$MODEL" --target-class person --mock-servo --no-display \
        --max-frames "${MAX_FRAMES:-10}" --log "$YOUT")
      printf 'running:'; printf ' %q' "${YCMD[@]}"; printf '\n'
      "${YCMD[@]}"
      echo "saved: $YOUT"
    fi
  fi
fi
