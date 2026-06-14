#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PYTHON:-/root/miniconda3/envs/b/bin/python3}

exec "$PYTHON" tool/benchmark_nms.py "$@"
