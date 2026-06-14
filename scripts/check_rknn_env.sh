#!/usr/bin/env bash
# Collect the board's RKNN / NPU runtime status into a single report.
# Safe and read-only. Usage: bash scripts/check_rknn_env.sh | tee result/logs/rknn_env.txt
set -uo pipefail

PYTHON="${PYTHON:-/root/miniconda3/envs/b/bin/python3}"

echo "===== RKNN / NPU environment report ====="
echo "date (board clock): $(date 2>/dev/null || echo unknown)"
echo "python: ${PYTHON}"
echo

echo "----- Python RKNN runtime modules -----"
"${PYTHON}" - <<'PY'
mods = ["rknn", "rknnlite", "rknn_toolkit_lite2", "rknnlite.api", "rknn.api"]
for name in mods:
    try:
        m = __import__(name)
        print(f"  {name}: OK ({getattr(m, '__file__', 'builtin')})")
    except Exception as exc:
        print(f"  {name}: FAIL ({type(exc).__name__}: {str(exc)[:80]})")
PY
echo

echo "----- onnxruntime (CPU fallback path) -----"
"${PYTHON}" -c "import onnxruntime as o; print('  onnxruntime', o.__version__, o.get_available_providers())" 2>&1 | grep -v -i 'device_discovery\|GPU device' || echo "  onnxruntime: not importable"
echo

echo "----- System RKNN runtime libraries -----"
for f in /usr/lib/librknnrt.so /usr/lib/aarch64-linux-gnu/librknnrt.so /usr/bin/rknn_server; do
    if [ -e "$f" ]; then ls -l "$f"; else echo "  missing: $f"; fi
done
echo

echo "----- DRI render nodes -----"
ls -l /dev/dri/renderD* 2>&1 || echo "  no render nodes"
echo

echo "----- Kernel RKNPU init (dmesg) -----"
( dmesg 2>/dev/null | grep -i rknpu | head -6 ) || echo "  dmesg not accessible (need root / kernel log perms)"
echo

echo "----- RKNPU debug/version nodes -----"
for n in /sys/kernel/debug/rknpu/version /proc/device-tree/npu*/status; do
    [ -e "$n" ] && { echo "  $n:"; cat "$n" 2>/dev/null | tr -d '\0'; echo; }
done
echo

echo "===== end of report ====="
