#!/usr/bin/env bash
# Put the RK3588 board into the same performance mode used by scripts/run.sh.
#
#   sudo bash scripts/run_performance.sh
#   DRY_RUN=1 bash scripts/run_performance.sh
#
# Env overrides: DRY_RUN=1, STRICT=1
set -euo pipefail

STRICT=${STRICT:-0}

write_governor() {
  local path="$1"
  local value="${2:-performance}"

  if [[ ! -e "$path" ]]; then
    echo "skip missing: $path" >&2
    return 0
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "would write: $value > $path"
    return 0
  fi

  if echo "$value" > "$path" 2>/dev/null; then
    echo "set: $path = $value"
  else
    echo "failed: $path (try running with sudo/root)" >&2
    if [[ "$STRICT" == "1" ]]; then
      return 1
    fi
  fi
}

write_governor /sys/class/devfreq/dmc/governor
write_governor /sys/class/devfreq/fdab0000.npu/governor

shopt -s nullglob
cpu_governors=(/sys/devices/system/cpu/cpufreq/policy*/scaling_governor)
if (( ${#cpu_governors[@]} == 0 )); then
  echo "skip missing: /sys/devices/system/cpu/cpufreq/policy*/scaling_governor" >&2
fi

for governor in "${cpu_governors[@]}"; do
  write_governor "$governor"
done
