#!/usr/bin/env bash
set -euo pipefail

DEVICE="${VOICE_DEVICE:-plughw:3,0}"
LOCK="${VOICE_LOCK:-/tmp/gimbal_tracker_voice.lock}"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <wav>" >&2
  exit 2
fi

exec flock -n "$LOCK" aplay -D "$DEVICE" "$1"
