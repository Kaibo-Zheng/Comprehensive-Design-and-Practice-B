#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON=${PYTHON:-/root/miniconda3/envs/b/bin/python3}
CAMERA=${CAMERA:-/dev/video0}
MODEL=${MODEL:-model/yolo11n.rknn}
WEB_HOST=${WEB_HOST:-0.0.0.0}
WEB_PORT=${WEB_PORT:-8080}
WEB_QUALITY=${WEB_QUALITY:-45}
WEB_MAX_FPS=${WEB_MAX_FPS:-4}
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
VOICE_AUDIO_PATH=${VOICE_AUDIO_PATH:-audio/found_person_zh_beep_loud_44k.wav}
VOICE_PLAYER=${VOICE_PLAYER:-scripts/play_voice_usb.sh}
VOICE_COOLDOWN=${VOICE_COOLDOWN:-5}
LOG=${LOG:-result/logs/rknn_web_voice_fast.csv}

echo performance > /sys/class/devfreq/dmc/governor
echo performance > /sys/class/devfreq/fdab0000.npu/governor
for p in /sys/devices/system/cpu/cpufreq/policy*; do
  echo performance > "$p/scaling_governor"
done

CMD=(
  "$PYTHON" -m app
  --camera "$CAMERA"
  --width 640
  --height 480
  --fps 30
  --detector yolo_rknn
  --model "$MODEL"
  --target-class person
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
  --web-host "$WEB_HOST"
  --web-port "$WEB_PORT"
  --web-quality "$WEB_QUALITY"
  --web-max-fps "$WEB_MAX_FPS"
  --no-display
  --voice-enabled
  --voice-mode play
  --voice-audio-path "$VOICE_AUDIO_PATH"
  --voice-player "$VOICE_PLAYER"
  --voice-cooldown "$VOICE_COOLDOWN"
  --log "$LOG"
)

printf 'running:'; printf ' %q' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
