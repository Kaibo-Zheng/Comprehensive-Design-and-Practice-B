#!/usr/bin/env bash
# Tracking + voice playback notification. This isolates the voice path from the
# all-in-one scripts/run.sh run.
#
#   bash scripts/run_voice.sh
#   bash scripts/run_voice.sh --mock
#
# Env overrides: CAMERA, WIDTH, HEIGHT, FPS, DETECTOR, MODEL, TARGET_CLASS,
# WEB_HOST, WEB_PORT, WEB_QUALITY, WEB_MAX_FPS, LOG, VOICE_TEXT,
# VOICE_COOLDOWN, VOICE_THREADS, VOICE_MAX_NEW_FRAMES, VOICE_AUDIO_PATH,
# VOICE_PLAYER, MAX_FRAMES, PYTHON, MOCK=1.
# Set WEB_HOST=0.0.0.0 if you also want the web monitor while testing voice.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-b}"
fi

PYTHON=${PYTHON:-python}

MOCK_FLAG=()
if [[ "${1:-}" == "--mock" || "${MOCK:-0}" == "1" ]]; then
  MOCK_FLAG=(--mock-servo)
  if [[ "${1:-}" == "--mock" ]]; then shift; fi
fi

CAMERA="${CAMERA:-/dev/video0}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"
DETECTOR="${DETECTOR:-face}"
MODEL="${MODEL:-}"
TARGET_CLASS="${TARGET_CLASS:-person}"
WEB_HOST="${WEB_HOST:-}"
WEB_PORT="${WEB_PORT:-8080}"
WEB_QUALITY="${WEB_QUALITY:-45}"
WEB_MAX_FPS="${WEB_MAX_FPS:-4}"
LOG="${LOG:-result/logs/voice_demo.csv}"
VOICE_TEXT="${VOICE_TEXT:-检测到{label_zh}。}"
VOICE_COOLDOWN="${VOICE_COOLDOWN:-5}"
VOICE_THREADS="${VOICE_THREADS:-2}"
VOICE_MAX_NEW_FRAMES="${VOICE_MAX_NEW_FRAMES:-24}"
VOICE_AUDIO_PATH="${VOICE_AUDIO_PATH:-audio/found_person_zh_beep_loud_44k.wav}"
VOICE_PLAYER="${VOICE_PLAYER:-scripts/run_player.sh}"

if [[ ! -f "$VOICE_AUDIO_PATH" ]]; then
  echo "Missing prepared voice WAV: $VOICE_AUDIO_PATH" >&2
  echo "Set VOICE_AUDIO_PATH to an existing WAV file." >&2
  exit 2
fi

CMD=("$PYTHON" -m tracker.app
  --detector "$DETECTOR"
  --camera "$CAMERA"
  --width "$WIDTH"
  --height "$HEIGHT"
  --fps "$FPS"
  --no-display
  --voice-enabled
  --voice-mode play
  --voice-audio-path "$VOICE_AUDIO_PATH"
  --voice-player "$VOICE_PLAYER"
  --voice-text "$VOICE_TEXT"
  --voice-cooldown "$VOICE_COOLDOWN"
  --voice-threads "$VOICE_THREADS"
  --voice-max-new-frames "$VOICE_MAX_NEW_FRAMES"
  --log "$LOG"
  "${MOCK_FLAG[@]}")
if [[ -n "$WEB_HOST" ]]; then
  CMD+=(--web-host "$WEB_HOST" --web-port "$WEB_PORT" \
    --web-quality "$WEB_QUALITY" --web-max-fps "$WEB_MAX_FPS")
fi
if [[ -n "$MODEL" ]]; then CMD+=(--model "$MODEL"); fi
if [[ "$DETECTOR" == yolo_* ]]; then CMD+=(--target-class "$TARGET_CLASS"); fi
if (( ${#MOCK_FLAG[@]} > 0 )); then
  CMD+=(--max-frames "${MAX_FRAMES:-30}")
fi
CMD+=("$@")

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
