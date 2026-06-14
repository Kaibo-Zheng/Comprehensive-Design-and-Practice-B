#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-b}"
fi

MOCK_FLAG=()
if [[ "${1:-}" == "--mock" || "${MOCK:-0}" == "1" ]]; then
  MOCK_FLAG=(--mock-servo)
fi

CAMERA="${CAMERA:-0}"
DETECTOR="${DETECTOR:-face}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-8080}"
LOG="${LOG:-result/logs/voice_demo.csv}"
VOICE_TEXT="${VOICE_TEXT:-检测到{label_zh}。}"
VOICE_COOLDOWN="${VOICE_COOLDOWN:-5}"
VOICE_THREADS="${VOICE_THREADS:-2}"
VOICE_MAX_NEW_FRAMES="${VOICE_MAX_NEW_FRAMES:-24}"
VOICE_AUDIO_PATH="${VOICE_AUDIO_PATH:-audio/found_person_zh_beep_loud_44k.wav}"

if [[ ! -f "$VOICE_AUDIO_PATH" ]]; then
  echo "Missing prepared voice WAV: $VOICE_AUDIO_PATH" >&2
  echo "Set VOICE_AUDIO_PATH to an existing WAV file." >&2
  exit 2
fi

CMD=(python -m app
  --detector "$DETECTOR"
  --camera "$CAMERA"
  --web-host "$WEB_HOST"
  --web-port "$WEB_PORT"
  --no-display
  --voice-enabled
  --voice-mode play
  --voice-audio-path "$VOICE_AUDIO_PATH"
  --voice-text "$VOICE_TEXT"
  --voice-cooldown "$VOICE_COOLDOWN"
  --voice-threads "$VOICE_THREADS"
  --voice-max-new-frames "$VOICE_MAX_NEW_FRAMES"
  --log "$LOG"
  "${MOCK_FLAG[@]}")

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
