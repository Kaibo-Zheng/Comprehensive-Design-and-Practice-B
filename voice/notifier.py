"""Voice notification support for target acquisition events."""

from __future__ import annotations

import hashlib
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.config import VoiceConfig
from inference.detector import Target


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VoiceEvent:
    text: str
    label: str
    confidence: float | None


LABEL_ZH = {
    "person": "人员",
    "face": "人员",
    "upperbody": "人员",
    "hog": "人员",
}


class SpeechEventGate:
    """Detect target-acquired edges and apply a cooldown."""

    def __init__(
        self,
        cooldown_s: float,
        *,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cooldown_s = max(0.0, float(cooldown_s))
        self.time_fn = time_fn
        self._was_active = False
        self._last_event_at = -1.0e12

    def update(self, target: Target | None) -> bool:
        active = target is not None and not target.stale
        now = self.time_fn()
        should_emit = active and not self._was_active and (now - self._last_event_at) >= self.cooldown_s
        self._was_active = active
        if should_emit:
            self._last_event_at = now
        return should_emit


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def format_voice_text(template: str, target: Target, detector: str) -> str:
    label = target.label or "target"
    label_zh = LABEL_ZH.get(label, label)
    confidence = "" if target.confidence is None else f"{target.confidence:.2f}"
    try:
        return template.format(label=label, label_zh=label_zh, label_cn=label_zh, confidence=confidence, detector=detector)
    except (KeyError, IndexError, ValueError):
        return template


class VoiceNotifier:
    """Background MOSS TTS synthesizer/player for target events."""

    def __init__(self, config: VoiceConfig, detector_name: str) -> None:
        self.config = config
        self.detector_name = detector_name
        self.gate = SpeechEventGate(config.cooldown_s)
        self.output_dir = resolve_repo_path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_path = self._resolve_audio_path()
        self._queue: "queue.Queue[VoiceEvent | None]" = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._runtime = None
        self._disabled = False
        if self.config.mode not in {"play", "synthesize"}:
            raise RuntimeError(f"Unsupported voice mode: {self.config.mode}")
        self._thread = threading.Thread(target=self._worker, name="voice-notifier", daemon=True)
        self._thread.start()

    def update(self, target: Target | None) -> None:
        if self._disabled or not self.gate.update(target):
            return
        assert target is not None
        event = VoiceEvent(
            text=format_voice_text(self.config.text, target, self.detector_name),
            label=target.label,
            confidence=target.confidence,
        )
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            print("voice: dropping notification because speech queue is full", file=sys.stderr, flush=True)

    def close(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if event is None:
                return
            try:
                wav_path = self._synthesize(event.text) if self.config.mode == "synthesize" else self._existing_audio(event.text)
                if self.config.playback:
                    self._play(wav_path)
            except Exception as exc:  # noqa: BLE001 - voice must not kill tracking
                self._disabled = True
                print(f"voice: disabled after error: {exc}", file=sys.stderr, flush=True)

    def _resolve_audio_path(self) -> Path | None:
        if not self.config.audio_path:
            return None
        path = resolve_repo_path(self.config.audio_path)
        if not path.is_file():
            raise RuntimeError(f"Voice audio file not found: {path}")
        return path

    def _existing_audio(self, text: str) -> Path:
        if self.audio_path is not None:
            return self.audio_path
        path = self._cached_audio_path(text)
        if not path.is_file() or path.stat().st_size <= 44:
            raise RuntimeError(
                f"Voice audio file not prepared: {path}. "
                "Provide an existing WAV with --voice-audio-path, or use --voice-mode synthesize "
                "with an external runtime installation."
            )
        return path

    def _ensure_runtime(self):
        if self._runtime is not None:
            return self._runtime

        if not self.config.runtime_dir or not self.config.model_dir:
            raise RuntimeError(
                "Voice synthesize mode requires external --voice-runtime-dir and --voice-model-dir paths."
            )

        runtime_dir = resolve_repo_path(self.config.runtime_dir)
        model_dir = resolve_repo_path(self.config.model_dir)
        if not runtime_dir.is_dir():
            raise RuntimeError(f"MOSS TTS runtime directory not found: {runtime_dir}")
        if not model_dir.is_dir():
            raise RuntimeError(f"MOSS TTS model directory not found: {model_dir}")

        runtime_path = str(runtime_dir)
        if runtime_path not in sys.path:
            sys.path.insert(0, runtime_path)

        from onnx_tts_runtime import OnnxTtsRuntime  # type: ignore

        self._runtime = OnnxTtsRuntime(
            model_dir=model_dir,
            thread_count=self.config.threads,
            max_new_frames=self.config.max_new_frames,
            do_sample=self.config.sample_mode != "greedy",
            sample_mode=self.config.sample_mode,
            output_dir=self.output_dir,
        )
        return self._runtime

    def _synthesize(self, text: str) -> Path:
        wav_path = self._cached_audio_path(text)
        if wav_path.is_file() and wav_path.stat().st_size > 44:
            return wav_path

        runtime = self._ensure_runtime()
        runtime.synthesize(
            text=text,
            voice=self.config.voice_preset,
            output_audio_path=wav_path,
            sample_mode=self.config.sample_mode,
            do_sample=self.config.sample_mode != "greedy",
            streaming=True,
            max_new_frames=self.config.max_new_frames,
            enable_wetext=self.config.enable_wetext,
            enable_normalize_tts_text=self.config.enable_wetext,
            seed=self.config.seed,
        )
        return wav_path

    def _cached_audio_path(self, text: str) -> Path:
        cache_key = hashlib.sha1(
            "|".join(
                [
                    text,
                    self.config.voice_preset,
                    self.config.sample_mode,
                    str(self.config.max_new_frames),
                ]
            ).encode("utf-8")
        ).hexdigest()[:16]
        return self.output_dir / f"voice_{cache_key}.wav"

    def _play(self, wav_path: Path) -> None:
        player_parts = shlex.split(self.config.player)
        if not player_parts:
            return
        executable = shutil.which(player_parts[0])
        if executable is None:
            raise RuntimeError(f"audio player not found on PATH: {player_parts[0]}")
        subprocess.run([executable, *player_parts[1:], str(wav_path)], check=True)
