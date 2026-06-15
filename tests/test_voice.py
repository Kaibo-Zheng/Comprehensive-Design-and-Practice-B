"""目标获取语音事件门控测试。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import VoiceConfig
from inference.detector import Target
from voice import SpeechEventGate, VoiceNotifier, format_voice_text


def _target(stale: bool = False) -> Target:
    return Target(center=(10, 10), bbox=(0, 0, 20, 20), score=1.0, stale=stale, label="person", confidence=0.91)


def test_gate_emits_only_on_target_acquired_edge_after_cooldown():
    now = 0.0

    def time_fn() -> float:
        return now

    gate = SpeechEventGate(cooldown_s=5.0, time_fn=time_fn)

    assert gate.update(None) is False
    assert gate.update(_target()) is True
    assert gate.update(_target()) is False

    assert gate.update(None) is False
    now = 3.0
    assert gate.update(_target()) is False

    assert gate.update(None) is False
    now = 5.0
    assert gate.update(_target()) is True


def test_gate_ignores_stale_targets():
    gate = SpeechEventGate(cooldown_s=0.0)
    assert gate.update(_target(stale=True)) is False
    assert gate.update(_target()) is True


def test_voice_text_template_uses_target_fields():
    text = format_voice_text("Detected {label} at {confidence} by {detector}.", _target(), "yolo_onnx")
    assert text == "Detected person at 0.91 by yolo_onnx."


def test_play_mode_requires_prepared_audio():
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = VoiceConfig(enabled=True, mode="play", audio_path=str(Path(temp_dir) / "missing.wav"))
        try:
            VoiceNotifier(cfg, "test")
        except RuntimeError as exc:
            assert "Voice audio file not found" in str(exc)
        else:
            raise AssertionError("missing audio path should fail at startup")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {exc!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
