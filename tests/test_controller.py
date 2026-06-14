"""Tests for per-axis tracking controller movement flags."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import ControllerConfig, ServoConfig
from motion import TrackingController


def test_controller_reports_pan_only_movement():
    controller = TrackingController(
        ControllerConfig(dead_zone=25, pan_gain=0.1, tilt_gain=0.1),
        ServoConfig(),
    )

    result = controller.update(target_center=(360, 240), frame_size=(640, 480))

    assert result.moved is True
    assert result.pan_moved is True
    assert result.tilt_moved is False


def test_controller_reports_tilt_only_movement():
    controller = TrackingController(
        ControllerConfig(dead_zone=25, pan_gain=0.1, tilt_gain=0.1),
        ServoConfig(),
    )

    result = controller.update(target_center=(320, 280), frame_size=(640, 480))

    assert result.moved is True
    assert result.pan_moved is False
    assert result.tilt_moved is True


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
