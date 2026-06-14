"""Tests for application-level configuration assembly."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import build_configs, parse_args


def test_shared_center_still_applies_to_both_axes():
    args = parse_args(["--center", "88", "--pan-range", "40,140", "--tilt-range", "60,120"])

    _camera, _detector, servo, _controller, _voice = build_configs(args)

    assert servo.pan.center_angle == 88
    assert servo.tilt.center_angle == 88


def test_per_axis_centers_override_shared_center():
    args = parse_args(
        [
            "--center",
            "90",
            "--pan-center",
            "91",
            "--tilt-center",
            "85",
            "--pan-range",
            "45,135",
            "--tilt-range",
            "80,105",
        ]
    )

    _camera, _detector, servo, _controller, _voice = build_configs(args)

    assert servo.pan.center_angle == 91
    assert servo.tilt.center_angle == 85


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
