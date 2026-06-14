#!/usr/bin/env python3
"""Safely sweep one PCA9685 servo axis for hardware bring-up."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import ServoAxisConfig, ServoConfig
from motion.gimbal import Gimbal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep one gimbal servo axis")
    parser.add_argument("--axis", choices=["pan", "tilt"], default="tilt")
    parser.add_argument("--channel", type=int, default=None, help="PCA9685 channel; defaults to axis convention")
    parser.add_argument("--bus", type=int, default=1)
    parser.add_argument("--address", type=lambda x: int(x, 0), default=0x40)
    parser.add_argument("--center", type=float, default=90.0)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--pulse-min", type=int, default=150)
    parser.add_argument("--pulse-max", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    channel = args.channel
    if channel is None:
        channel = 0 if args.axis == "pan" else 1

    axis_config = ServoAxisConfig(
        channel=channel,
        min_angle=args.center - abs(args.delta),
        max_angle=args.center + abs(args.delta),
        center_angle=args.center,
    )
    idle = ServoAxisConfig(channel=15, min_angle=90.0, max_angle=90.0, center_angle=90.0)
    config = ServoConfig(
        bus_number=args.bus,
        address=args.address,
        pan=axis_config if args.axis == "pan" else idle,
        tilt=axis_config if args.axis == "tilt" else idle,
        pulse_min=args.pulse_min,
        pulse_max=args.pulse_max,
    )

    gimbal = Gimbal(config, mock=False)
    try:
        for angle in (args.center, args.center + args.delta, args.center - args.delta, args.center):
            print(f"{args.axis} channel {channel}: {angle:.1f} deg", flush=True)
            gimbal.set_angle(args.axis, angle)
            time.sleep(args.delay)
    finally:
        gimbal.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
