"""Preview person detector output without moving the gimbal."""

from __future__ import annotations

import argparse
import sys

import cv2

from camera import open_camera
from core.config import CameraConfig, DetectorConfig, parse_camera_source

from .detector import PersonDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview OpenCV person detection")
    parser.add_argument("--camera", type=parse_camera_source, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--detector", choices=["face", "upperbody", "hog"], default="face")
    parser.add_argument("--cascade-path", default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = CameraConfig(source=args.camera, width=args.width, height=args.height)
    try:
        cap = open_camera(config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    detector = PersonDetector(
        DetectorConfig(
            detector=args.detector,
            cascade_path=args.cascade_path,
        )
    )

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("error: failed to read frame", file=sys.stderr)
                return 1

            target = detector.detect(frame)
            if target is not None:
                x, y, w, h = target.bbox
                color = (0, 255, 255) if target.stale else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.circle(frame, target.center, 4, color, -1)
                print(f"target center={target.center} bbox={target.bbox} stale={target.stale}")

            if not args.no_display:
                cv2.imshow("detector preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
