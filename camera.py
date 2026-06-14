"""Camera opening helpers for the gimbal tracker.

This module remains the canonical import path for camera access so older
callers and top-level docs stay valid while the package also exposes the
``capture`` subpackage for clearer ownership.
"""

from __future__ import annotations

import cv2

from core.config import CameraConfig


def open_camera(config: CameraConfig) -> cv2.VideoCapture:
    """Open an OpenCV capture from a :class:`CameraConfig`.

    String sources (device paths like ``/dev/video0`` or
    ``/dev/v4l/by-id/...``) use the V4L2 backend, which is the most reliable
    choice for USB webcams on Linux. Integer indexes use OpenCV's default
    backend. Requested FOURCC / resolution / FPS are applied before the open
    check so failures surface immediately with a clear error.
    """

    if isinstance(config.source, str):
        cap = cv2.VideoCapture(config.source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(config.source)

    if config.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.fourcc[:4]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cap.set(cv2.CAP_PROP_FPS, config.fps)

    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open camera source {config.source!r}. "
            "Check `ls -l /dev/video*`, `fuser -v /dev/video0` for a stale "
            "process, or pass a stable --camera /dev/v4l/by-id/... path."
        )
    return cap


__all__ = ["open_camera"]
