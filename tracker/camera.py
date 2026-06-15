"""云台跟踪系统的摄像头打开辅助函数。

该模块作为摄像头访问的统一入口，便于主程序和其他工具复用。
"""

from __future__ import annotations

import cv2

from common.config import CameraConfig


def open_camera(config: CameraConfig) -> cv2.VideoCapture:
    """根据 CameraConfig 打开 OpenCV 摄像头。

    字符串输入会被当作设备路径并使用 V4L2 后端，整数输入则使用 OpenCV 默认后端。
    FOURCC、分辨率和帧率会在打开检查前设置，便于尽早暴露设备问题。
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
