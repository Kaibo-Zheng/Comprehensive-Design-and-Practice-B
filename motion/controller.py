"""将图像误差转换为云台角度的跟踪控制逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from common.config import ControllerConfig, ServoConfig, clamp


@dataclass
class ControlResult:
    pan: float
    tilt: float
    x_error: int
    y_error: int
    moved: bool
    pan_moved: bool = False
    tilt_moved: bool = False


class TrackingController:
    def __init__(self, controller_config: ControllerConfig, servo_config: ServoConfig):
        self.controller_config = controller_config
        self.servo_config = servo_config
        self.pan = servo_config.pan.center_angle
        self.tilt = servo_config.tilt.center_angle

    def update(
        self,
        target_center: Tuple[int, int] | None,
        frame_size: Tuple[int, int],
    ) -> ControlResult:
        if target_center is None:
            return ControlResult(self.pan, self.tilt, 0, 0, False)

        width, height = frame_size
        x_error = int(target_center[0] - width / 2)
        y_error = int(target_center[1] - height / 2)

        pan_step = self._axis_step(
            error=x_error,
            gain=self.controller_config.pan_gain,
            sign=self.controller_config.pan_sign,
        )
        tilt_step = self._axis_step(
            error=y_error,
            gain=self.controller_config.tilt_gain,
            sign=self.controller_config.tilt_sign,
        )

        self.pan = clamp(
            self.pan + pan_step,
            self.servo_config.pan.min_angle,
            self.servo_config.pan.max_angle,
        )
        self.tilt = clamp(
            self.tilt + tilt_step,
            self.servo_config.tilt.min_angle,
            self.servo_config.tilt.max_angle,
        )

        return ControlResult(
            pan=self.pan,
            tilt=self.tilt,
            x_error=x_error,
            y_error=y_error,
            moved=bool(pan_step or tilt_step),
            pan_moved=bool(pan_step),
            tilt_moved=bool(tilt_step),
        )

    def _axis_step(self, error: int, gain: float, sign: int) -> float:
        if abs(error) <= self.controller_config.dead_zone:
            return 0.0

        raw_step = sign * gain * error
        return clamp(raw_step, -self.controller_config.max_step, self.controller_config.max_step)
