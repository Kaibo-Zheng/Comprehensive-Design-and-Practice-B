"""PCA9685-backed two-axis gimbal control with mock mode."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from core.config import ServoAxisConfig, ServoConfig, clamp

Axis = Literal["pan", "tilt"]


class PCA9685Lite:
    """Small smbus2 PCA9685 driver used by the tracker."""

    MODE1 = 0x00
    PRESCALE = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, bus_number: int, address: int, frequency_hz: int = 50):
        from smbus2 import SMBus

        self.bus = SMBus(bus_number)
        self.address = address
        self.set_frequency(frequency_hz)

    def set_frequency(self, frequency_hz: int) -> None:
        prescale = int(round(25_000_000.0 / (4096.0 * frequency_hz) - 1.0))
        self.bus.write_byte_data(self.address, self.MODE1, 0x10)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, self.PRESCALE, prescale)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, self.MODE1, 0x20)
        time.sleep(0.005)

    def set_pwm(self, channel: int, on: int, off: int) -> None:
        reg = self.LED0_ON_L + 4 * channel
        data = [on & 0xFF, on >> 8, off & 0xFF, off >> 8]
        self.bus.write_i2c_block_data(self.address, reg, data)

    def close(self) -> None:
        self.bus.close()


@dataclass
class GimbalState:
    pan: float
    tilt: float


class Gimbal:
    def __init__(self, config: ServoConfig, mock: bool = False):
        self.config = config
        self.mock = mock
        self.state = GimbalState(
            pan=config.pan.center_angle,
            tilt=config.tilt.center_angle,
        )
        self._driver = None if mock else PCA9685Lite(
            bus_number=config.bus_number,
            address=config.address,
            frequency_hz=config.frequency_hz,
        )

    def set_angle(self, axis: Axis, angle: float) -> None:
        axis_config = self._axis_config(axis)
        safe_angle = clamp(angle, axis_config.min_angle, axis_config.max_angle)

        if axis == "pan":
            self.state.pan = safe_angle
        else:
            self.state.tilt = safe_angle

        if self.mock:
            return

        pulse = self.angle_to_pulse(safe_angle)
        self._driver.set_pwm(axis_config.channel, 0, pulse)

    def move_to(self, pan: float, tilt: float) -> None:
        self.set_angle("pan", pan)
        self.set_angle("tilt", tilt)

    def center(self) -> None:
        self.move_to(self.config.pan.center_angle, self.config.tilt.center_angle)

    def center_axis(self, axis: Axis) -> None:
        self.set_angle(axis, self._axis_config(axis).center_angle)

    def safe_calibration(self, delay: float = 0.4) -> None:
        self.center()
        time.sleep(delay)

        for axis, axis_config in (("pan", self.config.pan), ("tilt", self.config.tilt)):
            for offset in (5.0, -5.0, 0.0):
                target = axis_config.center_angle + offset
                print(f"calibrate {axis}: {target:.1f} deg")
                self.set_angle(axis, target)
                time.sleep(delay)

    def angle_to_pulse(self, angle: float) -> int:
        ratio = clamp(angle, 0.0, 180.0) / 180.0
        pulse = self.config.pulse_min + ratio * (self.config.pulse_max - self.config.pulse_min)
        return int(round(clamp(pulse, 0, 4095)))

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def _axis_config(self, axis: Axis) -> ServoAxisConfig:
        if axis == "pan":
            return self.config.pan
        if axis == "tilt":
            return self.config.tilt
        raise ValueError(f"Unsupported axis: {axis}")
