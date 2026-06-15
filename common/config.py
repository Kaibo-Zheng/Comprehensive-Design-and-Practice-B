"""云台跟踪系统的配置对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CameraConfig:
    source: int | str = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = "MJPG"


@dataclass
class DetectorConfig:
    detector: str = "face"
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: Tuple[int, int] = (40, 40)
    hold_frames: int = 5
    cascade_path: str | None = None
    # YOLO（ONNX / RKNN）相关选项；OpenCV 检测器会忽略这些字段。
    model_path: str | None = None
    target_class: str = "person"
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    input_size: int = 640


@dataclass
class ServoAxisConfig:
    channel: int
    min_angle: float
    max_angle: float
    center_angle: float = 90.0


@dataclass
class ServoConfig:
    bus_number: int = 1
    address: int = 0x40
    frequency_hz: int = 50
    pan: ServoAxisConfig = field(
        default_factory=lambda: ServoAxisConfig(channel=0, min_angle=45.0, max_angle=135.0)
    )
    tilt: ServoAxisConfig = field(
        default_factory=lambda: ServoAxisConfig(channel=1, min_angle=60.0, max_angle=120.0)
    )
    pulse_min: int = 150
    pulse_max: int = 600


@dataclass
class ControllerConfig:
    pan_gain: float = 0.015
    tilt_gain: float = 0.015
    pan_sign: int = 1
    tilt_sign: int = -1
    dead_zone: int = 25
    max_step: float = 2.0


@dataclass
class VoiceConfig:
    enabled: bool = False
    mode: str = "play"
    text: str = "检测到{label_zh}。"
    cooldown_s: float = 5.0
    audio_path: str | None = None
    model_dir: str | None = None
    runtime_dir: str | None = None
    output_dir: str = "audio"
    voice_preset: str = "Junhao"
    sample_mode: str = "greedy"
    max_new_frames: int = 24
    threads: int = 2
    playback: bool = True
    player: str = "aplay"
    seed: int | None = 1
    enable_wetext: bool = False


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_camera_source(value: str) -> int | str:
    """将命令行字符串转换为 OpenCV 摄像头输入。

    纯数字字符串会被视为摄像头编号，其他内容会作为设备路径传给
    OpenCV，例如稳定的 USB 摄像头路径。
    """

    return int(value) if value.isdigit() else value
