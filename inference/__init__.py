"""推理与检测后端入口。"""

from .detector import PersonDetector, Target

__all__ = ["PersonDetector", "Target", "YoloOnnxDetector", "YoloRknnDetector"]


def __getattr__(name: str):
    if name == "YoloOnnxDetector":
        from .yolo_onnx_detector import YoloOnnxDetector

        return YoloOnnxDetector
    if name == "YoloRknnDetector":
        from .yolo_rknn_detector import YoloRknnDetector

        return YoloRknnDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
