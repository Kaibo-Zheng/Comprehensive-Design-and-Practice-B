"""基于 RK3588 NPU 的 YOLO RKNN 人体检测器。

该检测器通过板端 RKNN 轻量运行时执行 ``.rknn`` 模型。它与 ONNX 检测器
保持相同的 detect 接口、last_timing 结构和历史目标保持策略，因此主程序、
控制器和指标记录不需要关心检测运行在 CPU 还是 NPU 上。

预处理和后处理逻辑通过 inference.postprocess 与 ONNX 路线共用。模型转换在
主机上完成，板端只负责加载和推理。若无法导入轻量运行时，初始化会直接抛出
明确异常，不会静默回退到 CPU。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from common.config import DetectorConfig

from .detector import Target
from .postprocess import decode_yolo_output, letterbox, select_target


def _import_rknnlite():
    """从可用的轻量运行模块中导入 RKNNLite 类。

    不同版本的板端运行时模块名可能不同；如果都不可用，则抛出明确异常，
    调用方不应回退到 CPU。
    """

    attempts = []
    for module_name in ("rknnlite.api", "rknn_toolkit_lite2"):
        try:
            module = __import__(module_name, fromlist=["RKNNLite"])
            return getattr(module, "RKNNLite")
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{module_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "RKNN Lite runtime not available (tried " + "; ".join(attempts) + "). "
        "Install rknn_toolkit_lite2 on the board (matching the NPU driver from "
        "`cat /sys/kernel/debug/rknpu/version`), or use the yolo_onnx CPU "
        "detector. This detector does not silently fall back to CPU."
    )


class YoloRknnDetector:
    """NPU YOLO 检测器，对外接口与 ONNX 检测器保持一致。"""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.input_size = config.input_size
        self.target_class = config.target_class
        self.conf_thres = config.conf_thres
        self.iou_thres = config.iou_thres
        self.hold_frames = config.hold_frames
        self.last_timing: Dict[str, float] = {}
        self._last_target: Optional[Target] = None
        self._miss_count = 0
        self._rknn = None

        if not config.model_path:
            raise RuntimeError("yolo_rknn detector requires --model <path-to-.rknn>")

        # 先检查当前环境是否能运行 RKNN，避免把运行时问题误报为模型路径问题。
        rknn_lite_cls = _import_rknnlite()

        model_path = Path(config.model_path)
        if not model_path.is_file():
            raise RuntimeError(
                f"RKNN model not found: {model_path}. Convert one on an x86 host with "
                "`python tool/convert_yolo_rknn.py --onnx <model.onnx> "
                f"--output {model_path} --target rk3588` and copy it to the board."
            )

        self._rknn = rknn_lite_cls()
        if self._rknn.load_rknn(str(model_path)) != 0:
            raise RuntimeError(f"RKNNLite.load_rknn failed for {model_path}")

        # 如果运行时暴露了三核 NPU 掩码，则尽量使用全部 NPU 核心；否则交给运行时选择。
        core_mask = getattr(rknn_lite_cls, "NPU_CORE_0_1_2", None)
        ret = (
            self._rknn.init_runtime(core_mask=core_mask)
            if core_mask is not None
            else self._rknn.init_runtime()
        )
        if ret != 0:
            raise RuntimeError(
                "RKNNLite.init_runtime failed. Check the NPU driver and that "
                "`rknn_server` / `librknnrt.so` are present on the board."
            )

    def detect(self, frame: np.ndarray) -> Optional[Target]:
        orig_shape = (frame.shape[0], frame.shape[1])
        frame_center = (frame.shape[1] / 2.0, frame.shape[0] / 2.0)

        t0 = time.perf_counter()
        # 按转换参数，RKNN 模型接收 NHWC uint8 RGB 输入，并在模型内部完成归一化。
        # 因此这里不同于 ONNX 路线，不再除以 255，也不转成 NCHW。
        padded = letterbox(frame, self.input_size)            # 高 x 宽 x 3 的 uint8 BGR 图像。
        rgb = np.ascontiguousarray(padded[:, :, ::-1])        # BGR 转 RGB。
        tensor = rgb[np.newaxis, ...]                         # NHWC 格式，批量大小为 1。
        t1 = time.perf_counter()
        outputs = self._rknn.inference(inputs=[tensor])
        t2 = time.perf_counter()

        detections = decode_yolo_output(
            np.asarray(outputs[0]),
            orig_shape=orig_shape,
            input_size=self.input_size,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
        )
        chosen = select_target(detections, frame_center, target_class=self.target_class or None)
        t3 = time.perf_counter()

        self.last_timing = {
            "preprocess_ms": (t1 - t0) * 1000.0,
            "inference_ms": (t2 - t1) * 1000.0,
            "postprocess_ms": (t3 - t2) * 1000.0,
        }

        if chosen is not None:
            target = self._to_target(chosen)
            self._last_target = target
            self._miss_count = 0
            return target

        # 与其他检测器一致，短时间漏检时保留上一目标并标记为历史目标。
        if self._last_target is not None and self._miss_count < self.hold_frames:
            self._miss_count += 1
            return Target(
                center=self._last_target.center,
                bbox=self._last_target.bbox,
                score=self._last_target.score,
                stale=True,
                label=self._last_target.label,
                confidence=self._last_target.confidence,
            )

        self._last_target = None
        return None

    @staticmethod
    def _to_target(det: Dict) -> Target:
        x, y, w, h = det["bbox"]
        x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
        return Target(
            center=(x + w // 2, y + h // 2),
            bbox=(x, y, w, h),
            score=float(det["confidence"]),
            stale=False,
            label=str(det["label"]),
            confidence=float(det["confidence"]),
        )

    def close(self) -> None:
        """释放 NPU 运行时；重复调用也是安全的。"""

        rknn = getattr(self, "_rknn", None)
        if rknn is not None:
            try:
                rknn.release()
            except Exception:  # noqa: BLE001
                pass
            self._rknn = None
