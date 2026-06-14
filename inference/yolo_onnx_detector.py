"""YOLO ONNX (CPU) person detector.

Loads a YOLOv5/YOLOv8/YOLOv11 ONNX model with onnxruntime and exposes the same
``detect(frame) -> Target | None`` interface as the OpenCV ``PersonDetector``,
so the controller/gimbal/metrics never need to know which detector is active.

Pre/post-processing (letterbox, decode, NMS, target selection) is shared with
the RKNN detector via :mod:`inference.postprocess`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from core.config import DetectorConfig

from .detector import Target
from .postprocess import decode_yolo_output, preprocess_yolo, select_target


class YoloOnnxDetector:
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

        if not config.model_path:
            raise RuntimeError("yolo_onnx detector requires --model <path-to-.onnx>")
        model_path = Path(config.model_path)
        if not model_path.is_file():
            raise RuntimeError(
                f"ONNX model not found: {model_path}. "
                "Pass an existing model via --model, or export one "
                "(see model/README.md)."
            )

        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(f"onnxruntime is not installed: {exc}") from exc

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        # Honor a fixed model input size if the graph declares one.
        ishape = self.session.get_inputs()[0].shape
        if len(ishape) == 4 and isinstance(ishape[2], int) and ishape[2] > 0:
            self.input_size = int(ishape[2])

    def detect(self, frame: np.ndarray) -> Optional[Target]:
        orig_shape = (frame.shape[0], frame.shape[1])
        frame_center = (frame.shape[1] / 2.0, frame.shape[0] / 2.0)

        t0 = time.perf_counter()
        tensor = preprocess_yolo(frame, self.input_size)
        t1 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        t2 = time.perf_counter()

        detections = decode_yolo_output(
            outputs[0],
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

        # Hold the last target as stale for a few frames, like the OpenCV path.
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
