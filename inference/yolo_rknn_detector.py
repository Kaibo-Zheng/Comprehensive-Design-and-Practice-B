"""YOLO RKNN (RK3588 NPU) person detector.

Runs a ``.rknn`` model on the NPU through the on-board runtime
``rknn_toolkit_lite2`` (imported as ``rknnlite``). It mirrors
:class:`inference.yolo_onnx_detector.YoloOnnxDetector` exactly — same
``detect(frame) -> Target | None`` interface, same ``last_timing`` dict, same
stale-hold behaviour — so ``app.py`` / controller / metrics never need to know
whether detection runs on the CPU (ONNX) or the NPU (RKNN).

Pre/post-processing is shared with the ONNX path via
:mod:`inference.postprocess` (``letterbox`` + ``decode_yolo_output``).

Runtime split:

- **Conversion** (ONNX -> ``.rknn``) uses the full ``rknn-toolkit2`` on an x86
  host — see ``scripts/convert_yolo_rknn.sh``.
- **Inference** (this file) uses ``rknn_toolkit_lite2`` on the board.

If the lite runtime is not importable, ``__init__`` raises a clear
``RuntimeError`` and **never silently falls back to the CPU** (TASK-007).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from core.config import DetectorConfig

from .detector import Target
from .postprocess import decode_yolo_output, letterbox, select_target


def _import_rknnlite():
    """Return the ``RKNNLite`` class from whichever lite module is installed.

    The on-board runtime ships under two names depending on the release
    (``rknnlite`` or ``rknn_toolkit_lite2``). Raise a clear, actionable
    ``RuntimeError`` if neither imports — callers must not fall back to CPU.
    """

    attempts = []
    for module_name in ("rknnlite.api", "rknn_toolkit_lite2"):
        try:
            module = __import__(module_name, fromlist=["RKNNLite"])
            return getattr(module, "RKNNLite")
        except Exception as exc:  # noqa: BLE001 - report every failure verbatim
            attempts.append(f"{module_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "RKNN Lite runtime not available (tried " + "; ".join(attempts) + "). "
        "Install rknn_toolkit_lite2 on the board (matching the NPU driver from "
        "`cat /sys/kernel/debug/rknpu/version`), or use the yolo_onnx CPU "
        "detector. This detector does not silently fall back to CPU."
    )


class YoloRknnDetector:
    """NPU YOLO detector. Same public surface as ``YoloOnnxDetector``."""

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

        # Precondition first: can we even run RKNN on this host? If not, say so
        # plainly regardless of the model path (this is the common board blocker).
        rknn_lite_cls = _import_rknnlite()

        model_path = Path(config.model_path)
        if not model_path.is_file():
            raise RuntimeError(
                f"RKNN model not found: {model_path}. Convert one on an x86 host with "
                "`bash scripts/convert_yolo_rknn.sh --onnx <model.onnx> "
                f"--output {model_path} --target rk3588` and copy it to the board."
            )

        self._rknn = rknn_lite_cls()
        if self._rknn.load_rknn(str(model_path)) != 0:
            raise RuntimeError(f"RKNNLite.load_rknn failed for {model_path}")

        # Spread inference across all three RK3588 NPU cores when the constant is
        # exposed; otherwise let the runtime pick a default core.
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
        # RKNN models converted with mean=0 / std=255 (the default in
        # scripts/convert_yolo_rknn.sh) take a letterboxed **NHWC uint8 RGB**
        # image and apply the /255 normalization internally — so, unlike the
        # ONNX path, we do NOT divide here or transpose to NCHW.
        padded = letterbox(frame, self.input_size)            # HxWx3 uint8 BGR
        rgb = np.ascontiguousarray(padded[:, :, ::-1])        # BGR -> RGB
        tensor = rgb[np.newaxis, ...]                         # NHWC, batch=1
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

        # Hold the last target as stale for a few frames, like the other detectors.
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
        """Release the NPU runtime. Safe to call more than once."""

        rknn = getattr(self, "_rknn", None)
        if rknn is not None:
            try:
                rknn.release()
            except Exception:  # noqa: BLE001 - best-effort cleanup on shutdown
                pass
            self._rknn = None
