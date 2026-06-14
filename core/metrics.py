"""Runtime metrics and CSV logging.

The CSV schema is detector-agnostic so OpenCV, YOLO ONNX and YOLO RKNN runs are
directly comparable. Per-stage timings (`capture_ms`, `preprocess_ms`,
`inference_ms`, `postprocess_ms`, `total_ms`) default to ``0.0`` for detectors
that do not break their work down (e.g. the OpenCV cascades), but the columns
are always present and stable.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TextIO


@dataclass
class FrameTimings:
    """Per-frame stage timings in milliseconds."""

    capture_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class MetricsSnapshot:
    frame: int
    fps: float
    error_px: float
    average_error_px: float
    lost_frames: int


CSV_FIELDS = [
    "timestamp",
    "frame_index",
    "fps",
    "detector",
    "target_found",
    "target_stale",
    "target_label",
    "target_confidence",
    "x_error",
    "y_error",
    "error_norm",
    "pan",
    "tilt",
    "capture_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "total_ms",
    "lost_frames",
]


class TrackingMetrics:
    def __init__(self, log_path: str | None = None, detector: str = ""):
        self.start_time = time.perf_counter()
        self.last_frame_time = self.start_time
        self.frame_count = 0
        self.fps_ema = 0.0
        self.detector = detector
        self.errors: list[float] = []
        self.lost_frames = 0
        # Running sums for averaged timing in summary().
        self._timing_sums = {k: 0.0 for k in
                             ("capture_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "total_ms")}
        self._log_file: TextIO | None = None
        self._writer: csv.DictWriter | None = None

        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._log_file, fieldnames=CSV_FIELDS)
            self._writer.writeheader()

    def update(
        self,
        found: bool,
        stale: bool,
        x_error: int,
        y_error: int,
        pan: float,
        tilt: float,
        label: str = "",
        confidence: Optional[float] = None,
        timings: Optional[FrameTimings] = None,
    ) -> MetricsSnapshot:
        now = time.perf_counter()
        dt = max(1e-9, now - self.last_frame_time)
        self.last_frame_time = now
        instant_fps = 1.0 / dt
        self.fps_ema = instant_fps if self.fps_ema == 0.0 else 0.9 * self.fps_ema + 0.1 * instant_fps
        self.frame_count += 1

        error_px = math.hypot(x_error, y_error) if found else 0.0
        if found and not stale:
            self.errors.append(error_px)
        else:
            self.lost_frames += 1

        average_error = sum(self.errors) / len(self.errors) if self.errors else 0.0

        timings = timings or FrameTimings()
        for key in self._timing_sums:
            self._timing_sums[key] += getattr(timings, key)

        if self._writer is not None:
            self._writer.writerow(
                {
                    "timestamp": f"{now - self.start_time:.3f}",
                    "frame_index": self.frame_count,
                    "fps": f"{self.fps_ema:.2f}",
                    "detector": self.detector,
                    "target_found": int(found),
                    "target_stale": int(stale),
                    "target_label": label,
                    "target_confidence": "" if confidence is None else f"{confidence:.4f}",
                    "x_error": x_error,
                    "y_error": y_error,
                    "error_norm": f"{error_px:.2f}",
                    "pan": f"{pan:.2f}",
                    "tilt": f"{tilt:.2f}",
                    "capture_ms": f"{timings.capture_ms:.3f}",
                    "preprocess_ms": f"{timings.preprocess_ms:.3f}",
                    "inference_ms": f"{timings.inference_ms:.3f}",
                    "postprocess_ms": f"{timings.postprocess_ms:.3f}",
                    "total_ms": f"{timings.total_ms:.3f}",
                    "lost_frames": self.lost_frames,
                }
            )

        return MetricsSnapshot(
            frame=self.frame_count,
            fps=self.fps_ema,
            error_px=error_px,
            average_error_px=average_error,
            lost_frames=self.lost_frames,
        )

    def summary(self) -> dict[str, float | int | str]:
        runtime = time.perf_counter() - self.start_time
        average_error = sum(self.errors) / len(self.errors) if self.errors else 0.0
        max_error = max(self.errors) if self.errors else 0.0
        frames = max(1, self.frame_count)
        return {
            "detector": self.detector,
            "frames": self.frame_count,
            "runtime_s": runtime,
            "average_fps": self.frame_count / runtime if runtime > 0 else 0.0,
            "ema_fps": self.fps_ema,
            "average_error_px": average_error,
            "max_error_px": max_error,
            "lost_frames": self.lost_frames,
            "avg_capture_ms": self._timing_sums["capture_ms"] / frames,
            "avg_inference_ms": self._timing_sums["inference_ms"] / frames,
            "avg_total_ms": self._timing_sums["total_ms"] / frames,
        }

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
