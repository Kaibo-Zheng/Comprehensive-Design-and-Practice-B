"""OpenCV person detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from core.config import DetectorConfig


@dataclass
class Target:
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    score: float
    stale: bool = False
    label: str = "person"
    confidence: float | None = None

    @property
    def area(self) -> int:
        _, _, w, h = self.bbox
        return int(w * h)


class PersonDetector:
    """Detect a person using lightweight OpenCV detectors.

    The default mode tracks a visible face and treats it as the person's center.
    This is usually more stable than full-body detection for a gimbal-mounted
    webcam at close range.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.detector = config.detector
        self._last_target: Target | None = None
        self._miss_count = 0

        if self.detector in {"face", "upperbody"}:
            self._cascade = cv2.CascadeClassifier(self._cascade_path())
            if self._cascade.empty():
                raise RuntimeError(f"Failed to load OpenCV cascade for {self.detector}")
        elif self.detector == "hog":
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        else:
            raise ValueError(f"Unsupported detector: {self.detector}")

    def detect(self, frame: np.ndarray) -> Target | None:
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        boxes = self._detect_boxes(frame)

        if boxes:
            target = self._select_target(boxes, frame_center)
            self._last_target = target
            self._miss_count = 0
            return target

        if self._last_target is not None and self._miss_count < self.config.hold_frames:
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

    def _cascade_path(self) -> str:
        if self.config.cascade_path:
            return self.config.cascade_path

        cascade_name = {
            "face": "haarcascade_frontalface_default.xml",
            "upperbody": "haarcascade_upperbody.xml",
        }[self.detector]
        return str(Path(cv2.data.haarcascades) / cascade_name)

    def _detect_boxes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.detector in {"face", "upperbody"}:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            boxes = self._cascade.detectMultiScale(
                gray,
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
                minSize=self.config.min_size,
            )
            return [tuple(map(int, box)) for box in boxes]

        boxes, _ = self._hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        return [tuple(map(int, box)) for box in boxes]

    def _select_target(
        self,
        boxes: Iterable[Tuple[int, int, int, int]],
        frame_center: Tuple[int, int],
    ) -> Target:
        def rank(box: Tuple[int, int, int, int]) -> float:
            x, y, w, h = box
            cx = x + w // 2
            cy = y + h // 2
            dist = ((cx - frame_center[0]) ** 2 + (cy - frame_center[1]) ** 2) ** 0.5
            return float(w * h) - 0.25 * dist

        best = max(boxes, key=rank)
        x, y, w, h = best
        return Target(
            center=(int(x + w // 2), int(y + h // 2)),
            bbox=(int(x), int(y), int(w), int(h)),
            score=rank(best),
            stale=False,
            label=self.detector,
        )
