#!/usr/bin/env python3
"""Benchmark Python NMS against the optional C extension NMS."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ACCEL_DIR = ROOT / "acceleration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO postprocess NMS")
    parser.add_argument("--boxes", type=int, default=1200)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def make_boxes(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 620, size=count)
    y1 = rng.uniform(0, 460, size=count)
    w = rng.uniform(10, 120, size=count)
    h = rng.uniform(10, 160, size=count)
    boxes = np.stack([x1, y1, np.minimum(x1 + w, 640), np.minimum(y1 + h, 480)], axis=1).astype(np.float64)
    scores = rng.uniform(0.25, 0.99, size=count).astype(np.float64)
    return boxes, scores


def time_call(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) * 1000.0 / iterations


def python_nms_reference(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    # Force the pure-Python path by temporarily bypassing the optional C helper.
    import inference.postprocess as postprocess

    original = postprocess._nms_c
    postprocess._nms_c = lambda *_args, **_kwargs: None
    try:
        return postprocess.nms(boxes, scores, iou_threshold)
    finally:
        postprocess._nms_c = original


def load_cconv_module():
    try:
        installed = importlib.import_module("cpp_conv._cconv")
        if getattr(installed, "nms_xyxy", None) is not None:
            return installed
    except Exception:
        pass

    for shared_object in sorted(ACCEL_DIR.glob("_cconv*.so")):
        spec = importlib.util.spec_from_file_location("_cconv", shared_object)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError(f"local _cconv extension not found under {ACCEL_DIR}")


def main() -> int:
    args = parse_args()
    boxes, scores = make_boxes(args.boxes, args.seed)

    from inference.postprocess import nms

    _cconv = load_cconv_module()
    py_keep = python_nms_reference(boxes, scores, args.iou_thres)
    c_keep = nms(boxes, scores, args.iou_thres)
    direct_c_keep = [int(i) for i in _cconv.nms_xyxy(boxes, scores, args.iou_thres)]
    if py_keep != c_keep or c_keep != direct_c_keep:
        print("error: Python NMS and C NMS results differ")
        print(f"python head={py_keep[:10]}")
        print(f"wrapped C head={c_keep[:10]}")
        print(f"direct C head={direct_c_keep[:10]}")
        return 1

    py_ms = time_call(lambda: python_nms_reference(boxes, scores, args.iou_thres), args.iterations)
    c_ms = time_call(lambda: nms(boxes, scores, args.iou_thres), args.iterations)
    speedup = py_ms / c_ms if c_ms > 0 else float("inf")

    print(f"Boxes: {args.boxes}")
    print(f"Iterations: {args.iterations}")
    print(f"Kept boxes: {len(c_keep)}")
    print(f"Python NMS: {py_ms:.3f} ms/run")
    print(f"C NMS: {c_ms:.3f} ms/run")
    print(f"NMS speedup: {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
