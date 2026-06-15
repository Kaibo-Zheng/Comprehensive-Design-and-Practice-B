"""对比 Python、pymp 和 C 扩展的 5x5 卷积性能。"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACCEL_DIR = Path(__file__).resolve().parent
if str(ACCEL_DIR) not in sys.path:
    sys.path.insert(0, str(ACCEL_DIR))

from py_conv import conv5x5_pymp, conv5x5_python, default_kernel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark 5x5 convolution implementations")
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pymp-threads", type=int, default=4)
    parser.add_argument("--skip-pymp", action="store_true", help="Only benchmark Python and C")
    return parser.parse_args()


def time_call(func, image: np.ndarray, kernel: np.ndarray, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        func(image, kernel)
    end = time.perf_counter()
    return (end - start) * 1000.0 / iterations


def load_cconv_module():
    try:
        return importlib.import_module("cpp_conv._cconv")
    except Exception:
        pass

    for pattern in ("_cconv*.so", "_cconv*.pyd"):
        for shared_object in sorted(ACCEL_DIR.glob(pattern)):
            spec = importlib.util.spec_from_file_location("_cconv", shared_object)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    for pattern in ("build/**/_cconv*.so", "build/**/_cconv*.pyd"):
        for shared_object in sorted(ACCEL_DIR.glob(pattern)):
            spec = importlib.util.spec_from_file_location("_cconv", shared_object)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ImportError(f"local _cconv extension not found under {ACCEL_DIR}")


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    image = rng.integers(0, 256, size=(args.height, args.width), dtype=np.uint8)
    kernel = default_kernel()

    try:
        _cconv = load_cconv_module()
    except ImportError as exc:
        print("C extension is not built yet.")
        print("Build it with: python acceleration/setup.py build_ext --inplace")
        print(f"Import error: {exc}")
        return 1

    py_out = conv5x5_python(image, kernel)
    c_out = _cconv.conv5x5(image, kernel)
    max_diff = float(np.max(np.abs(py_out - c_out)))
    if max_diff > 1e-4:
        print(f"warning: output max difference is {max_diff:.6f}")
    py_ms = time_call(conv5x5_python, image, kernel, args.iterations)
    pymp_ms = None
    pymp_speedup = None
    pymp_diff = None
    pymp_error = None
    if not args.skip_pymp:
        try:
            pymp_out = conv5x5_pymp(image, kernel, threads=args.pymp_threads)
            pymp_diff = float(np.max(np.abs(py_out - pymp_out)))
            pymp_ms = time_call(
                lambda img, ker: conv5x5_pymp(img, ker, threads=args.pymp_threads),
                image,
                kernel,
                args.iterations,
            )
            pymp_speedup = py_ms / pymp_ms if pymp_ms > 0 else float("inf")
        except RuntimeError as exc:
            pymp_error = str(exc)

    c_ms = time_call(_cconv.conv5x5, image, kernel, args.iterations)
    speedup = py_ms / c_ms if c_ms > 0 else float("inf")

    print(f"Frame size: {args.width}x{args.height}")
    print(f"Iterations: {args.iterations}")
    print(f"Max output diff: {max_diff:.6f}")
    print(f"Python 5x5 convolution: {py_ms:.3f} ms/frame")
    if pymp_ms is not None:
        print(f"pymp 5x5 convolution ({args.pymp_threads} threads): {pymp_ms:.3f} ms/frame")
        print(f"pymp max output diff: {pymp_diff:.6f}")
        print(f"pymp speedup: {pymp_speedup:.2f}x")
    elif pymp_error is not None:
        print(f"pymp 5x5 convolution: skipped ({pymp_error})")
    print(f"C extension 5x5 convolution: {c_ms:.3f} ms/frame")
    print(f"Speedup: {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
