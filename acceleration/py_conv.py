"""5x5 卷积的 Python/NumPy 参考实现。"""

from __future__ import annotations

from typing import Any

import numpy as np


def default_kernel() -> np.ndarray:
    kernel = np.array(
        [
            [1, 2, 3, 2, 1],
            [2, 4, 6, 4, 2],
            [3, 6, 9, 6, 3],
            [2, 4, 6, 4, 2],
            [1, 2, 3, 2, 1],
        ],
        dtype=np.float32,
    )
    return kernel / kernel.sum()


def conv5x5_python(image: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    """执行边缘复制填充的 5x5 卷积。

    这里刻意使用 Python 循环作为较慢的参考实现，用于和 C 扩展对比。
    """

    if image.ndim != 2:
        raise ValueError("conv5x5_python expects a 2D grayscale image")

    if kernel is None:
        kernel = default_kernel()

    kernel = np.asarray(kernel, dtype=np.float32)
    if kernel.shape != (5, 5):
        raise ValueError("kernel must have shape (5, 5)")

    src = np.asarray(image, dtype=np.float32)
    height, width = src.shape
    out = np.empty((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            acc = 0.0
            for ky in range(5):
                yy = min(height - 1, max(0, y + ky - 2))
                for kx in range(5):
                    xx = min(width - 1, max(0, x + kx - 2))
                    acc += float(src[yy, xx]) * float(kernel[ky, kx])
            out[y, x] = acc

    return out


def conv5x5_pymp(
    image: np.ndarray,
    kernel: np.ndarray | None = None,
    *,
    threads: int = 4,
) -> np.ndarray:
    """使用 pymp 按行并行执行相同的 5x5 卷积。

    该实现保留 Python 循环算法，但把图像行分配给多个 worker 处理，用于和串行
    Python 以及 C 扩展进行性能对比。
    """

    try:
        import pymp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("pymp is not installed; install pymp-pypi") from exc

    if image.ndim != 2:
        raise ValueError("conv5x5_pymp expects a 2D grayscale image")

    if kernel is None:
        kernel = default_kernel()

    kernel = np.asarray(kernel, dtype=np.float32)
    if kernel.shape != (5, 5):
        raise ValueError("kernel must have shape (5, 5)")

    src = np.asarray(image, dtype=np.float32)
    height, width = src.shape
    out: Any = pymp.shared.array((height, width), dtype="float32")
    worker_count = max(1, int(threads))

    with pymp.Parallel(worker_count) as parallel:
        for y in parallel.range(height):
            for x in range(width):
                acc = 0.0
                for ky in range(5):
                    yy = min(height - 1, max(0, y + ky - 2))
                    for kx in range(5):
                        xx = min(width - 1, max(0, x + kx - 2))
                        acc += float(src[yy, xx]) * float(kernel[ky, kx])
                out[y, x] = acc

    return np.asarray(out, dtype=np.float32)
