#!/usr/bin/env python3
"""Generate comparison figures for the four bonus items."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "illustration"
OUT.mkdir(parents=True, exist_ok=True)


def save_bar(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
    color: str = "#2a9d8f",
    annotate_fmt: str = "{:.2f}",
) -> None:
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(labels, values, color=color)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    ymax = max(values) if values else 1.0
    plt.ylim(0, ymax * 1.18)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.03,
            annotate_fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> int:
    # Bonus 1: C/C++ wheel acceleration
    save_bar(
        OUT / "bonus1_cpp_conv_compare.png",
        "Bonus 1: 5x5 Convolution Acceleration",
        ["Python", "pymp x4", "C"],
        [1493.016, 447.617, 0.915],
        "Latency (ms/frame)",
        color="#457b9d",
        annotate_fmt="{:.3f}",
    )

    # Bonus 2: network programming / web integration
    save_bar(
        OUT / "bonus2_web_modes_fps.png",
        "Bonus 2: Networked Web Monitor Runtime Modes",
        ["RKNN only", "RKNN + Web", "RKNN + Web + Voice"],
        [14.65, 7.18, 16.00],
        "Observed EMA FPS",
        color="#e76f51",
        annotate_fmt="{:.2f}",
    )

    # Bonus 3: task acceleration with pymp
    save_bar(
        OUT / "bonus3_pymp_thread_compare.png",
        "Bonus 3: pymp Thread Scaling on 5x5 Convolution",
        ["Python", "pymp x1", "pymp x2", "pymp x4", "pymp x8"],
        [741.116, 814.057, 429.372, 253.273, 219.841],
        "Latency (ms/frame)",
        color="#6a4c93",
        annotate_fmt="{:.1f}",
    )

    # Bonus 4: RKNN deployment + C NMS for the live DNN path
    save_bar(
        OUT / "bonus4_rknn_and_nms_compare.png",
        "Bonus 4: DNN Deployment and Postprocess Acceleration",
        ["YOLO ONNX CPU", "YOLO RKNN NPU", "Python NMS", "C NMS"],
        [137.0, 53.6, 52.463, 7.093],
        "Latency (ms)",
        color="#2a9d8f",
        annotate_fmt="{:.3f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
