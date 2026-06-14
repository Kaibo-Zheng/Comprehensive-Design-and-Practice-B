#!/usr/bin/env python3
"""Convert a YOLO ONNX model to RKNN for the RK3588 NPU.

This uses the **full** conversion toolkit ``rknn-toolkit2`` (the ``rknn.api.RKNN``
class), which is normally installed on an x86 host. The board itself only needs
``rknn_toolkit_lite2`` to *run* the resulting ``.rknn`` (see
``inference/yolo_rknn_detector.py``).

If the toolkit is not importable, this script exits with a clear message rather
than a traceback, and tells you where to get it.

Example::

    bash scripts/convert_yolo_rknn.sh \
        --onnx model/yolo11n.onnx \
        --output model/yolo11n.rknn \
        --target rk3588
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLO ONNX to RKNN")
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output .rknn path")
    parser.add_argument("--target", default="rk3588", help="Target platform (default rk3588)")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional text file listing calibration images, one path per line. "
        "If given, INT8 quantization is enabled; otherwise the model stays float.",
    )
    parser.add_argument("--mean", default="0,0,0", help="Per-channel mean, comma-separated")
    parser.add_argument("--std", default="255,255,255", help="Per-channel std, comma-separated")
    return parser.parse_args()


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",")]


def main() -> int:
    args = parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        print(f"error: ONNX model not found: {onnx_path}", file=sys.stderr)
        return 2

    try:
        from rknn.api import RKNN  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(
            "error: rknn-toolkit2 is not available in this environment "
            f"({type(exc).__name__}: {exc}).\n"
            "The full conversion toolkit normally runs on an x86_64 host:\n"
            "  pip install rknn-toolkit2   # x86 host, matching your runtime version\n"
            "Then run this script there and copy the .rknn back to the board.\n"
            "The board only needs rknn_toolkit_lite2 to run the model.",
            file=sys.stderr,
        )
        return 3

    quantize = args.dataset is not None
    mean = _floats(args.mean)
    std = _floats(args.std)

    rknn = RKNN(verbose=True)
    print(f"[1/5] config (target={args.target}, quantize={quantize})")
    rknn.config(mean_values=[mean], std_values=[std], target_platform=args.target)

    print(f"[2/5] load_onnx {onnx_path}")
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        print("error: load_onnx failed", file=sys.stderr)
        return 4

    print("[3/5] build")
    if rknn.build(do_quantization=quantize, dataset=args.dataset) != 0:
        print("error: build failed", file=sys.stderr)
        return 5

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[4/5] export_rknn {out_path}")
    if rknn.export_rknn(str(out_path)) != 0:
        print("error: export_rknn failed", file=sys.stderr)
        return 6

    print("[5/5] release")
    rknn.release()
    print(f"done: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
