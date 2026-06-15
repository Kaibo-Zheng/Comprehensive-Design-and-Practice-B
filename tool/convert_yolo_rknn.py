#!/usr/bin/env python3
"""将 YOLO ONNX 模型转换为 RK3588 NPU 使用的 RKNN 模型。

本脚本使用完整转换工具 ``rknn-toolkit2``（即 ``rknn.api.RKNN`` 类），通常在
x86 主机上安装。开发板运行生成的 ``.rknn`` 文件时只需要 ``rknn_toolkit_lite2``，
对应运行代码见 ``inference/yolo_rknn_detector.py``。

如果当前环境无法导入转换工具，脚本会输出明确提示并退出，而不是直接打印 traceback。

示例：

    python tool/convert_yolo_rknn.py \
        --onnx model/yolo11n.onnx \
        --output model/yolo11n.rknn \
        --target rk3588
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 YOLO ONNX 模型转换为 RKNN")
    parser.add_argument("--onnx", required=True, help="输入 ONNX 模型路径")
    parser.add_argument("--output", required=True, help="输出 .rknn 模型路径")
    parser.add_argument("--target", default="rk3588", help="目标平台，默认 rk3588")
    parser.add_argument(
        "--dataset",
        default=None,
        help="可选校准图像列表文件，每行一个路径；提供后启用 INT8 量化，否则保留浮点模型。",
    )
    parser.add_argument("--mean", default="0,0,0", help="各通道均值，使用逗号分隔")
    parser.add_argument("--std", default="255,255,255", help="各通道标准差，使用逗号分隔")
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
            "error: 当前环境不可用 rknn-toolkit2 "
            f"({type(exc).__name__}: {exc}).\n"
            "完整转换工具通常在 x86_64 主机上运行：\n"
            "  pip install rknn-toolkit2\n"
            "然后在主机上运行本脚本，并把生成的 .rknn 文件复制回开发板。\n"
            "开发板只需要 rknn_toolkit_lite2 来运行模型。",
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
