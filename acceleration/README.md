# acceleration

本目录集中存放项目里的 C/C++ 加速相关代码。

- `src/cconv.c`: C 扩展源码，提供 `_cconv.conv5x5` 和 `_cconv.nms_xyxy`。
- `py_conv.py`: 纯 Python/NumPy 与 `pymp` 对照实现。
- `benchmark.py`: 卷积加速 benchmark。
- `dist/`: 已构建的 wheel 包。

常用命令：

```bash
python acceleration/benchmark.py --height 160 --width 240 --iterations 3
python acceleration/setup.py build_ext --inplace
python -m build acceleration
```
