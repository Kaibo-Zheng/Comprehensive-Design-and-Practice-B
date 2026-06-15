"""5x5 卷积加速实验模块。"""

from .py_conv import conv5x5_pymp, conv5x5_python, default_kernel

__all__ = ["conv5x5_pymp", "conv5x5_python", "default_kernel"]
