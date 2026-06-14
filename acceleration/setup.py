from pathlib import Path
import os

from setuptools import Extension, setup

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required to build cpp_conv") from exc


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

setup(
    name="cpp-conv",
    version="0.1.0",
    description="5x5 convolution C extension for the gimbal tracker project",
    packages=["cpp_conv"],
    package_dir={"cpp_conv": "."},
    ext_modules=[
        Extension(
            "cpp_conv._cconv",
            sources=[str(ROOT / "src" / "cconv.c")],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"],
        )
    ],
)
