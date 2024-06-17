import os, sys

try:
    from skbuild import setup
except ImportError:
    print('scikit-build is required to build from source.', file=sys.stderr)
    print('Please run:', file=sys.stderr)
    print('', file=sys.stderr)
    print('  python -m pip install scikit-build')
    sys.exit(1)

import torch
torch_root = os.path.dirname(torch.__file__)

setup(
    name="tet_spheres",
    version="0.0.1",
    description="a minimal example package for pytorch extension (with pybind11 and scikit-build)",
    license="MIT",
    packages=['tet_spheres'],
    cmake_args=[f'-DCMAKE_PREFIX_PATH={torch_root}'] # to specify CUDA location: -DCMAKE_CUDA_COMPILER=...
)