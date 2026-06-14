from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

import sys
import os

if os.path.basename(os.getcwd()) != "vpd":
    print("WARNING: You must run this setup.py script from the parent directory of 'vpd' for paths to resolve correctly.", file=sys.stderr)
    sys.exit(1)

# TODO: For some reason include didn't work when we just typed in ./src/include on line 20, this seems to fix it though.
include_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'include')

setup(
    name='vpd',
    ext_modules=[
        CppExtension(
            '_cpp',
            ['./src/heat_flow.cpp', './src/heat_rff.cpp', './src/bindings.cpp'],
            include_dirs=[include_dir],
            extra_compile_args=['-O3', '-std=c++20']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
