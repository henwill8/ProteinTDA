from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

import sys
import os

if os.path.basename(os.getcwd()) != "vpd":
    print("WARNING: You must run this setup.py script from the parent directory of 'vpd' for paths to resolve correctly.", file=sys.stderr)
    sys.exit(1)


setup(
    name='vpd',
    ext_modules=[
        CppExtension(
            'vpd._cpp',
            ['./src/heat_flow.cpp'],
            include_dirs=['./src/include'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
