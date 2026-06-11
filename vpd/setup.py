# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='vpd', # Name of your Python package
    ext_modules=[
        CppExtension(
            'vpd._cpp', # Module name as imported in Python
            ['src/heat_flow.cpp']   # List of your C++ source files
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
