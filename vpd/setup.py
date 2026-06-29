from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

import os

USE_OPENMP = True

cwd = os.getcwd()
include_dir = os.path.join(cwd, "src", "include")
sampling_dir = os.path.join(cwd, "src", "sampling")

sources = [
    os.path.join("src", "straight_through.cpp"),
    os.path.join("src", "heat_kernel.cpp"),
    os.path.join("src", "sampling", "sampling_method.cpp"),
    os.path.join("src", "sampling", "rejection_sampling.cpp"),
    os.path.join("src", "sampling", "metropolis_hastings_sampling.cpp"),
    os.path.join("src", "vpd.cpp"),
    os.path.join("src", "bindings.cpp"),
]

if os.name == "nt":
    extra_compile_args = ["/O2", "/std:c++20"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-std=c++20"]
    extra_link_args = []

if USE_OPENMP:
    if os.name == "nt":
        extra_compile_args.append("/openmp")
    else:
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")

setup(
    name="vpd",
    version="0.1.0",
    packages=["vpd"],
    package_dir={"vpd": "."},
    ext_modules=[
        CppExtension(
            name="vpd._cpp",
            sources=sources,
            include_dirs=[include_dir, sampling_dir],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
