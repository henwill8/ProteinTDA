from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

import os
import torch

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
    os.path.join("src", "sampling", "mala_sampling.cpp"),
    os.path.join("src", "vpd.cpp"),
    os.path.join("src", "bindings.cpp"),
]

abi_flag = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"

if os.name == "nt":
    extra_compile_args = ["/O2", "/std:c++20"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-std=c++20", f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}"]
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
