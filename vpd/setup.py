from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

import os
import torch

USE_OPENMP = True
USE_CUDA = torch.cuda.is_available() and CUDA_HOME is not None

cwd = os.getcwd()
include_dir = os.path.join(cwd, "src", "include")
sampling_dir = os.path.join(cwd, "src", "sampling")
cuda_dir = os.path.join(cwd, "src", "cuda")

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

cuda_sources = [
    os.path.join("src", "cuda", "sampling_common.cu"),
    os.path.join("src", "cuda", "mala_sampling_cuda.cu"),
    os.path.join("src", "cuda", "metropolis_hastings_sampling_cuda.cu"),
    os.path.join("src", "cuda", "rejection_sampling_cuda.cu"),
]

include_dirs = [include_dir, sampling_dir]
if USE_CUDA:
    sources += cuda_sources
    include_dirs.append(cuda_dir)

abi_flag = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"

if os.name == "nt":
    cxx_args = ["/O2", "/std:c++20"]
    nvcc_args = ["-O3", "-std=c++20", "-Xcompiler", "/O2"]
    extra_link_args = []
else:
    cxx_args = ["-O3", "-std=c++20", f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}"]
    nvcc_args = [
        "-O3",
        "-std=c++20",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}",
        "--expt-relaxed-constexpr",
    ]
    extra_link_args = []

if USE_OPENMP:
    if os.name == "nt":
        cxx_args += ["/openmp", "-openmp:experimental"]
        nvcc_args += ["-Xcompiler", "/openmp"]
    else:
        cxx_args += ["-fopenmp", "-openmp:experimental"]
        nvcc_args += ["-Xcompiler", "-fopenmp"]
        extra_link_args.append("-fopenmp")

define_macros = []
if USE_CUDA:
    define_macros.append(("VPD_WITH_CUDA", None))
if USE_OPENMP:
    define_macros.append(("VPD_WITH_OPENMP", None))

ext_kwargs = dict(
    name="vpd._cpp",
    sources=sources,
    include_dirs=include_dirs,
    extra_link_args=extra_link_args,
    define_macros=define_macros,
)

if USE_CUDA:
    ext_module = CUDAExtension(
        extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
        **ext_kwargs,
    )
else:
    ext_module = CppExtension(
        extra_compile_args={"cxx": cxx_args},
        **ext_kwargs,
    )

setup(
    name="vpd",
    version="0.1.0",
    packages=["vpd"],
    package_dir={"vpd": "."},
    ext_modules=[ext_module],
    cmdclass={"build_ext": BuildExtension},
)