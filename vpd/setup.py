from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

import os

include_dir = os.path.join("src", "include")
sources = [
    os.path.join("src", "heat_flow.cpp"),
    os.path.join("src", "heat_rff.cpp"),
    os.path.join("src", "bindings.cpp"),
]

if os.name == "nt":
    extra_compile_args = ["/O2", "/std:c++20"]
else:
    extra_compile_args = ["-O3", "-std=c++20"]

setup(
    name="vpd",
    version="0.1.0",
    packages=["vpd"],
    package_dir={"vpd": "."},
    ext_modules=[
        CppExtension(
            name="vpd._cpp",
            sources=sources,
            include_dirs=[include_dir],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
