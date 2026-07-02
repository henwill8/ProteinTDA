#!/usr/bin/env python3
"""Install the correct PyTorch version for CUDA into the current environment."""

import glob
import os
import re
import shutil
import subprocess
import sys

CUDA_MAJOR_TO_WHEEL = {
    11: "cu118",
    12: "cu124",
    13: "cu130",
}
DEFAULT_GPU_WHEEL = "cu124"


def find_nvcc() -> str | None:
    if path := shutil.which("nvcc"):
        return path
    if sys.platform == "win32":
        pattern = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe"
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            return matches[0]
    return None


def nvcc_cuda_major() -> int | None:
    nvcc = find_nvcc()
    if not nvcc:
        return None
    output = subprocess.check_output([nvcc, "--version"], text=True)
    match = re.search(r"release (\d+)\.", output)
    return int(match.group(1)) if match else None


def has_nvidia_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    return subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def detect_wheel_tag() -> str:
    major = nvcc_cuda_major()
    if major is not None:
        return CUDA_MAJOR_TO_WHEEL.get(major, DEFAULT_GPU_WHEEL)
    if has_nvidia_gpu():
        return DEFAULT_GPU_WHEEL
    return "cpu"


def resolve_index_url() -> str:
    if url := os.environ.get("PYTORCH_INDEX_URL"):
        return url

    tag = os.environ.get("PYTORCH_CUDA") or detect_wheel_tag()
    if tag == "cpu":
        return "https://download.pytorch.org/whl/cpu"
    return f"https://download.pytorch.org/whl/{tag}"


def main() -> None:
    index_url = resolve_index_url()
    tag = index_url.rstrip("/").rsplit("/", 1)[-1]
    print(f"Installing torch ({tag}) from {index_url}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "torch", "--index-url", index_url]
    )


if __name__ == "__main__":
    main()
