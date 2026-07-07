#!/usr/bin/env python3
"""Install project requirements (build deps, torch, then requirements.txt)."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"
INSTALL_TORCH = Path(__file__).resolve().parent / "install_torch.py"

BUILD_DEPS = ("numpy", "pybind11")


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def main() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *BUILD_DEPS])

    if has_torch():
        print("torch already installed, skipping install_torch.py")
    else:
        subprocess.check_call([sys.executable, str(INSTALL_TORCH)])

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS),
            "--no-build-isolation",
        ]
    )


if __name__ == "__main__":
    main()
