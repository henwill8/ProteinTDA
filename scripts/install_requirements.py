#!/usr/bin/env python3
"""Install project requirements (build deps, torch, then requirements.txt)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"
INSTALL_TORCH = Path(__file__).resolve().parent / "install_torch.py"

# Needed by sdist packages when using --no-build-isolation (e.g. tmtools, openfold).
BUILD_DEPS = ("numpy", "pybind11")


def main() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *BUILD_DEPS])
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
