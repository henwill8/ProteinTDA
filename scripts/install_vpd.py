"""Build/install vpd against the current torch, then restore setuptools<82."""

import subprocess
import sys
from pathlib import Path

# Keep in sync with root pyproject.toml (sidechainnet / pkg_resources).
_SETUPTOOLS_PIN = "setuptools>=77,<82"
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(ROOT / "vpd"),
            "--no-build-isolation",
            "--no-deps",
        ],
        check=True,
        cwd=ROOT,
    )
    # torch build deps can upgrade setuptools past sidechainnet's supported range.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", _SETUPTOOLS_PIN],
        check=True,
    )
    print("vpd installed; setuptools pinned for sidechainnet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
