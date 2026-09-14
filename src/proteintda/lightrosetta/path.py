"""Add vendored LightRoseTTA training code to sys.path."""

import sys
from pathlib import Path

_LIGHTROSETTA_ROOT = (
    Path(__file__).resolve().parents[3] / "third_party" / "LightRoseTTA_training"
)


def ensure_lightrosetta_on_path() -> Path:
    root = _LIGHTROSETTA_ROOT.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"LightRoseTTA training code not found at {root}. "
            "Run: python scripts/setup_lightrosetta.py"
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
