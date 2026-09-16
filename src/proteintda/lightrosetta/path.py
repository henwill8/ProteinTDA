"""Add vendored LightRoseTTA training code to sys.path."""

import sys
from pathlib import Path

_LIGHTROSETTA_ROOT = (
    Path(__file__).resolve().parents[3] / "third_party" / "LightRoseTTA_training"
)

_GRAPHBOLT_PATCHED = False


def ensure_dgl_graphbolt_compat() -> None:
    """Ignore GraphBolt C++ load failures; LightRoseTTA only needs dgl.graph().

    PyPI dgl 2.1.0's graphbolt libs often mismatch newer torch. Must run before
    ``import dgl``.
    """
    global _GRAPHBOLT_PATCHED
    if _GRAPHBOLT_PATCHED:
        return
    try:
        import torch
    except ImportError:
        return

    real_load = torch.classes.load_library

    def load_library(path: str):
        if "graphbolt_pytorch" in path:
            try:
                return real_load(path)
            except (OSError, RuntimeError, ImportError):
                return None
        return real_load(path)

    torch.classes.load_library = load_library
    _GRAPHBOLT_PATCHED = True


def ensure_lightrosetta_on_path() -> Path:
    root = _LIGHTROSETTA_ROOT.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"LightRoseTTA training code not found at {root}. "
            "Run: python scripts/setup_lightrosetta.py"
        )
    ensure_dgl_graphbolt_compat()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
