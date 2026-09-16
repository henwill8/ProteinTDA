"""Add vendored LightRoseTTA training code to sys.path."""

import sys
from pathlib import Path

_LIGHTROSETTA_ROOT = (
    Path(__file__).resolve().parents[3] / "third_party" / "LightRoseTTA_training"
)


def ensure_dgl_graphbolt_compat() -> None:
    """Symlink DGL graphbolt .so when torch patch version has no matching build.

    PyPI dgl 2.1.0 ships graphbolt libs through torch 2.2.1 only. Newer torch
    still works for LightRoseTTA if we point at the newest available lib.
    Must run before ``import dgl``.
    """
    try:
        import importlib.util

        import torch
    except ImportError:
        return

    spec = importlib.util.find_spec("dgl")
    if spec is None or not spec.origin:
        return
    gb_dir = Path(spec.origin).resolve().parent / "graphbolt"
    if not gb_dir.is_dir():
        return

    vers = torch.__version__.split("+", 1)[0]
    if sys.platform.startswith("linux"):
        pattern, needed_name = "libgraphbolt_pytorch_*.so", f"libgraphbolt_pytorch_{vers}.so"
    elif sys.platform.startswith("darwin"):
        pattern, needed_name = (
            "libgraphbolt_pytorch_*.dylib",
            f"libgraphbolt_pytorch_{vers}.dylib",
        )
    elif sys.platform.startswith("win"):
        pattern, needed_name = "graphbolt_pytorch_*.dll", f"graphbolt_pytorch_{vers}.dll"
    else:
        return

    needed = gb_dir / needed_name
    if needed.exists():
        return
    available = sorted(gb_dir.glob(pattern))
    if not available:
        return
    src = available[-1]
    try:
        needed.symlink_to(src.name)
    except OSError:
        import shutil

        shutil.copy2(src, needed)


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
