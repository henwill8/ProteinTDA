"""Add vendored LightRoseTTA training code to sys.path."""

import importlib.util
import shutil
import sys
from pathlib import Path

_LIGHTROSETTA_ROOT = (
    Path(__file__).resolve().parents[3] / "third_party" / "LightRoseTTA_training"
)

_GRAPHBOLT_PATCHED = False


def _graphbolt_dir() -> Path | None:
    spec = importlib.util.find_spec("dgl")
    if spec is None or not spec.origin:
        return None
    gb_dir = Path(spec.origin).resolve().parent / "graphbolt"
    return gb_dir if gb_dir.is_dir() else None


def _ensure_graphbolt_lib(torch) -> None:
    """Make the torch-versioned graphbolt lib path exist so dgl import doesn't FileNotFound."""
    gb_dir = _graphbolt_dir()
    if gb_dir is None:
        return

    vers = torch.__version__.split("+", 1)[0]
    if sys.platform.startswith("linux"):
        pattern, needed = "libgraphbolt_pytorch_*.so", f"libgraphbolt_pytorch_{vers}.so"
    elif sys.platform.startswith("darwin"):
        pattern, needed = (
            "libgraphbolt_pytorch_*.dylib",
            f"libgraphbolt_pytorch_{vers}.dylib",
        )
    elif sys.platform.startswith("win"):
        pattern, needed = "graphbolt_pytorch_*.dll", f"graphbolt_pytorch_{vers}.dll"
    else:
        return

    target = gb_dir / needed
    if target.exists():
        return
    available = sorted(gb_dir.glob(pattern))
    if available:
        src = available[-1]
        try:
            target.symlink_to(src.name)
        except OSError:
            shutil.copy2(src, target)
        return
    # Exists-check must pass; patched load_library below will no-op on this stub.
    target.write_bytes(b"")


def ensure_dgl_graphbolt_compat() -> None:
    """Allow dgl import when graphbolt C++ lib is missing/mismatched.

    LightRoseTTA only needs dgl.graph(). Must run before ``import dgl``.
    """
    global _GRAPHBOLT_PATCHED
    if _GRAPHBOLT_PATCHED:
        return
    try:
        import torch
    except ImportError:
        return

    _ensure_graphbolt_lib(torch)

    real_load = torch.classes.load_library

    def load_library(path: str):
        if "graphbolt_pytorch" in path:
            try:
                return real_load(path)
            except (OSError, RuntimeError, ImportError):
                return None
        return real_load(path)

    torch.classes.load_library = load_library  # type: ignore[method-assign]
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
