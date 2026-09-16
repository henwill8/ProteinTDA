"""Add vendored LightRoseTTA training code to sys.path."""

import shutil
import sys
from pathlib import Path

_LIGHTROSETTA_ROOT = (
    Path(__file__).resolve().parents[3] / "third_party" / "LightRoseTTA_training"
)

_GRAPHBOLT_PATCHED = False


def _graphbolt_lib_names() -> tuple[str, str]:
    if sys.platform.startswith("linux"):
        return "libgraphbolt_pytorch_*.so", "libgraphbolt_pytorch_{vers}.so"
    if sys.platform.startswith("darwin"):
        return "libgraphbolt_pytorch_*.dylib", "libgraphbolt_pytorch_{vers}.dylib"
    if sys.platform.startswith("win"):
        return "graphbolt_pytorch_*.dll", "graphbolt_pytorch_{vers}.dll"
    return "", ""


def _graphbolt_lib_loads(torch, path: Path) -> bool:
    try:
        torch.classes.load_library(str(path))
        return True
    except Exception:
        return False


def _link_graphbolt_lib(gb_dir: Path, needed: Path, src: Path) -> None:
    if needed.is_symlink() or needed.exists():
        needed.unlink()
    try:
        needed.symlink_to(src.name)
    except OSError:
        shutil.copy2(src, needed)


def _patch_graphbolt_loader(torch) -> None:
    """Skip graphbolt C++ load failures; LightRoseTTA only uses dgl.graph()."""
    global _GRAPHBOLT_PATCHED
    if _GRAPHBOLT_PATCHED:
        return
    real_load = torch.classes.load_library

    def load_library(path: str):
        if "graphbolt_pytorch" in path:
            try:
                return real_load(path)
            except Exception:
                return None
        return real_load(path)

    torch.classes.load_library = load_library  # type: ignore[method-assign]
    _GRAPHBOLT_PATCHED = True


def ensure_dgl_graphbolt_compat() -> None:
    """Pick a loadable graphbolt lib for this torch, or skip loading it.

    PyPI dgl 2.1.0 ships graphbolt libs only through torch 2.2.1. Must run
    before ``import dgl``.
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

    pattern, name_fmt = _graphbolt_lib_names()
    if not pattern:
        return

    vers = torch.__version__.split("+", 1)[0]
    needed = gb_dir / name_fmt.format(vers=vers)
    if needed.exists() and _graphbolt_lib_loads(torch, needed):
        return
    if needed.is_symlink() or needed.exists():
        needed.unlink()

    for candidate in reversed(sorted(gb_dir.glob(pattern))):
        if _graphbolt_lib_loads(torch, candidate):
            _link_graphbolt_lib(gb_dir, needed, candidate)
            return

    _patch_graphbolt_loader(torch)


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
