"""Build the patched MiniFold wheel used by ProteinTDA."""

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WHEELS_DIR = REPO_ROOT / "wheels"
UPSTREAM = "https://github.com/jwohlwend/minifold.git"
WHEEL_NAME = "minifold-0.1.0-py3-none-any.whl"

PYPROJECT = """\
[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "minifold"
version = "0.1.0"
requires-python = ">=3.9"
description = "MiniFold"
readme = "README.md"
dependencies = [
    "numpy",
    "scipy",
    "pytorch_lightning",
    "torch>=2.3.0",
    "fair-esm",
    "biopython==1.81",
    "edit_distance",
    "pandas",
    "modelcif",
    "dm-tree",
    "psutil",
    "click",
    "ml_collections",
    "einops",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["minifold*"]

[tool.setuptools.package-data]
minifold = ["utils/stereo_chemical_props.txt"]
"""


def _optional_triton_imports(text: str) -> str:
    return text.replace(
        "import triton\nimport triton.language as tl",
        "try:\n    import triton\n    import triton.language as tl\nexcept ImportError:\n    triton = None\n    tl = None",
        1,
    )


def _wrap_block(text: str, start_marker: str, end_marker: str, wrapper: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    block = text[start:end]
    indented = "\n".join(f"    {line}" if line else "" for line in block.splitlines())
    return text[:start] + f"{wrapper}\n{indented}\n" + text[end:]


def patch_kernel_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = _optional_triton_imports(text)
    kernel_name = "gating_kernel" if path.name == "gating.py" else "mlp_kernel"
    text = _wrap_block(
        text,
        "@triton.autotune",
        f"\ndef {kernel_name}",
        "if triton is not None:",
    )
    text = text.replace(
        'if X.device.type == "cuda":',
        'if X.device.type == "cuda" and triton is not None:',
        1,
    )
    if "@triton.testing.perf_report" in text:
        text = _wrap_block(
            text,
            "@triton.testing.perf_report",
            "\ndef clear_gradients",
            "if triton is not None:",
        )
        text = text.replace(
            "\ndef clear_gradients",
            "\nelse:\n    benchmark = None\n\ndef clear_gradients",
            1,
        )
        text = text.replace(
            "        benchmark.run(print_data=True, show_plots=False)",
            "        benchmark.run(print_data=True, show_plots=False) if benchmark is not None else None",
        )
        text = text.replace(
            "        benchmark.run(print_data=True, show_plots=False, device=\"cuda\")",
            "        benchmark.run(print_data=True, show_plots=False, device=\"cuda\") if benchmark is not None else None",
        )
        text = text.replace(
            "        benchmark.run(print_data=True, show_plots=False, device=\"mps\")",
            "        benchmark.run(print_data=True, show_plots=False, device=\"mps\") if benchmark is not None else None",
        )
        text = text.replace(
            "        benchmark.run(print_data=True, show_plots=False, device=\"cpu\")",
            "        benchmark.run(print_data=True, show_plots=False, device=\"cpu\") if benchmark is not None else None",
        )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "minifold"
        subprocess.run(["git", "clone", "--depth", "1", UPSTREAM, str(src)], check=True)
        (src / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
        patch_kernel_file(src / "minifold" / "model" / "kernels" / "gating.py")
        patch_kernel_file(src / "minifold" / "model" / "kernels" / "mlp.py")
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", str(src), "-w", str(WHEELS_DIR), "--no-deps"],
            check=True,
        )
    built = WHEELS_DIR / WHEEL_NAME
    if not built.exists():
        wheels = list(WHEELS_DIR.glob("minifold-*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"Expected one wheel in {WHEELS_DIR}, found: {wheels}")
        wheels[0].replace(built)
    print(f"Built {built}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
