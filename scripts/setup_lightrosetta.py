"""Fetch LightRoseTTA training code into third_party/ and install its runtime deps."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
TRAINING_DIR = THIRD_PARTY / "LightRoseTTA_training"
REPO_URL = "https://github.com/psp3dcg/LightRoseTTA.git"
TRAINING_ZIP_URL = (
    "https://github.com/psp3dcg/LightRoseTTA/raw/master/LightRoseTTA_training_code.zip"
)

# Runtime packages needed to import/train LightRoseTTA (from their env.yml + model imports).
# Not the full frozen workstation env (Jupyter/TensorBoard/etc.).
# dgl is installed separately: PyPI's wheel is CPU-only and breaks CUDA graphs.
_RUNTIME_PIP = [
    "torch-geometric",
    "performer-pytorch",
    "axial-positional-embedding",
    "local-attention==1.9.14",  # newer pulls hyper-connections needing torch>=2.5
    "einops",
    "biopython",
    "fire",
    "yacs",
    "networkx",
    "dm-haiku",
    "jmp",
    "termcolor",
    "tabulate",
    "pygtrie",
    "googledrivedownloader",
    "pandas",  # required by dgl.graphbolt
    "pydantic",  # required by dgl.graphbolt
]


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install torch before running this script.") from exc
    return torch


def _torch_minor(torch) -> tuple[int, int]:
    major, minor, *_ = torch.__version__.split("+", 1)[0].split(".")
    return int(major), int(minor)


def _install_dgl(torch) -> None:
    """Force a CUDA-matched dgl wheel when torch was built with CUDA."""
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        "dgl",
    ]
    if not torch.version.cuda:
        print("Installing DGL (CPU / PyPI)...")
        subprocess.run(cmd, check=True)
        return

    major, minor = _torch_minor(torch)
    cu = "".join(torch.version.cuda.split(".")[:2])
    for url in (
        f"https://data.dgl.ai/wheels/torch-{major}.{minor}/cu{cu}/repo.html",
        f"https://data.dgl.ai/wheels/cu{cu}/repo.html",
    ):
        print(f"Installing DGL from {url}...")
        try:
            subprocess.run([*cmd, "-f", url], check=True)
            return
        except subprocess.CalledProcessError:
            continue
    raise SystemExit(f"Failed to install CUDA DGL for torch {torch.__version__} / CUDA {torch.version.cuda}")


def _install_python_deps() -> None:
    torch = _require_torch()
    _install_dgl(torch)
    print(f"Installing LightRoseTTA runtime deps ({len(_RUNTIME_PIP)} packages)...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade-strategy",
            "only-if-needed",
            *_RUNTIME_PIP,
        ],
        check=True,
    )
    # DGL imports torchdata.datapipes (removed after 0.9).
    pin = "torchdata==0.8.0" if _torch_minor(torch) == (2, 3) else "torchdata==0.9.0"
    print(f"Pinning {pin} for DGL datapipes...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            pin,
        ],
        check=True,
    )


def _patch_fcntl(cache_file: Path) -> None:
    text = cache_file.read_text(encoding="utf-8")
    if "msvcrt" in text:
        return
    text = text.replace(
        "import fcntl\n",
        "try:\n"
        "    import fcntl\n"
        "except ImportError:  # Windows\n"
        "    fcntl = None\n"
        "    import msvcrt\n",
        1,
    )
    text = text.replace(
        "        fcntl.lockf(self.handle, fcntl.LOCK_EX)\n",
        "        if fcntl is not None:\n"
        "            fcntl.lockf(self.handle, fcntl.LOCK_EX)\n"
        "        else:\n"
        "            msvcrt.locking(self.handle.fileno(), msvcrt.LK_LOCK, 1)\n",
        1,
    )
    text = text.replace(
        "        fcntl.lockf(self.handle, fcntl.LOCK_UN)\n",
        "        if fcntl is not None:\n"
        "            fcntl.lockf(self.handle, fcntl.LOCK_UN)\n"
        "        else:\n"
        "            try:\n"
        "                self.handle.seek(0)\n"
        "                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)\n"
        "            except OSError:\n"
        "                pass\n",
        1,
    )
    cache_file.write_text(text, encoding="utf-8")


def _patch_sym_cnn(sym_cnn: Path) -> None:
    text = sym_cnn.read_text(encoding="utf-8")
    text = text.replace(
        "self.ini_kernel = nn.Parameter(ini_kernel.cuda()) ",
        "self.ini_kernel = nn.Parameter(ini_kernel)",
    )
    text = text.replace(
        "self.ini_bias = nn.Parameter(ini_bias.cuda()) #(B, L, L, d)",
        "self.ini_bias = nn.Parameter(ini_bias)  # (B, L, L, d)",
    )
    sym_cnn.write_text(text, encoding="utf-8")


def _strip_empty_cache(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    updated = text.replace("        torch.cuda.empty_cache()\n", "")
    updated = updated.replace("    torch.cuda.empty_cache()\n", "")
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def _apply_local_patches() -> None:
    cache_file = (
        TRAINING_DIR
        / "utils"
        / "equivariant_attention"
        / "from_se3cnn"
        / "cache_file.py"
    )
    sym_cnn = TRAINING_DIR / "model" / "sym_cnn.py"
    if cache_file.is_file():
        _patch_fcntl(cache_file)
    if sym_cnn.is_file():
        _patch_sym_cnn(sym_cnn)

    for rel in (
        "model/LightRoseTTA.py",
        "model/build_graph.py",
        "model/parse_msa_info.py",
        "model/refine_net.py",
    ):
        path = TRAINING_DIR / rel
        if path.is_file():
            _strip_empty_cache(path)


def main() -> int:
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    clone_dir = THIRD_PARTY / "LightRoseTTA"
    if not clone_dir.is_dir():
        print(f"Cloning {REPO_URL} -> {clone_dir}")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)],
            check=True,
        )
    else:
        print(f"Already cloned: {clone_dir}")

    if TRAINING_DIR.is_dir() and (TRAINING_DIR / "model" / "LightRoseTTA.py").is_file():
        print(f"Training code already present: {TRAINING_DIR}")
    else:
        zip_path = THIRD_PARTY / "LightRoseTTA_training_code.zip"
        if not zip_path.is_file():
            src_zip = clone_dir / "LightRoseTTA_training_code.zip"
            if src_zip.is_file():
                shutil.copy2(src_zip, zip_path)
            else:
                print(f"Downloading {TRAINING_ZIP_URL}")
                urlretrieve(TRAINING_ZIP_URL, zip_path)
        print(f"Unpacking {zip_path} -> {TRAINING_DIR}")
        if TRAINING_DIR.exists():
            shutil.rmtree(TRAINING_DIR)
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(TRAINING_DIR)

    _apply_local_patches()
    _install_python_deps()
    print("Done. third_party/ is gitignored; re-run this script after a fresh clone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
