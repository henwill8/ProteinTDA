"""Fetch LightRoseTTA training code into third_party/"""

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

# Owned by ProteinTDA / would break the current env if taken from their frozen yml.
_SKIP_PIP_NAMES = {
    "torch",
    "torchvision",
    "torchaudio",
    "pytorch",
    "openmm",  # their pin is a different PyPI package, not OpenMM
    "simtk",
}

# Conda package name (prefix) → PyPI name for Python deps that are not under pip:.
_CONDA_TO_PIP = {
    "pyg": "torch-geometric",
    "dgl": "dgl",
    "dgl-cuda": "dgl",
    "yacs": "yacs",
    "networkx": "networkx",
    "googledrivedownloader": "googledrivedownloader",
}


def _package_name(req: str) -> str:
    name = req.strip()
    for sep in ("===", "==", ">=", "<=", "~=", "!=", ">", "<"):
        if sep in name:
            name = name.split(sep, 1)[0]
            break
    # Conda build strings: name=version=build
    if "=" in name:
        name = name.split("=", 1)[0]
    return name.strip().lower().replace("_", "-")


def _pip_reqs_from_env_yml(env_yml: Path) -> list[str]:
    """Read the pip: list from LightRoseTTA-env.yml (their published dep list)."""
    reqs: list[str] = []
    in_pip = False
    for raw in env_yml.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not in_pip:
            # Conda form is "  - pip:" not a bare "pip:" key.
            if stripped in {"pip:", "- pip:"} or stripped.lstrip("- ").strip() == "pip:":
                in_pip = True
            continue
        if stripped.startswith("- "):
            reqs.append(stripped[2:].strip())
            continue
        if stripped.startswith("prefix:") or (stripped and not line[:1].isspace()):
            break
    return reqs


def _conda_python_pkgs_from_env_yml(env_yml: Path) -> list[str]:
    """Map conda Python packages in the env file to PyPI names."""
    found: list[str] = []
    for raw in env_yml.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.lstrip("- ").strip() == "pip:":
            break
        if not stripped.startswith("- "):
            continue
        name = _package_name(stripped[2:].strip())
        for prefix, pip_name in _CONDA_TO_PIP.items():
            if name == prefix or name.startswith(prefix + "-"):
                if pip_name not in found:
                    found.append(pip_name)
                break
    return found


def _install_python_deps(clone_dir: Path) -> None:
    env_yml = clone_dir / "LightRoseTTA-env.yml"
    if not env_yml.is_file():
        raise FileNotFoundError(f"Missing {env_yml}; cannot install LightRoseTTA deps")

    # Drop pins so we resolve against the current Torch/Python instead of their 2021 freeze.
    reqs: list[str] = []
    seen: set[str] = set()
    for req in _pip_reqs_from_env_yml(env_yml) + _conda_python_pkgs_from_env_yml(env_yml):
        name = _package_name(req)
        if name in _SKIP_PIP_NAMES or name in seen:
            continue
        seen.add(name)
        reqs.append(name)

    # Current DGL imports torchdata.datapipes at import time.
    if "dgl" in seen and "torchdata" not in seen:
        reqs.append("torchdata")
        seen.add("torchdata")

    if not reqs:
        raise RuntimeError(f"Parsed zero packages from {env_yml}; check YAML format")

    print(f"Installing {len(reqs)} packages from {env_yml.name} (unpinned, conflicts skipped)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *reqs],
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
    print("Installing LightRoseTTA Python dependencies from LightRoseTTA-env.yml...")
    _install_python_deps(clone_dir)
    print("Done. third_party/ is gitignored; re-run this script after a fresh clone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
