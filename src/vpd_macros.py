import hashlib
import sys
import os
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vpd import _cpp
from config import HEAT_RFF_CONFIG

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "heat_rff"


def _heat_rff_cache_path(n, axis_dim, resolution, R, tau, mask, seed):
    key = repr((n, axis_dim, resolution, R, tau, mask, seed))
    digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    return _CACHE_DIR / f"{digest}.pt"


def create_heat_random_fourier_features(n, axis_dim, resolution, R=100, tau=1, mask=None, seed=42):
    cache_path = _heat_rff_cache_path(n, axis_dim, resolution, R, tau, mask, seed)
    if cache_path.is_file():
        cached = torch.load(cache_path, weights_only=False)
        return _cpp.Heat_RFF(
            n, axis_dim, resolution, R, tau, cached["thetas"], cached["weights"]
        )

    rff = _cpp.Heat_RFF(n, axis_dim, resolution, R, tau, mask, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "n": n,
            "axis_dim": axis_dim,
            "resolution": resolution,
            "R": R,
            "tau": tau,
            "mask": mask,
            "seed": seed,
            "thetas": rff.thetas,
            "weights": rff.weights,
        },
        cache_path,
    )
    return rff


if __name__ == "__main__":
    print("Python is searching in these folders:", sys.path)
    print(create_heat_random_fourier_features(**HEAT_RFF_CONFIG.h0rff))
    print(create_heat_random_fourier_features(**HEAT_RFF_CONFIG.h1rff))
