import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys

from numpy.random import sample
from sidechainnet import SCNProtein
import torch

from proteintda.config import CONFIG_OF, LOSS_CONFIG, RUN_CONFIG
from proteintda.utils.conversions import Atom37, atom_positions_from_atom37, atom_positions_from_sidechainnet, SideChainAtom
from proteintda.utils.dataset import load_dataset, sample_proteins
from proteintda.minifold.loss import _distance_matrix
from proteintda.tda.persistence import pd_from_graph

def convert_for_weight(peak, r):
    t = 1 / (peak * (r - 1)) * math.log(r)
    return t , r * t

def run_case(rff, pd1: torch.Tensor, pd2: torch.Tensor, name: str | None = None) -> None:
    if name is not None:
        print(f"\n=== {name} ===")

    loss = rff.vpd_loss(pd1, pd2)
    print(f"vpd_loss: {loss.item():.6f}")

def make_histogram(lambdas, bins):
    lambdas = np.array(lambdas, dtype=float)
    plt.hist(lambdas, bins=bins, edgecolor="black", color="skyblue")
    plt.savefig("out/hist.png")

def _resolve_device() -> torch.device:
    device = RUN_CONFIG.runtime.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)

def _to_numpy(diags, dim):
    if len(diags) < dim + 1:
        return np.empty((0, 2))
    arr = diags[dim]
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)

def _scalar(x):
    return x.item() if hasattr(x, "item") else float(x)

def load_proteins(max_proteins: int | None = None) -> list:
    dataset = load_dataset()
    rng  = np.random.default_rng(seed=42)
    return sample_proteins(dataset, max_proteins, rng)

def positions_from_scn(proteins: list[SCNProtein]) -> list[torch.Tensor]:
    return [atom_positions_from_sidechainnet(p, SideChainAtom.CB) for p in proteins]


def positions_from_atom37(proteins: list[torch.Tensor]) -> list[torch.Tensor]:
    return [atom_positions_from_atom37(p, None, Atom37.CB) for p in proteins]


def protein_positions(proteins: list[torch.Tensor] | list[SCNProtein]) -> list[torch.Tensor]:
    if not proteins:
        return []
    if isinstance(proteins[0], torch.Tensor):
        return positions_from_atom37(proteins)
    return positions_from_scn(proteins)


def protein_adj_matrices(proteins: list[torch.Tensor] | list[SCNProtein]) -> list[torch.Tensor]:
    return [_distance_matrix(pos) for pos in protein_positions(proteins)]

def protein_pds(proteins: list[torch.Tensor] | list[SCNProtein]):
    return[pd_from_graph(adj, **LOSS_CONFIG.pd) for adj in protein_adj_matrices(proteins)]

def print_results(results: dict[str, dict]):
    for test, t_results in results.items():
        print(f"\n\n========== Results for Test: {test} ==========")
        for k, v in t_results.items():
            print(f"\n{k}: {v}")

def save_results(results: dict[str, dict]):
    cwd = Path.cwd()
    tests_dir = cwd / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    for test, t_results in results.items():
        with open(tests_dir / test, "a") as f:
            for k, v in  t_results.items():
                f.write(f"\n{k}: {v}")
