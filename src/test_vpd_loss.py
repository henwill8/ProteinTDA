import argparse
import sys
from pathlib import Path
from config import HEAT_RFF_CONFIG

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vpd_macros import create_heat_random_fourier_features


def run_case(name: str, rff, pd1: torch.Tensor, pd2: torch.Tensor) -> None:
    print(f"\n=== {name} ===")

    loss = rff.vpd_loss(pd1, pd2)
    vec = rff.vpd_loss_vector(pd1, pd2)
    print(f"vpd_loss: {loss.item():.6f}")
    print(f"sum of vpd_loss_vector squared: {vec.pow(2).sum().item():.6f}")
    print(f"vpd_loss_vector: {vec}")
    print(f"pd_difference: {rff.get_vpd(pd1) - rff.get_vpd(pd2)}")


def main() -> None:
    h0rff = create_heat_random_fourier_features(**HEAT_RFF_CONFIG["h0rff"])
    h1rff = create_heat_random_fourier_features(**HEAT_RFF_CONFIG["h1rff"])

    pd_a = torch.tensor([[0.0, 0.25], [1.0, 1.25], [1.0, 1.25], [0, 1.5]], dtype=torch.float64)
    pd_b = torch.tensor([[0.0, 1.0], [0.0, 1.5]], dtype=torch.float64)
    pd_same = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)

    run_case("H1 different diagrams", h1rff, pd_a, pd_b)
    run_case("H1 identical diagrams", h1rff, pd_same, pd_same.clone())

    run_case("H0 different diagrams", h0rff, pd_a, pd_b)
    run_case("H0 identical diagrams", h0rff, pd_same, pd_same.clone())


if __name__ == "__main__":
    main()
