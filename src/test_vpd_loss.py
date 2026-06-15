import argparse
import sys
from pathlib import Path
from config import HEAT_RFF_CONFIG

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vpd_macros import create_heat_random_fourier_features


def fake_pd(
    n_points: int,
    *,
    birth_hi: float = 1.0,
    persistence_hi: float = 2.0,
    seed: int = 0,
    requires_grad: bool = True,
) -> torch.Tensor:
    """Random diagram points (birth, death) with birth <= death."""
    gen = torch.Generator().manual_seed(seed)
    birth = torch.rand(n_points, generator=gen, dtype=torch.float64) * birth_hi
    persistence = torch.rand(n_points, generator=gen, dtype=torch.float64) * persistence_hi
    death = birth + persistence
    pd = torch.stack([birth, death], dim=-1)
    if requires_grad:
        pd = pd.requires_grad_(True)
    return pd


def run_case(name: str, rff, pd1: torch.Tensor, pd2: torch.Tensor) -> None:
    print(f"\n=== {name} ===")

    loss = rff.vpd_loss(pd1, pd2)
    vec = rff.vpd_loss_vector(pd1, pd2)
    print(f"vpd_loss: {loss.item():.6f}")
    print(f"sum of vpd_loss_vector squared: {vec.pow(2).sum().item():.6f}")
    print(f"vpd_loss_vector: {vec}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=12, help="points per diagram")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Keep RFF small so this finishes quickly.
    h0rff = create_heat_random_fourier_features(**HEAT_RFF_CONFIG["h0rff"])
    h1rff = create_heat_random_fourier_features(**HEAT_RFF_CONFIG["h1rff"])

    pd_a = fake_pd(args.points, seed=1)
    pd_b = fake_pd(args.points, seed=2)
    pd_same = fake_pd(args.points, seed=3)

    run_case("H1 different diagrams", h1rff, pd_a, pd_b)
    run_case("H1 identical diagrams", h1rff, pd_same, pd_same.clone())

    pd_h0_a = fake_pd(args.points, birth_hi=0.5, persistence_hi=1.5, seed=4)
    pd_h0_b = fake_pd(args.points, birth_hi=0.5, persistence_hi=1.5, seed=5)
    run_case("H0 different diagrams", h0rff, pd_h0_a, pd_h0_b)


if __name__ == "__main__":
    main()
