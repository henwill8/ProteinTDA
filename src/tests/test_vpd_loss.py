import sys
import time

import torch

from proteintda.config import HEAT_RFF_CONFIG
from proteintda.tda.vpd_kernels import create_heat_random_fourier_features


def run_case(name: str, rff, pd1: torch.Tensor, pd2: torch.Tensor) -> None:
    print(f"\n=== {name} ===")

    loss = rff.vpd_loss(pd1, pd2)
    print(f"vpd_loss: {loss.item():.6f}")


def main() -> None:
    timer = time.time()
    print("Creating heat kernels...")
    h0rff = create_heat_random_fourier_features(**HEAT_RFF_CONFIG["h0rff"])
    print(f"Time taken to create heat kernels: {time.time() - timer:.2f} seconds")

    print("Running cases...")
    pd_a = torch.tensor([[0.0, 0.5], [0, 1.5], [0, 1.6]], dtype=torch.float64)
    pd_b = torch.tensor([[0.0, 1.0], [0.0, 1.5]], dtype=torch.float64)
    pd_same = torch.tensor([[0.0, 1.0], [1.0, 1.5]], dtype=torch.float64)

    run_case("H0 different diagrams", h0rff, pd_a, pd_b)
    run_case("H0 identical diagrams", h0rff, pd_same, pd_same.clone())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
