import math
import matplotlib.pyplot as plt
import numpy as np

import torch

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

