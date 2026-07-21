import math
import matplotlib.pyplot as plt
import numpy as np

import torch

from proteintda.config import CONFIG_OF, LOSS_CONFIG, RUN_CONFIG

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

