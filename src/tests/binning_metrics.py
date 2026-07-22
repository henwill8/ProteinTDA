import time

import torch

from proteintda.config import HEAT_RFF_CONFIG
from proteintda.tda.vpd_kernels import create_heat_random_fourier_features

def compute_metrics(pd : torch.Tensor, rff):
    vpd = rff.get_vpd(pd)
    torch.set_printoptions(profile="full")
    print(vpd)
    out = {}

    out["total_nonzero"] = torch.count_nonzero(vpd)
    out["total_zero"] = (vpd == 0).sum().item()

    out["min_nonzero"] = torch.min(vpd[vpd != 0])
    out["max_nonzero"] = torch.max(vpd[vpd != 0])
    out["mean_nonzero"] = vpd[vpd != 0].mean()
    out["std_nonzero"] = vpd[vpd != 0].std()

    return out

def print_metrics(out, n):
    print(f"\n=== VPD Binning Metrics H{n} ===\n")
    for k, v in out.items():
        print(f"\n VPD {k}: {v}")

def main():
    timer = time.time()
    print("Creating heat kernels...")
    h1rff = create_heat_random_fourier_features(**HEAT_RFF_CONFIG["h1rff"])
    print(f"Time taken to create heat kernels: {time.time() - timer:.2f} seconds")
    pd = torch.tensor([[1, 2], [2, 3], [3, 4], [3,4], [3, 4]], dtype=torch.float64)
    out = compute_metrics(pd, h1rff)
    print_metrics(out)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
