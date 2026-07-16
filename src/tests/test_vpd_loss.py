import sys
import time

import torch

import math
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import spearmanr

from proteintda.config import HEAT_RFF_CONFIG
from proteintda.tda.vpd_kernels import create_heat_random_fourier_features

def get_arrays(rff): 
    lam = np.asarray(rff.weights, dtype=float).ravel()        
    th  = np.asarray(rff.thetas, dtype=float)                 
    return lam, th

def convert_for_weight(peak, r):
    t = 1 / (peak * (r - 1)) * math.log(r)
    return t, r * t

def run_case(name: str, rff, pd1: torch.Tensor, pd2: torch.Tensor) -> None:
    print(f"\n=== {name} ===")

    loss = rff.vpd_loss(pd1, pd2)
    print(f"vpd_loss: {loss.item():.6f}")

def make_histogram(weights, bins): 
    lambdas = np.array(weights, dtype=float)
    plt.hist(lambdas, bins=bins, edgecolor="black", color="skyblue")
    plt.savefig("hist.png")

def make_gammas(dim, support_sizes, n_reps, max_mult, device):
    rows = []
    k_labels = []
    for k in support_sizes:
        k = min((int(k), dim))
        batch = torch.zeros(n_reps, dim, dtype=torch.long, device=device)
        keys = torch.rand(n_reps, dim, device=device)
        coords = keys.argsort(dim=1)[:, :k]
        mags  = torch.randint(1, max_mult + 1, (n_reps, k))          
        signs = torch.where(torch.rand(n_reps, k) < 0.5, -1, 1)     
        batch.scatter_(1, coords, (signs * mags).to(torch.long))
        rows.append(batch)
        k_labels.append(torch.full((n_reps,), k))
    return rows, k_labels

def sweep(gammas, labels, lam, th, peaks, r_values, dim, reff_floor=0.05):
    G   = np.asarray(gammas, dtype=float)
    lam = np.asarray(lam, dtype=float).ravel()
    R   = lam.shape[0]
    th  = np.asarray(th, dtype=float).ravel()
    assert th.size == R * dim, f"th has {th.size}, expected R*dim={R*dim}"
    th  = th.reshape(R, dim)                     

    C = np.cos(G @ th.T)                          

    A, B = len(peaks), len(r_values)

    T = np.empty((A, B, 2))
    for a, l in enumerate(peaks):                  
        for b, r in enumerate(r_values):
            T[a, b] = convert_for_weight(l, r)

    taus = T.reshape(-1)                           
    W    = np.exp(-np.outer(taus, lam))           
    Khat = ((W @ C.T) / R).reshape(A, B, 2, -1)  

    kband = Khat[..., 0, :] - Khat[..., 1, :]   
    w     = np.exp(-T[..., 0, None] * lam) \
          - np.exp(-T[..., 1, None] * lam)     
    zband = w.mean(axis=2)                    
    loss  = 2.0 * (zband[..., None] - kband) 

    rho = np.full((A, B), np.nan)
    for a in range(A):
        for b in range(B):
            li = loss[a, b]
            if np.ptp(li) > 0:              
                rho[a, b] = spearmanr(li, labels).correlation

    return dict(rho=rho,  zband=zband, 
                peaks=np.asarray(peaks), r_values=np.asarray(r_values), taus=taus)

def main() -> None:
    timer = time.time()
    print("Creating heat kernels...")
    h0rff = create_heat_random_fourier_features(**HEAT_RFF_CONFIG["h0rff"])
    print(f"Time taken to create heat kernels: {time.time() - timer:.2f} seconds")

    lam, th = get_arrays(h0rff)
    make_histogram(lam, 30)

    lo, mid, hi = np.quantile(lam, [0.10, 0.50, 0.90])

    peaks = np.linspace(lo, hi, 13)
    r_values    = np.array([1.2, 1.5, 2.0, 3.0, 5.0, 8.0])

    dim = HEAT_RFF_CONFIG["h0rff"]["resolution"] * HEAT_RFF_CONFIG["h0rff"]["axis_dim"] - 1
    print(f"Dim: {dim}")

    gammas, labels = make_gammas(dim, support_sizes=[1, 2, 4, 8, 16, 32, 64], n_reps=32, max_mult=5, device="cpu")
    gammas = np.asarray(torch.cat(gammas).cpu())
    labels = np.asarray(torch.cat(labels).cpu())
    
    out = sweep(gammas, labels, lam, th, peaks, r_values, dim, reff_floor=0.05)
    rm = out["rho"]
    if np.all(np.isnan(rm)):
        print("fuck")
    else: 
        ia, ib = np.unravel_index(np.nanargmax(rm), rm.shape)
        print(f"best: lambda*={out['peaks'][ia]:.4g} r={out['r_values'][ib]:.3g} "
              f"rho={rm[ia,ib]:.3f} Z_band={out['zband'][ia,ib]:.3g} ")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
