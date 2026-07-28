import time
import numpy as np
import torch
import sys

from tests.mlp_test import PointMLP, _train_point_cloud
from tests.test_utils import _resolve_device, _scalar, load_proteins, protein_positions, save_results

from proteintda.config import LOSS_CONFIG, HEAT_RFF_CONFIG
from proteintda.minifold.pipeline import _current_lr, build_loss_fn, build_lr_scheduler
from proteintda.minifold.loss import _distance_matrix
from proteintda.tda.persistence import pd_from_graph
from proteintda.tda.vpd_kernels import create_heat_random_fourier_features

SCALE_SWEEP = [0.25, 0.67, 1.0, 1.3, 1.7]

MLP_STEPS = 500
MLP_LR = 0.0001
HIDDEN_DIM = 256
LOG_EVERY = 1
TDA_WARMUP_STEPS = 0
TDA_RAMP_STEPS = 0


dist_rmses=[]
dist_w1s=[]
rel_rmses=[]

def _make_problem(device, protein):
    target_pts = protein_positions([protein])[0].to(device).float()
    target_adj = _distance_matrix(target_pts).detach()
    return target_pts, target_adj

def run_mlp(device, loss_fn, proteins, scale, h0rff, h1rff, h2rff, visualize=False, seed=42):
    results = []
    for pts, adj in zip(*[_make_problem(device, p) for p in proteins])
        n = pts.shape[0]
        model = PointMLP(n, hidden_dim=HIDDEN_DIM).to(device)
        model_input = (pts.flatten() * scale).unsqueeze(0) 
        optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)
        scheduler = build_lr_scheduler(optimizer)
        results.append(_train_point_cloud(
            "mlp",
            adj,
            pts,
            optimizer,
            scheduler,
            get_pred_pts=lambda m=model, x=model_input: m(x),
            loss_fn=loss_fn,
            h0rff=h0rff,
            h1rff=h1rff,
            h2rff=h2rff,
            visualize=visualize
        ))
    return results

def main(write):
    device = _resolve_device()
    loss_fn = build_loss_fn()
    if not loss_fn.tda_enabled:
        raise ValueError(
            "No TDA loss terms enabled. Enable wasserstein and/or vpd in LOSS_CONFIG."
        )
    timer = time.time()
    print("Creating heat kernels...")
    terms = LOSS_CONFIG.tda.terms
    if terms.vpd_h0.enabled:
        h0rff = create_heat_random_fourier_features(**HEAT_RFF_CONFIG["h0rff"])
    else:
        h0rff = None
    if terms.vpd_h1.enabled:
        h1rff = create_heat_random_fourier_features(**HEAT_RFF_CONFIG["h1rff"])
    else:
        h1rff = None
    if terms.vpd_h2.enabled:
        h2rff = create_heat_random_fourier_features(**HEAT_RFF_CONFIG["h2rff"])
    else:
        h2rff = None
    print(f"Time taken to create heat kernels: {time.time() - timer:.2f} seconds")

    proteins = load_proteins(50) 

    results = {}

    for scale in SCALE_SWEEP:
        results[f"Scale {scale}"] = run_mlp(device, loss_fn, proteins, scale, h0rff, h1rff, h2rff, False, 42)

    if write:
        save_results(results)
    else:
        print_results(results)
    return 0

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        if sys.argv[1] == "--write":
            sys.exit(main(write=True))
    sys.exit(main(write=False))
