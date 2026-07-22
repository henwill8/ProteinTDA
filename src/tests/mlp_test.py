import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tests.binning_metrics import compute_metrics, print_metrics
from tests.test_utils import _resolve_device, _scalar
from tests.plotting import _make_history_frame, show_history, _tda_terms_msg

from proteintda.config import CONFIG_OF, HEAT_RFF_CONFIG, LOSS_CONFIG, RUN_CONFIG
from proteintda.minifold.pipeline import _current_lr, build_loss_fn, build_lr_scheduler
from proteintda.minifold.loss import _distance_matrix
from proteintda.tda.persistence import pd_from_graph
from proteintda.tda.vpd_kernels import create_heat_random_fourier_features

N_POINTS = 100
POINT_SCALE = 4.0
MLP_STEPS = 500
MLP_LR = 0.0001
HIDDEN_DIM = 256

# Second eval: random sample from the full SidechainNet split (not the TM/length-selected set).
LOG_EVERY = 1
TDA_WARMUP_STEPS = 0
TDA_RAMP_STEPS = 0
CHECKPOINT_DIR = Path("logs/tda_loss")

_TDA_BREAKDOWN_KEYS = ("wasserstein_h0", "wasserstein_h1", "wasserstein_h2", "vpd_h0", "vpd_h1", "vpd_h2")

dist_rmses=[]
dist_w1s=[]
rel_rmses=[]

class PointMLP(nn.Module):
    def __init__(self, n_points, hidden_dim):
        super().__init__()
        self.n_points = n_points
        d = n_points * 3
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return (x + self.net(x)).view(self.n_points, 3)

def _make_problem(device, seed=42):
    torch.manual_seed(seed)
    target_pts = torch.randn(N_POINTS, 3, device=device) * POINT_SCALE
    target_adj = _distance_matrix(target_pts).detach()
    return target_pts, target_adj

def point_cloud_metrics(pred_pts, target_pts):
    pred = pred_pts.detach()
    tgt = target_pts.detach()
    n = pred.shape[0]

    iu = torch.triu_indices(n, n, offset=1, device=pred.device)
    pred_d = torch.sort(_distance_matrix(pred)[iu[0], iu[1]]).values
    tgt_d = torch.sort(_distance_matrix(tgt)[iu[0], iu[1]]).values

    dist_rmse = torch.sqrt(torch.mean((pred_d - tgt_d) ** 2)).item()
    dist_w1 = torch.mean(torch.abs(pred_d - tgt_d)).item()   
    rel_rmse = dist_rmse / (tgt_d.mean().item() + 1e-8)
    return {"dist_rmse": dist_rmse, "dist_w1": dist_w1, "rel_rmse": rel_rmse}

def _log_loss_config():
    parts = []
    if LOSS_CONFIG.tda.enabled:
        tda_terms = [
            name
            for name in _TDA_BREAKDOWN_KEYS
            if LOSS_CONFIG[name].enabled
        ]
        if tda_terms:
            parts.append(f"tda ({', '.join(tda_terms)})")
        if TDA_WARMUP_STEPS > 0 or TDA_RAMP_STEPS > 0:
            parts.append(f"warmup={TDA_WARMUP_STEPS} ramp={TDA_RAMP_STEPS}")
    if LOSS_CONFIG.distogram.enabled:
        parts.append(f"distogram (w={LOSS_CONFIG.distogram.weight})")
    if LOSS_CONFIG.structure.enabled:
        parts.append(f"structure (w={LOSS_CONFIG.structure.weight})")
    if parts:
        print(f"  losses: {', '.join(parts)}")

def _train_point_cloud(name, target_adj, target_pts, optimizer, scheduler, *, get_pred_pts, loss_fn, h0rff, h1rff, h2rff, visualize=False):
    """Overfit TDA loss on a fixed target point cloud (points / MLP modes)."""
    n_pts = target_pts.shape[0]
    print(f"[{name}] {n_pts} points, {MLP_STEPS} steps, lr={MLP_LR}")
    _log_loss_config()

    tda_loss_fn = loss_fn._tda if loss_fn.tda_enabled else None
    if tda_loss_fn is None or not tda_loss_fn._enabled:
        raise ValueError(
            "No TDA loss terms enabled. Enable wasserstein and/or vpd in LOSS_CONFIG."
        )

    target_diags = pd_from_graph(target_adj, **LOSS_CONFIG.pd)
    if h0rff is not None:
        h0metrics = compute_metrics(target_diags[0], h0rff)
        print_metrics(h0metrics, 0)
    if h1rff is not None:
        h1metrics = compute_metrics(target_diags[1], h1rff)
        print_metrics(h1metrics, 1)
    if h2rff is not None:
        h2metrics = compute_metrics(target_diags[2], h2rff)
        print_metrics(h2metrics, 2)
    tgt_counts = [len(d) for d in target_diags]
    hom_label = "/".join(f"H{d}" for d in range(LOSS_CONFIG.pd.hom_dim))
    print(f"target diagram sizes {hom_label}={tgt_counts}")

    history = []
    target_pts_np = target_pts.detach().cpu().numpy()

    for step in range(MLP_STEPS + 1):
        pred_pts = get_pred_pts()
        pred_adj = _distance_matrix(pred_pts)
        pred_diags = pd_from_graph(pred_adj, **LOSS_CONFIG.pd)
        w = 1

        if w > 0:
            tda_loss, tda_breakdown = tda_loss_fn._loss_from_adjs(
                pred_adj,
                target_adj,
                _return_breakdown=True,
            )
            loss = w * LOSS_CONFIG.tda.weight * tda_loss
            breakdown = dict(tda_breakdown)
        else:
            with torch.no_grad():
                _, tda_breakdown = tda_loss_fn._loss_from_adjs(
                    pred_adj,
                    target_adj,
                    _return_breakdown=True,
                )
            loss = pred_pts.new_zeros(())
            breakdown = dict(tda_breakdown)

        breakdown["tda_w"] = w
        breakdown["total"] = loss

        if step % LOG_EVERY == 0 or step == MLP_STEPS:
            tda_w_msg = ""
            if w < 1.0:
                tda_w_msg = f"  tda_w={w:.3f}"
            pred_counts = [len(pred_diags[i]) for i in range(LOSS_CONFIG.pd.hom_dim)]
            lr = _current_lr(optimizer) if optimizer is not None else MLP_LR
            print(
                f"step {step:4d}  lr={lr:.6g}  loss={_scalar(breakdown['total']):.4f}  "
                f"{_tda_terms_msg(breakdown)}{tda_w_msg}  "
                f"pred_n={pred_counts}  tgt_n={tgt_counts}"
            )
            view_pts = pred_pts.detach().cpu().numpy()
            history.append(
                _make_history_frame(step, pred_pts, pred_diags, breakdown, view_pts)
            )

        if step < MLP_STEPS:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    final_pred = get_pred_pts().detach()
    pc_metrics = point_cloud_metrics(final_pred, target_pts)
    print(
        f"[mlp] point-cloud diff  "
        f"dist_rmse={pc_metrics['dist_rmse']:.4f}  "
        f"dist_w1={pc_metrics['dist_w1']:.4f}  "
        f"rel_rmse={pc_metrics['rel_rmse']:.3f}"
    )
    if visualize:
        show_history(
            [
                {
                    "name": name,
                    "target_diags": target_diags,
                    "target_pts": target_pts_np,
                    "history": history,
                }
            ],
            name,
        )
    return pc_metrics

def run_mlp(device, loss_fn, h0rff, h1rff, h2rff, visualize=False, seed=42):
    target_pts, target_adj = _make_problem(device, seed)
    model = PointMLP(N_POINTS, hidden_dim=HIDDEN_DIM).to(device)
    model_input = torch.randn(1, N_POINTS * 3, device=device) * POINT_SCALE
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    scheduler = build_lr_scheduler(optimizer)
    return _train_point_cloud(
        "mlp",
        target_adj,
        target_pts,
        optimizer,
        scheduler,
        get_pred_pts=lambda: model(model_input),
        loss_fn=loss_fn,
        h0rff=h0rff,
        h1rff=h1rff,
        h2rff=h2rff,
        visualize=visualize
    )

def main():
    device = _resolve_device()

    loss_fn = build_loss_fn()
    if not loss_fn.tda_enabled:
        raise ValueError(
            "No TDA loss terms enabled. Enable wasserstein and/or vpd in LOSS_CONFIG."
        )

    timer = time.time()
    print("Creating heat kernels...")
    if LOSS_CONFIG["vpd_h0"]["enabled"]:
        h0rff = create_heat_random_fourier_features(**HEAT_RFF_CONFIG["h0rff"])
    else:
        h0rff = None
    if LOSS_CONFIG["vpd_h1"]["enabled"]:
        h1rff = create_heat_random_fourier_features(**HEAT_RFF_CONFIG["h1rff"])
    else:
        h1rff = None
    if LOSS_CONFIG["vpd_h2"]["enabled"]:
        h2rff = create_heat_random_fourier_features(**HEAT_RFF_CONFIG["h2rff"])
    else:
        h2rff = None
    print(f"Time taken to create heat kernels: {time.time() - timer:.2f} seconds")

    seed = 42

    for i in range(10):
        seed = seed + 1
        pc_metrics = run_mlp(device, loss_fn, h0rff, h1rff, h2rff, False, seed)
        dist_rmses.append(pc_metrics['dist_rmse'])
        dist_w1s.append(pc_metrics['dist_w1'])
        rel_rmses.append(pc_metrics['rel_rmse'])
        with open("src/tests/mlp_output.txt", "a") as f:
            f.write(
                f"[mlp] point-cloud diff  "
                f"dist_rmse={pc_metrics['dist_rmse']:.4f}  "
                f"dist_w1={pc_metrics['dist_w1']:.4f}  "
                f"rel_rmse={pc_metrics['rel_rmse']:.3f}"
            )
    print(f"\n\n========== Final Results ==========")
    print(f"\n Average Dist RMSE: {np.mean(dist_rmses)}")
    print(f"\n Average Dist W1: {np.mean(dist_w1s)}")
    print(f"\n Average Rel RMSE: {np.mean(rel_rmses)}")
    
if __name__ == "__main__":
    main()
