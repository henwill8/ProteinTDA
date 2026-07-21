from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tests.test_utils import _resolve_device, _scalar
from tests.plotting import _make_history_frame, show_history, _tda_terms_msg

from proteintda.config import CONFIG_OF, LOSS_CONFIG, RUN_CONFIG
from proteintda.minifold.pipeline import _current_lr, build_loss_fn, build_lr_scheduler
from proteintda.minifold.loss import _distance_matrix
from proteintda.tda.persistence import pd_from_graph

N_POINTS = 100
POINT_SCALE = 4.0
MLP_STEPS = 10
MLP_LR = 0.0001
HIDDEN_DIM = 256
SEED = 42

LOG_EVERY_NTH_PROTEIN = 20
EVAL_FRACTION = 0.2
EVAL_SEED = 42
# Second eval: random sample from the full SidechainNet split (not the TM/length-selected set).
REPRESENTATIVE_EVAL_SIZE = 1000
REPRESENTATIVE_EVAL_SEED = 43
REPRESENTATIVE_LOG_EVERY_NTH = 20
LOG_EVERY = 1
TDA_WARMUP_STEPS = 0
TDA_RAMP_STEPS = 0
CHECKPOINT_DIR = Path("logs/tda_loss")

_TDA_BREAKDOWN_KEYS = ("wasserstein_h0", "wasserstein_h1", "wasserstein_h2", "vpd_h0", "vpd_h1", "vpd_h2")

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

def _make_problem(device):
    torch.manual_seed(SEED)
    target_pts = torch.randn(N_POINTS, 3, device=device) * POINT_SCALE
    target_adj = _distance_matrix(target_pts).detach()
    return target_pts, target_adj

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

def _train_point_cloud(name, target_adj, target_pts, optimizer, scheduler, *, get_pred_pts, loss_fn):
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

def run_mlp(device, loss_fn):
    target_pts, target_adj = _make_problem(device)
    model = PointMLP(N_POINTS, hidden_dim=HIDDEN_DIM).to(device)
    model_input = torch.randn(1, N_POINTS * 3, device=device) * POINT_SCALE
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    scheduler = build_lr_scheduler(optimizer)
    _train_point_cloud(
        "mlp",
        target_adj,
        target_pts,
        optimizer,
        scheduler,
        get_pred_pts=lambda: model(model_input),
        loss_fn=loss_fn,
    )


def main():
    device = _resolve_device()

    loss_fn = build_loss_fn()
    if not loss_fn.tda_enabled:
        raise ValueError(
            "No TDA loss terms enabled. Enable wasserstein and/or vpd in LOSS_CONFIG."
        )

    run_mlp(device, loss_fn)
    
if __name__ == "__main__":
    main()
