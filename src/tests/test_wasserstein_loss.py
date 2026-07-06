import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from proteintda.minifold.loss import _distance_matrix, _wasserstein_terms
from proteintda.minifold.runner import MiniFoldRunner
from proteintda.tda.persistence import pd_from_graph
from proteintda.utils.conversions import (
    Atom37,
    SideChainAtom,
    atom_positions_from_atom37,
    atom_positions_from_sidechainnet,
)
from proteintda.utils.dataset import load_dataset

N_POINTS = 100
STEPS = 200
LR = 0.05
SCHEDULER_STEP = 5
SCHEDULER_GAMMA = 0.9
LOG_EVERY = 10
SEED = 42
TRAIL_LENGTH = 50

MODE = "minifold"  # "points", "mlp", or "minifold"
MINIFOLD_CACHE_DIR = Path("cache/minifold")
MINIFOLD_MODEL_SIZE = "12L"  # "12L" or "48L"
MINIFOLD_RECYCLES = 3
MINIFOLD_LR = 1e-4
MINIFOLD_UNFREEZE_FOLD_BLOCKS = 0
MINIFOLD_UNFREEZE_STRUCTURE_MODULE = True

HIDDEN_DIM = 256
USE_H2 = False
HOM_DIM = 3 if USE_H2 else 2
ACTIVE_HOMS = (0, 1, 2) if USE_H2 else (0, 1)
W_H0 = 0.2
W_H1 = 1.0
W_H2 = 1.0
PD_MARKERS = ("o", "^", "s")

def _scalar(x):
    return x.item() if hasattr(x, "item") else float(x)


def _to_numpy(diags, dim):
    if len(diags) < dim + 1:
        return np.empty((0, 2))
    return diags[dim].detach().cpu().numpy()


def _gudhi_wasserstein_terms(pred_adj, target_diags):
    pred_diags = pd_from_graph(pred_adj, max_dimension=HOM_DIM, hom_dim=HOM_DIM)
    terms = _wasserstein_terms(pred_diags, target_diags, hom_dim=HOM_DIM)
    return terms, pred_diags


def wasserstein_loss(pred_pts, target_diags):
    pred_adj = _distance_matrix(pred_pts)
    terms, pred_diags = _gudhi_wasserstein_terms(pred_adj, target_diags)
    loss = W_H0 * terms["h0"] + W_H1 * terms["h1"] + (W_H2 * terms["h2"] if USE_H2 else 0)
    if not isinstance(loss, torch.Tensor):
        loss = pred_adj.new_tensor(loss)
    breakdown = {
        "wasserstein_h0": terms["h0"],
        "wasserstein_h1": terms["h1"],
        "total": loss,
    }
    if USE_H2:
        breakdown["wasserstein_h2"] = terms["h2"]
    return loss, pred_diags, breakdown


def _nearest_trails(prev, curr):
    if len(prev) == 0 or len(curr) == 0:
        return []
    dists = ((prev[:, None, :] - curr[None, :, :]) ** 2).sum(axis=2)
    return [(prev[i], curr[j]) for j, i in enumerate(dists.argmin(axis=0))]


def _trail_alpha(frame_idx, seg_idx, trail_start):
    age = frame_idx - seg_idx
    n_segs = frame_idx - trail_start + 1
    t = age / max(n_segs - 1, 1)
    return 0.2 * (1 - t) + 0.04 * t


def _pd_limits(target_diags, history):
    pts = []
    for dim in ACTIVE_HOMS:
        pts.append(_to_numpy(target_diags, dim))
        for frame in history:
            pts.append(frame["pred"][dim])
    return max(float(p[:, 1].max()) for p in pts if len(p)) * 1.1 + 0.1


def _pts_limits(target_pts, history):
    all_coords = np.concatenate([target_pts, *[f["pts"] for f in history]], axis=0)
    lo, hi = float(all_coords.min()), float(all_coords.max())
    pad = (hi - lo) * 0.1 + 0.1
    return lo - pad, hi + pad


def _attach_buttons(fig, show, n_frames):
    """Keep Button refs alive or callbacks get garbage-collected."""
    state = {"i": 0}
    holders = []

    def go(i):
        state["i"] = max(0, min(n_frames - 1, i))
        show(state["i"])

    fig.subplots_adjust(bottom=0.14)
    btn_h, btn_y = 0.05, 0.03
    specs = [
        ([0.12, btn_y, 0.12, btn_h], "|<", lambda _e: go(0)),
        ([0.26, btn_y, 0.12, btn_h], "<", lambda _e: go(state["i"] - 1)),
        ([0.62, btn_y, 0.12, btn_h], ">", lambda _e: go(state["i"] + 1)),
        ([0.76, btn_y, 0.12, btn_h], ">|", lambda _e: go(n_frames - 1)),
    ]
    for rect, label, handler in specs:
        ax_btn = fig.add_axes(rect)
        btn = Button(ax_btn, label)
        btn.on_clicked(handler)
        holders.extend((btn, ax_btn))

    fig._button_widgets = holders
    return go


def show_history(history, target_diags, target_pts, title):
    if not history:
        return

    ncols = 2
    fig = plt.figure(figsize=(6 * ncols, 6))

    ax_pd = None
    ax_pts = None
    ax_pd = fig.add_subplot(1, ncols, 1)
    lim_hi = _pd_limits(target_diags, history)
    ax_pd.plot([0, lim_hi], [0, lim_hi], "k--", alpha=0.3, linewidth=1)
    ax_pd.set_xlim(0, lim_hi)
    ax_pd.set_ylim(0, lim_hi)
    ax_pd.set_aspect("equal")
    ax_pd.set_xlabel("birth")
    ax_pd.set_ylabel("death")
    for dim in ACTIVE_HOMS:
        tgt = _to_numpy(target_diags, dim)
        if len(tgt):
            ax_pd.scatter(
                tgt[:, 0], tgt[:, 1], c="blue", marker=PD_MARKERS[dim],
                label=f"target H{dim}", s=40, alpha=0.8, zorder=2,
            )
    ax_pd.legend(loc="lower right")

    col = 2
    ax_pts = fig.add_subplot(1, ncols, col, projection="3d")
    pts_lo, pts_hi = _pts_limits(target_pts, history)
    ax_pts.scatter(
        target_pts[:, 0], target_pts[:, 1], target_pts[:, 2],
        c="blue", label="target", s=30, alpha=0.8,
    )
    init_pts = history[0]["pts"]
    pred_sc = ax_pts.scatter(
        init_pts[:, 0], init_pts[:, 1], init_pts[:, 2],
        c="red", label="pred", s=30, alpha=0.8,
    )
    ax_pts.set_xlim(pts_lo, pts_hi)
    ax_pts.set_ylim(pts_lo, pts_hi)
    ax_pts.set_zlim(pts_lo, pts_hi)
    ax_pts.set_xlabel("x")
    ax_pts.set_ylabel("y")
    ax_pts.set_zlabel("z")
    ax_pts.legend(loc="upper right")

    step_text = fig.text(0.02, 0.96, "", fontsize=10)
    pd_dynamic = []
    pt_trail_artists = []

    def clear_dynamic():
        for art in pd_dynamic:
            art.remove()
        pd_dynamic.clear()
        for art in pt_trail_artists:
            art.remove()
        pt_trail_artists.clear()

    def draw_pd(frame_idx):
        frame = history[frame_idx]
        trail_start = max(1, frame_idx - TRAIL_LENGTH + 1)
        for dim in ACTIVE_HOMS:
            pred = frame["pred"][dim]
            if len(pred):
                pd_dynamic.append(
                    ax_pd.scatter(
                        pred[:, 0], pred[:, 1], c="red", marker=PD_MARKERS[dim],
                        s=40, alpha=0.8, zorder=3,
                    )
                )
            for seg_idx in range(trail_start, frame_idx + 1):
                prev = history[seg_idx - 1]["pred"][dim]
                curr = history[seg_idx]["pred"][dim]
                alpha = _trail_alpha(frame_idx, seg_idx, trail_start)
                for p0, p1 in _nearest_trails(prev, curr):
                    (ln,) = ax_pd.plot(
                        [p0[0], p1[0]], [p0[1], p1[1]],
                        c="red", alpha=alpha, linewidth=1, zorder=1,
                    )
                    pd_dynamic.append(ln)

    def draw_pts(frame_idx):
        frame = history[frame_idx]
        pts = frame["pts"]
        trail_start = max(1, frame_idx - TRAIL_LENGTH + 1)
        pred_sc._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        for seg_idx in range(trail_start, frame_idx + 1):
            prev = history[seg_idx - 1]["pts"]
            curr = history[seg_idx]["pts"]
            alpha = _trail_alpha(frame_idx, seg_idx, trail_start)
            for i in range(len(curr)):
                (ln,) = ax_pts.plot(
                    [prev[i, 0], curr[i, 0]],
                    [prev[i, 1], curr[i, 1]],
                    [prev[i, 2], curr[i, 2]],
                    c="red", alpha=alpha, linewidth=1,
                )
                pt_trail_artists.append(ln)

    def show(frame_idx):
        clear_dynamic()
        frame = history[frame_idx]
        if ax_pd is not None:
            draw_pd(frame_idx)
        if ax_pts is not None:
            draw_pts(frame_idx)
        step_text.set_text(
            f"{title}  frame {frame_idx + 1}/{len(history)}  "
            f"step {frame['step']}  loss={frame['loss']:.4f}  "
            f"h0={frame['h0']:.4f}  h1={frame['h1']:.4f}"
            + (f"  h2={frame['h2']:.4f}" if USE_H2 else "")
        )
        fig.canvas.draw()

    go = _attach_buttons(fig, show, len(history))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.14, top=0.9, wspace=0.25)
    go(0)
    plt.show(block=True)


def train(name, target_adj, target_pts, get_pred_pts, optimizer, scheduler, device):
    n_pts = target_pts.shape[0]
    print(f"[{name}] {n_pts} points, {STEPS} steps, lr={LR}")
    history = []
    target_diags = pd_from_graph(target_adj.detach(), max_dimension=HOM_DIM, hom_dim=HOM_DIM)
    tgt_counts = [len(d) for d in target_diags]
    hom_label = "/".join(f"H{d}" for d in ACTIVE_HOMS)
    print(f"target diagram sizes {hom_label}={tgt_counts}")

    for step in range(STEPS + 1):
        pred_pts = get_pred_pts()
        log_step = step % LOG_EVERY == 0 or step == STEPS
        loss, pred_diags, breakdown = wasserstein_loss(pred_pts, target_diags)
        loss_val = _scalar(breakdown["total"])

        if step < STEPS:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if log_step:
            h0 = _scalar(breakdown["wasserstein_h0"])
            h1 = _scalar(breakdown["wasserstein_h1"])
            pred_counts = [len(pred_diags[i]) for i in range(HOM_DIM)]
            hom_msg = f"h0={h0:.4f}  h1={h1:.4f}"
            if USE_H2:
                h2 = _scalar(breakdown["wasserstein_h2"])
                hom_msg += f"  h2={h2:.4f}"
            print(
                f"step {step:4d}  lr={scheduler.get_last_lr()[0]:.6g}  loss={loss_val:.4f}  "
                f"{hom_msg}  pred_n={pred_counts}  tgt_n={tgt_counts}"
            )
            frame = {
                "step": step,
                "loss": loss_val,
                "h0": h0,
                "h1": h1,
                "pred": [_to_numpy(pred_diags, d).copy() for d in ACTIVE_HOMS],
                "pts": pred_pts.detach().cpu().numpy().copy(),
            }
            if USE_H2:
                frame["h2"] = h2
            history.append(frame)

    show_history(history, target_diags, target_pts.detach().cpu().numpy(), name)


def _make_problem(device):
    torch.manual_seed(SEED)
    target_pts = torch.randn(N_POINTS, 3, device=device)
    target_adj = _distance_matrix(target_pts).detach()
    return target_pts, target_adj


def _load_protein(device):
    proteins = load_dataset()
    if not proteins:
        raise ValueError("dataset is empty")
    exact = [p for p in proteins if len(p.seq) == N_POINTS]
    if exact:
        protein = exact[0]
        selection = f"exact length {N_POINTS}"
    else:
        protein = min(proteins, key=lambda p: (abs(len(p.seq) - N_POINTS), p.id))
        selection = f"closest to N_POINTS={N_POINTS}"
    target_pts = atom_positions_from_sidechainnet(
        protein, SideChainAtom.CB, device=device,
    )
    n_pts = target_pts.shape[0]
    if n_pts != len(protein.seq):
        raise ValueError(
            f"protein {protein.id} has {n_pts} C-beta points but len(seq)={len(protein.seq)}",
        )
    print(
        f"loaded protein {protein.id}  ({selection})  "
        f"len={n_pts}  seq={protein.seq[:32]}...",
    )
    return protein, target_pts


def _cbeta_from_minifold_output(r_dict):
    pred_positions = r_dict["final_atom_positions"][0]
    pred_mask = r_dict["final_atom_mask"][0]
    return atom_positions_from_atom37(pred_positions, pred_mask, Atom37.CB)


def _make_optimizer_scheduler(params, lr=LR):
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)
    return optimizer, scheduler


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

    def forward(self, x):
        return self.net(x).view(self.n_points, 3)


def run_points(device):
    target_pts, target_adj = _make_problem(device)
    pred_pts = torch.randn(N_POINTS, 3, device=device, requires_grad=True)
    optimizer, scheduler = _make_optimizer_scheduler([pred_pts])
    train(
        "points",
        target_adj,
        target_pts,
        lambda: pred_pts,
        optimizer,
        scheduler,
        device,
    )


def run_mlp(device):
    target_pts, target_adj = _make_problem(device)
    model = PointMLP(N_POINTS, hidden_dim=HIDDEN_DIM).to(device)
    model_input = torch.randn(1, N_POINTS * 3, device=device)
    optimizer, scheduler = _make_optimizer_scheduler(model.parameters())
    train(
        "mlp",
        target_adj,
        target_pts,
        lambda: model(model_input),
        optimizer,
        scheduler,
        device,
    )


def run_minifold(device):
    protein, target_pts = _load_protein(device)
    target_adj = _distance_matrix(target_pts).detach()
    runner = MiniFoldRunner(
        MINIFOLD_CACHE_DIR,
        model_size=MINIFOLD_MODEL_SIZE,
        device=device,
        train=True,
        unfreeze_fold_blocks=MINIFOLD_UNFREEZE_FOLD_BLOCKS,
        unfreeze_structure_module=MINIFOLD_UNFREEZE_STRUCTURE_MODULE,
    )
    trainable, total = runner.trainable_parameter_count
    print(f"MiniFold trainable parameters: {trainable:,} / {total:,}")

    trainable_params = [p for p in runner.model.parameters() if p.requires_grad]
    optimizer, scheduler = _make_optimizer_scheduler(trainable_params, lr=MINIFOLD_LR)
    model_batch = runner.prepare_batch(protein, train=True)

    def get_pred_pts():
        runner._set_training_mode()
        out = runner.model(model_batch, num_recycling=MINIFOLD_RECYCLES)
        return _cbeta_from_minifold_output(out)

    train(
        f"minifold:{protein.id}",
        target_adj,
        target_pts,
        get_pred_pts,
        optimizer,
        scheduler,
        device,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MODE == "points":
        run_points(device)
    elif MODE == "mlp":
        run_mlp(device)
    elif MODE == "minifold":
        run_minifold(device)
    else:
        raise ValueError(f"unknown MODE={MODE!r}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
