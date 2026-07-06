import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import torch
import torch.nn as nn
from minifold.train.loss import AlphaFoldLoss
from minifold.utils.tensor_utils import tensor_tree_map
from torch.optim.lr_scheduler import StepLR

from proteintda.config import CONFIG_OF
from proteintda.minifold.loss import MiniFoldLoss, _distance_matrix, _wasserstein_terms
from proteintda.minifold.runner import MiniFoldRunner
from proteintda.tda.persistence import pd_from_graph
from proteintda.utils.conversions import (
    Atom37,
    SideChainAtom,
    atom_positions_from_atom37,
    atom_positions_from_sidechainnet,
)
from proteintda.utils.dataset import load_all_proteins

N_POINTS = 100
STEPS = 200
LR = 0.05
SCHEDULER_STEP = 5
SCHEDULER_GAMMA = 0.9
LOG_EVERY = 10
SEED = 42

MODE = "minifold"  # "points", "mlp", or "minifold"
MINIFOLD_CACHE_DIR = Path("cache/minifold")
MINIFOLD_MODEL_SIZE = "12L"  # "12L" or "48L"
MINIFOLD_RECYCLES = 3
MINIFOLD_UNFREEZE_FOLD_BLOCKS = 0
MINIFOLD_UNFREEZE_STRUCTURE_MODULE = True
USE_DISTOGRAM_LOSS = False
USE_STRUCTURE_LOSS = False
USE_TDA_LOSS = True
W_DISTOGRAM = 0.8
W_STRUCTURE = 0.2

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


def _pd_limits(target_diags, history):
    pts = []
    for dim in ACTIVE_HOMS:
        pts.append(_to_numpy(target_diags, dim))
        for frame in history:
            pts.append(frame["pred"][dim])
    return max(float(p[:, 1].max()) for p in pts if len(p)) * 1.1 + 0.1


def _pts_limits(target_pts, history, align_pred_pts=False):
    pred_pts = [
        _kabsch_align(f["pts"], target_pts) if align_pred_pts else f["pts"]
        for f in history
    ]
    all_coords = np.concatenate([target_pts, *pred_pts], axis=0)
    lo, hi = float(all_coords.min()), float(all_coords.max())
    pad = (hi - lo) * 0.1 + 0.1
    return lo - pad, hi + pad


def _kabsch_align(pred, target):
    pc = pred.mean(axis=0)
    tc = target.mean(axis=0)
    pred_c = pred - pc
    tgt_c = target - tc
    h = pred_c.T @ tgt_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[-1] *= -1
        r = vt.T @ u.T
    return (pred - pc) @ r.T + tc


def _view_pred_pts(pred, target_pts, align_pred_pts):
    if not align_pred_pts:
        return pred
    return _kabsch_align(pred, target_pts)


def _attach_buttons(fig, show, n_frames):
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


def show_history(history, target_diags, target_pts, title, *, align_pred_pts=False):
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
    pts_lo, pts_hi = _pts_limits(target_pts, history, align_pred_pts=align_pred_pts)
    ax_pts.scatter(
        target_pts[:, 0], target_pts[:, 1], target_pts[:, 2],
        c="blue", label="target", s=30, alpha=0.8,
    )
    init_pts = _view_pred_pts(history[0]["pts"], target_pts, align_pred_pts)
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

    def clear_dynamic():
        for art in pd_dynamic:
            art.remove()
        pd_dynamic.clear()

    def draw_pd(frame_idx):
        frame = history[frame_idx]
        for dim in ACTIVE_HOMS:
            pred = frame["pred"][dim]
            if len(pred):
                pd_dynamic.append(
                    ax_pd.scatter(
                        pred[:, 0], pred[:, 1], c="red", marker=PD_MARKERS[dim],
                        s=40, alpha=0.8, zorder=3,
                    )
                )

    def draw_pts(frame_idx):
        frame = history[frame_idx]
        pts = _view_pred_pts(frame["pts"], target_pts, align_pred_pts)
        pred_sc._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

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


def _extra_loss_msg(breakdown):
    parts = []
    if "distogram" in breakdown:
        parts.append(f"disto={_scalar(breakdown['distogram']):.4f}")
    if "structure" in breakdown:
        parts.append(f"struct={_scalar(breakdown['structure']):.4f}")
    return f"  {'  '.join(parts)}" if parts else ""


def _log_loss_config():
    parts = []
    if USE_TDA_LOSS:
        parts.append("tda (wasserstein)")
    if USE_DISTOGRAM_LOSS:
        parts.append(f"distogram (w={W_DISTOGRAM})")
    if USE_STRUCTURE_LOSS:
        parts.append(f"structure (w={W_STRUCTURE})")
    if parts:
        print(f"  losses: {', '.join(parts)}")


def train(
    name, target_adj, target_pts, optimizer, scheduler, device, *,
    get_pred_pts=None,
    get_step=None,
    align_pred_pts=False,
):
    n_pts = target_pts.shape[0]
    print(f"[{name}] {n_pts} points, {STEPS} steps, lr={LR}")
    _log_loss_config()
    history = []
    target_diags = pd_from_graph(target_adj.detach(), max_dimension=HOM_DIM, hom_dim=HOM_DIM)
    tgt_counts = [len(d) for d in target_diags]
    hom_label = "/".join(f"H{d}" for d in ACTIVE_HOMS)
    print(f"target diagram sizes {hom_label}={tgt_counts}")

    if get_step is None:
        if get_pred_pts is None:
            raise ValueError("train requires get_pred_pts or get_step")
        def get_step():
            pred_pts = get_pred_pts()
            loss, pred_diags, breakdown = wasserstein_loss(pred_pts, target_diags)
            return loss, pred_pts, pred_diags, breakdown

    for step in range(STEPS + 1):
        loss, pred_pts, pred_diags, breakdown = get_step()
        log_step = step % LOG_EVERY == 0 or step == STEPS
        loss_val = _scalar(breakdown["total"])

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
                f"{hom_msg}{_extra_loss_msg(breakdown)}  "
                f"pred_n={pred_counts}  tgt_n={tgt_counts}"
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

        if step < STEPS:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    show_history(
        history,
        target_diags,
        target_pts.detach().cpu().numpy(),
        name,
        align_pred_pts=align_pred_pts,
    )


def _make_problem(device):
    torch.manual_seed(SEED)
    target_pts = torch.randn(N_POINTS, 3, device=device)
    target_adj = _distance_matrix(target_pts).detach()
    return target_pts, target_adj


def _load_protein(device):
    proteins = load_all_proteins()
    if not proteins:
        raise ValueError("dataset is empty")
    exact = [p for p in proteins if len(p.seq) == N_POINTS]
    if exact:
        protein = exact[0]
        print(f"selected {protein.id}  len={N_POINTS}  (exact match for N_POINTS={N_POINTS})")
    else:
        protein = min(proteins, key=lambda p: (abs(len(p.seq) - N_POINTS), p.id))
        n_seq = len(protein.seq)
        print(
            f"no protein with length {N_POINTS}; "
            f"selected {protein.id}  len={n_seq}  "
        )
    target_pts = atom_positions_from_sidechainnet(
        protein, SideChainAtom.CB, device=device,
    )
    n_pts = target_pts.shape[0]
    if n_pts != len(protein.seq):
        raise ValueError(
            f"protein {protein.id} has {n_pts} C-beta points but len(seq)={len(protein.seq)}",
        )
    return protein, target_pts


def _cbeta_from_minifold_output(r_dict):
    pred_positions = r_dict["final_atom_positions"][0]
    pred_mask = r_dict["final_atom_mask"][0]
    return atom_positions_from_atom37(pred_positions, pred_mask, Atom37.CB)


def _minifold_step(runner, model_batch, target_diags, structure_loss_fn):
    runner._set_training_mode()
    r_dict = runner.model(model_batch, num_recycling=MINIFOLD_RECYCLES)
    pred_pts = _cbeta_from_minifold_output(r_dict)

    if USE_TDA_LOSS:
        tda_loss, pred_diags, breakdown = wasserstein_loss(pred_pts, target_diags)
        total = tda_loss
    else:
        with torch.no_grad():
            _, pred_diags, breakdown = wasserstein_loss(pred_pts, target_diags)
        total = pred_pts.new_zeros(())

    if USE_DISTOGRAM_LOSS:
        disto = MiniFoldLoss._distogram_loss(
            r_dict["preds"],
            model_batch["coords"],
            model_batch["mask"],
            runner.model.boundaries,
            no_bins=r_dict["preds"].shape[-1],
        )
        total = total + W_DISTOGRAM * disto
        breakdown["distogram"] = disto

    if USE_STRUCTURE_LOSS:
        batch_of = tensor_tree_map(lambda t: t[..., -1], model_batch["batch_of"])
        struct_loss, _ = structure_loss_fn(r_dict, batch_of, _return_breakdown=True)
        total = total + W_STRUCTURE * struct_loss
        breakdown["structure"] = struct_loss

    breakdown["total"] = total
    return total, pred_pts, pred_diags, breakdown


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
        optimizer,
        scheduler,
        device,
        get_pred_pts=lambda: pred_pts,
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
        optimizer,
        scheduler,
        device,
        get_pred_pts=lambda: model(model_input),
    )


def run_minifold(device):
    if not USE_TDA_LOSS and not USE_DISTOGRAM_LOSS and not USE_STRUCTURE_LOSS:
        raise ValueError("enable at least one of USE_TDA_LOSS, USE_DISTOGRAM_LOSS, USE_STRUCTURE_LOSS")
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
    optimizer, scheduler = _make_optimizer_scheduler(trainable_params)
    model_batch = runner.prepare_batch(protein, train=True)
    structure_loss_fn = AlphaFoldLoss(CONFIG_OF.loss)
    target_diags = pd_from_graph(target_adj.detach(), max_dimension=HOM_DIM, hom_dim=HOM_DIM)

    def get_step():
        return _minifold_step(runner, model_batch, target_diags, structure_loss_fn)

    train(
        f"minifold:{protein.id}",
        target_adj,
        target_pts,
        optimizer,
        scheduler,
        device,
        get_step=get_step,
        align_pred_pts=True,
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
