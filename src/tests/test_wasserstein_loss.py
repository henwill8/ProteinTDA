import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, Slider
from minifold.train.loss import AlphaFoldLoss
from minifold.utils.residue_constants import atom_order
from minifold.utils.tensor_utils import tensor_tree_map
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from tmtools import tm_align

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

N_PROTEINS = 200
BATCH_SIZE = 8
LOG_EVERY_NTH_PROTEIN = 20
EVAL_FRACTION = 0.2
EVAL_SEED = 42
TARGET_LENGTH = 100
STEPS = 200
LR = 0.05
SCHEDULER_STEP = 5
SCHEDULER_GAMMA = 0.9
LOG_EVERY = 10

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
TDA_WARMUP_STEPS = 50
TDA_RAMP_STEPS = 50

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


def wasserstein_loss(pred_pts, target_diags):
    pred_adj = _distance_matrix(pred_pts)
    pred_diags = pd_from_graph(pred_adj, max_dimension=HOM_DIM, hom_dim=HOM_DIM)
    terms = _wasserstein_terms(pred_diags, target_diags, hom_dim=HOM_DIM)
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


def tda_weight(step):
    if TDA_WARMUP_STEPS == 0 and TDA_RAMP_STEPS == 0:
        return 1.0
    if step < TDA_WARMUP_STEPS:
        return 0.0
    if TDA_RAMP_STEPS <= 0:
        return 1.0
    ramp_step = step - TDA_WARMUP_STEPS
    if ramp_step >= TDA_RAMP_STEPS:
        return 1.0
    return ramp_step / TDA_RAMP_STEPS


def _pd_limits(target_diags, history):
    pts = []
    for dim in ACTIVE_HOMS:
        pts.append(_to_numpy(target_diags, dim))
        for frame in history:
            pts.append(frame["pred"][dim])
    if not any(len(p) for p in pts):
        return 1.0
    return max(float(p[:, 1].max()) for p in pts if len(p)) * 1.1 + 0.1


def _point_view_limits(target_pts):
    lo = target_pts.min(axis=0)
    hi = target_pts.max(axis=0)
    pad = (hi - lo) * 0.1 + 0.1
    return lo - pad, hi + pad


def _tm_align(pred_ca, target_ca, seq):
    return tm_align(
        np.asarray(pred_ca, dtype=float),
        np.asarray(target_ca, dtype=float),
        seq,
        seq,
    )


def _apply_tm_align(pts, alignment):
    return pts @ alignment.u.T + alignment.t


def _attach_controls(fig, show, n_steps, *, n_proteins=1):
    state = {"step": 0, "protein": 0}
    holders = []

    def refresh():
        show(state["step"], state["protein"])

    def set_step(step_idx):
        state["step"] = max(0, min(n_steps - 1, step_idx))
        refresh()

    def set_protein(protein_idx):
        state["protein"] = max(0, min(n_proteins - 1, protein_idx))
        refresh()

    fig.subplots_adjust(bottom=0.2 if n_proteins > 1 else 0.16)

    ax_slider = fig.add_axes([0.26, 0.06, 0.48, 0.03])
    slider = Slider(ax_slider, "step", 0, max(n_steps - 1, 0), valinit=0, valstep=1)
    slider.on_changed(lambda val: set_step(int(val)))
    holders.extend((slider, ax_slider))

    ax_prev = fig.add_axes([0.12, 0.055, 0.1, 0.04])
    btn_prev = Button(ax_prev, "<")
    btn_prev.on_clicked(lambda _e: slider.set_val(max(0, state["step"] - 1)))
    holders.extend((btn_prev, ax_prev))

    ax_next = fig.add_axes([0.78, 0.055, 0.1, 0.04])
    btn_next = Button(ax_next, ">")
    btn_next.on_clicked(lambda _e: slider.set_val(min(n_steps - 1, state["step"] + 1)))
    holders.extend((btn_next, ax_next))

    if n_proteins > 1:
        ax_pprev = fig.add_axes([0.12, 0.11, 0.1, 0.04])
        btn_pprev = Button(ax_pprev, "prot <")
        btn_pprev.on_clicked(lambda _e: set_protein(state["protein"] - 1))
        holders.extend((btn_pprev, ax_pprev))

        ax_pnext = fig.add_axes([0.78, 0.11, 0.1, 0.04])
        btn_pnext = Button(ax_pnext, "prot >")
        btn_pnext.on_clicked(lambda _e: set_protein(state["protein"] + 1))
        holders.extend((btn_pnext, ax_pnext))

    fig._nav_widgets = holders

    def go(step_idx=0, protein_idx=0):
        state["step"] = max(0, min(n_steps - 1, step_idx))
        state["protein"] = max(0, min(n_proteins - 1, protein_idx))
        slider.set_val(state["step"])
        refresh()

    return go


def show_history(cases, title):
    if not cases or not cases[0]["history"]:
        return

    n_proteins = len(cases)
    n_steps = len(cases[0]["history"])
    fig = plt.figure(figsize=(12, 6))

    ax_pd = fig.add_subplot(1, 2, 1)
    ax_pts = fig.add_subplot(1, 2, 2, projection="3d")
    ax_pts.set_autoscale_on(False)
    ax_pd.set_xlabel("birth")
    ax_pd.set_ylabel("death")
    ax_pts.set_xlabel("x")
    ax_pts.set_ylabel("y")
    ax_pts.set_zlabel("z")

    step_text = fig.text(0.02, 0.96, "", fontsize=10)
    pd_dynamic = []
    target_scatters = []
    pred_sc = None
    diag_line = None

    def clear_dynamic():
        nonlocal pred_sc, diag_line
        for art in pd_dynamic:
            art.remove()
        pd_dynamic.clear()
        for art in target_scatters:
            art.remove()
        target_scatters.clear()
        if pred_sc is not None:
            pred_sc.remove()
            pred_sc = None
        if diag_line is not None:
            diag_line.remove()
            diag_line = None

    def draw_case(step_idx, protein_idx):
        nonlocal pred_sc, diag_line
        case = cases[protein_idx]
        history = case["history"]
        frame = history[step_idx]
        target_diags = case["target_diags"]
        target_pts = case["target_pts"]

        lim_hi = _pd_limits(target_diags, history)
        diag_line = ax_pd.plot([0, lim_hi], [0, lim_hi], "k--", alpha=0.3, linewidth=1)[0]
        ax_pd.set_xlim(0, lim_hi)
        ax_pd.set_ylim(0, lim_hi)
        ax_pd.set_aspect("equal")

        for dim in ACTIVE_HOMS:
            tgt = _to_numpy(target_diags, dim)
            if len(tgt):
                target_scatters.append(
                    ax_pd.scatter(
                        tgt[:, 0], tgt[:, 1], c="blue", marker=PD_MARKERS[dim],
                        label=f"target H{dim}", s=40, alpha=0.8, zorder=2,
                    )
                )
            pred = frame["pred"][dim]
            if len(pred):
                pd_dynamic.append(
                    ax_pd.scatter(
                        pred[:, 0], pred[:, 1], c="red", marker=PD_MARKERS[dim],
                        s=40, alpha=0.8, zorder=3,
                    )
                )

        handles, labels = ax_pd.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax_pd.legend(by_label.values(), by_label.keys(), loc="lower right")

        lo, hi = _point_view_limits(target_pts)
        target_scatters.append(
            ax_pts.scatter(
                target_pts[:, 0], target_pts[:, 1], target_pts[:, 2],
                c="blue", label="target", s=30, alpha=0.8,
            )
        )
        pred_sc = ax_pts.scatter(
            frame["view_pts"][:, 0], frame["view_pts"][:, 1], frame["view_pts"][:, 2],
            c="red", label="pred", s=30, alpha=0.8,
        )
        ax_pts.set_xlim(lo[0], hi[0])
        ax_pts.set_ylim(lo[1], hi[1])
        ax_pts.set_zlim(lo[2], hi[2])
        ax_pts.legend(loc="upper right")

        protein_msg = ""
        if n_proteins > 1:
            protein_msg = f"  protein {protein_idx + 1}/{n_proteins} ({case['name']})"
        step_text.set_text(
            f"{title}{protein_msg}  frame {step_idx + 1}/{n_steps}  "
            f"step {frame['step']}  loss={frame['loss']:.4f}  "
            f"h0={frame['h0']:.4f}  h1={frame['h1']:.4f}"
            + (f"  h2={frame['h2']:.4f}" if USE_H2 else "")
            + _extra_loss_msg(frame.get("breakdown", {}))
        )
        fig.canvas.draw_idle()

    def show(step_idx, protein_idx):
        clear_dynamic()
        draw_case(step_idx, protein_idx)

    go = _attach_controls(fig, show, n_steps, n_proteins=n_proteins)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, wspace=0.25)
    go(0, 0)
    plt.show(block=True)


def _extra_loss_msg(breakdown):
    parts = []
    if "distogram" in breakdown:
        parts.append(f"disto={_scalar(breakdown['distogram']):.4f}")
    if "structure" in breakdown:
        parts.append(f"struct={_scalar(breakdown['structure']):.4f}")
    if "tm_score" in breakdown:
        parts.append(f"tm_score={breakdown['tm_score']:.4f}")
    return f"  {'  '.join(parts)}" if parts else ""


def _log_loss_config():
    parts = []
    if USE_TDA_LOSS:
        parts.append("tda (wasserstein)")
        if TDA_WARMUP_STEPS > 0 or TDA_RAMP_STEPS > 0:
            parts.append(f"warmup={TDA_WARMUP_STEPS} ramp={TDA_RAMP_STEPS}")
    if USE_DISTOGRAM_LOSS:
        parts.append(f"distogram (w={W_DISTOGRAM})")
    if USE_STRUCTURE_LOSS:
        parts.append(f"structure (w={W_STRUCTURE})")
    if parts:
        print(f"  losses: {', '.join(parts)}")


def _make_history_frame(step, pred_pts, pred_diags, breakdown, view_pts):
    frame = {
        "step": step,
        "loss": _scalar(breakdown["total"]),
        "h0": _scalar(breakdown["wasserstein_h0"]),
        "h1": _scalar(breakdown["wasserstein_h1"]),
        "pred": [_to_numpy(pred_diags, d).copy() for d in ACTIVE_HOMS],
        "pts": pred_pts.detach().cpu().numpy().copy(),
        "view_pts": np.asarray(view_pts, dtype=float).copy(),
        "breakdown": breakdown,
    }
    if USE_H2:
        frame["h2"] = _scalar(breakdown["wasserstein_h2"])
    return frame


def _logged_protein_indices(n_proteins: int, every_nth: int) -> list[int]:
    stride = max(1, every_nth)
    return list(range(0, n_proteins, stride))


class _NullScheduler:
    def get_last_lr(self):
        return [0.0]


def _split_train_eval(cases, eval_fraction: float, seed: int):
    if eval_fraction <= 0 or len(cases) <= 1:
        return cases, []
    n_eval = max(1, int(round(len(cases) * eval_fraction)))
    if n_eval >= len(cases):
        n_eval = len(cases) - 1
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cases))
    eval_set = set(perm[:n_eval])
    train_cases = [case for i, case in enumerate(cases) if i not in eval_set]
    eval_cases = [case for i, case in enumerate(cases) if i in eval_set]
    return train_cases, eval_cases


def _log_step_metrics(step, scheduler, cases, step_results, *, batch_loss):
    for case, (_, pred_diags, breakdown, _) in zip(cases, step_results):
        tgt_counts = [len(d) for d in case["target_diags"]]
        pred_counts = [len(pred_diags[i]) for i in range(HOM_DIM)]
        hom_msg = (
            f"h0={_scalar(breakdown['wasserstein_h0']):.4f}  "
            f"h1={_scalar(breakdown['wasserstein_h1']):.4f}"
        )
        if USE_H2:
            hom_msg += f"  h2={_scalar(breakdown['wasserstein_h2']):.4f}"
        tda_w_msg = ""
        if breakdown.get("tda_w", 1.0) < 1.0:
            tda_w_msg = f"  tda_w={breakdown['tda_w']:.3f}"
        prefix = f"[{case['name']}] " if len(cases) > 1 else ""
        print(
            f"{prefix}step {step:4d}  lr={scheduler.get_last_lr()[0]:.6g}  "
            f"loss={_scalar(breakdown['total']):.4f}  {hom_msg}{tda_w_msg}"
            f"{_extra_loss_msg(breakdown)}  pred_n={pred_counts}  tgt_n={tgt_counts}"
        )
    if len(cases) > 1:
        print(f"  batch_loss={_scalar(batch_loss):.4f}")


def _run_batches(
    runner,
    cases,
    structure_loss_fn,
    step,
    *,
    batch_size: int,
    train: bool,
    progress_desc: str | None = None,
):
    num_batches = max(1, (len(cases) + batch_size - 1) // batch_size)
    batch_losses = []
    step_results: dict[int, tuple] = {}

    batch_starts = range(0, len(cases), batch_size)
    if progress_desc is not None:
        batch_starts = tqdm(batch_starts, desc=progress_desc, leave=False, total=num_batches)

    grad_context = nullcontext() if train else torch.no_grad()
    with grad_context:
        for batch_start in batch_starts:
            batch_cases = cases[batch_start : batch_start + batch_size]
            model_batch = runner.prepare_batch(
                [case["protein"] for case in batch_cases],
                train=train,
            )
            loss, results = _batch_step(
                runner,
                model_batch,
                batch_cases,
                structure_loss_fn,
                step,
                train=train,
            )
            batch_losses.append(loss)
            if train:
                (loss / num_batches).backward()
            for local_idx, result in enumerate(results):
                step_results[batch_start + local_idx] = result

    if len(batch_losses) == 1:
        batch_loss = batch_losses[0]
    else:
        batch_loss = torch.stack(batch_losses).mean()
    return batch_loss, step_results


def train(
    runner,
    cases,
    optimizer,
    scheduler,
    structure_loss_fn,
    *,
    batch_size: int = 1,
    log_every_nth: int = 1,
    group_name: str = "train",
):
    print(
        f"[{group_name}] {len(cases)} protein(s), batch_size={batch_size}, "
        f"{STEPS} steps, lr={LR}"
    )
    _log_loss_config()
    logged_indices = _logged_protein_indices(len(cases), log_every_nth)
    if log_every_nth > 1:
        print(f"  logging/visualizing every {log_every_nth}th protein ({len(logged_indices)} total)")
    histories = {idx: [] for idx in logged_indices}
    hom_label = "/".join(f"H{d}" for d in ACTIVE_HOMS)
    for case in cases:
        tgt_counts = [len(d) for d in case["target_diags"]]
        print(f"  {case['name']}: {case['target_pts'].shape[0]} points  {hom_label}={tgt_counts}")

    for step in range(STEPS + 1):
        if step < STEPS:
            optimizer.zero_grad()
        batch_loss, step_results = _run_batches(
            runner,
            cases,
            structure_loss_fn,
            step,
            batch_size=batch_size,
            train=True,
            progress_desc=f"epoch {step + 1}/{STEPS + 1}",
        )
        if step % LOG_EVERY == 0 or step == STEPS:
            logged_cases = [cases[i] for i in logged_indices]
            logged_results = [step_results[i] for i in logged_indices]
            _log_step_metrics(
                step,
                scheduler,
                logged_cases,
                logged_results,
                batch_loss=batch_loss,
            )
            for idx in logged_indices:
                histories[idx].append(_make_history_frame(step, *step_results[idx]))

        if step < STEPS:
            optimizer.step()
            scheduler.step()

    show_history(
        [
            {
                "name": cases[idx]["name"],
                "target_diags": cases[idx]["target_diags"],
                "target_pts": cases[idx]["target_pts_np"],
                "history": histories[idx],
            }
            for idx in logged_indices
            if histories[idx]
        ],
        group_name,
    )


def evaluate(
    runner,
    cases,
    structure_loss_fn,
    *,
    batch_size: int = 1,
    log_every_nth: int = 1,
    group_name: str = "eval",
):
    if not cases:
        return

    print(
        f"[{group_name}] {len(cases)} protein(s), batch_size={batch_size} "
        f"(held out from training)"
    )
    logged_indices = _logged_protein_indices(len(cases), log_every_nth)
    if log_every_nth > 1:
        print(f"  logging/visualizing every {log_every_nth}th protein ({len(logged_indices)} total)")
    hom_label = "/".join(f"H{d}" for d in ACTIVE_HOMS)
    for case in cases:
        tgt_counts = [len(d) for d in case["target_diags"]]
        print(f"  {case['name']}: {case['target_pts'].shape[0]} points  {hom_label}={tgt_counts}")

    runner.model.eval()
    for module in runner._frozen_modules:
        module.eval()

    batch_loss, step_results = _run_batches(
        runner,
        cases,
        structure_loss_fn,
        STEPS,
        batch_size=batch_size,
        train=False,
        progress_desc="eval",
    )
    logged_cases = [cases[i] for i in logged_indices]
    logged_results = [step_results[i] for i in logged_indices]
    _log_step_metrics(
        STEPS,
        _NullScheduler(),
        logged_cases,
        logged_results,
        batch_loss=batch_loss,
    )

    histories = {
        idx: [_make_history_frame(STEPS, *step_results[idx])]
        for idx in logged_indices
    }
    show_history(
        [
            {
                "name": cases[idx]["name"],
                "target_diags": cases[idx]["target_diags"],
                "target_pts": cases[idx]["target_pts_np"],
                "history": histories[idx],
            }
            for idx in logged_indices
        ],
        group_name,
    )


def _load_cases(device, n=1, target_length=TARGET_LENGTH):
    proteins = load_all_proteins()
    if not proteins:
        raise ValueError("dataset is empty")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    selected = sorted(proteins, key=lambda p: (abs(len(p.seq) - target_length), p.id))[:n]
    if n == 1 and len(selected[0].seq) == target_length:
        print(f"selected {selected[0].id}  len={target_length}")
    else:
        print(f"selected {n} proteins (closest to TARGET_LENGTH={target_length}):")
        for protein in selected:
            print(f"  {protein.id}  len={len(protein.seq)}")

    cases = []
    for protein in selected:
        target_pts = atom_positions_from_sidechainnet(
            protein, SideChainAtom.CB, device=device,
        )
        if target_pts.shape[0] != len(protein.seq):
            raise ValueError(
                f"protein {protein.id} has {target_pts.shape[0]} C-beta points "
                f"but len(seq)={len(protein.seq)}",
            )
        target_ca = atom_positions_from_sidechainnet(
            protein, SideChainAtom.CA, device=device,
        )
        target_adj = _distance_matrix(target_pts).detach()
        cases.append({
            "protein": protein,
            "name": protein.id,
            "seq": str(protein.seq),
            "target_pts": target_pts,
            "target_ca_np": target_ca.detach().cpu().numpy(),
            "target_adj": target_adj,
            "target_diags": pd_from_graph(
                target_adj, max_dimension=HOM_DIM, hom_dim=HOM_DIM,
            ),
            "target_pts_np": target_pts.detach().cpu().numpy(),
        })
    return cases


def _cbeta_from_output(r_dict, index=0):
    return atom_positions_from_atom37(
        r_dict["final_atom_positions"][index],
        r_dict["final_atom_mask"][index],
        Atom37.CB,
    )


def _ca_from_output(r_dict, index=0):
    ca_idx = atom_order["CA"]
    return r_dict["final_atom_positions"][index, :, ca_idx].detach().float().cpu()


def _case_step(r_dict, case, index, w, shared_breakdown):
    pred_pts = _cbeta_from_output(r_dict, index)
    breakdown = dict(shared_breakdown)

    if USE_TDA_LOSS and w > 0:
        tda_loss, pred_diags, tda_breakdown = wasserstein_loss(pred_pts, case["target_diags"])
        protein_total = w * tda_loss
        breakdown.update(tda_breakdown)
    else:
        with torch.no_grad():
            _, pred_diags, tda_breakdown = wasserstein_loss(pred_pts, case["target_diags"])
        protein_total = pred_pts.new_zeros(())
        breakdown.update(tda_breakdown)

    breakdown["tda_w"] = w
    with torch.no_grad():
        pred_ca = _ca_from_output(r_dict, index).numpy()
        alignment = _tm_align(pred_ca, case["target_ca_np"], case["seq"])
        breakdown["tm_score"] = float(alignment.tm_norm_chain2)
        view_pts = _apply_tm_align(pred_pts.detach().cpu().numpy(), alignment)

    breakdown["total"] = protein_total
    return protein_total, pred_pts, pred_diags, breakdown, view_pts


def _batch_step(runner, model_batch, cases, structure_loss_fn, step, *, train=True):
    if train:
        runner._set_training_mode()
    else:
        runner.model.eval()
        for module in runner._frozen_modules:
            module.eval()
    r_dict = runner.model(model_batch, num_recycling=MINIFOLD_RECYCLES)
    w = tda_weight(step)

    total = r_dict["preds"].new_zeros(())
    shared_breakdown = {}

    if USE_DISTOGRAM_LOSS:
        disto = MiniFoldLoss._distogram_loss(
            r_dict["preds"],
            model_batch["coords"],
            model_batch["mask"],
            runner.model.boundaries,
            no_bins=r_dict["preds"].shape[-1],
        )
        total = total + W_DISTOGRAM * disto
        shared_breakdown["distogram"] = disto

    if USE_STRUCTURE_LOSS:
        batch_of = tensor_tree_map(lambda t: t[..., -1], model_batch["batch_of"])
        struct_loss, _ = structure_loss_fn(r_dict, batch_of, _return_breakdown=True)
        total = total + W_STRUCTURE * struct_loss
        shared_breakdown["structure"] = struct_loss

    protein_totals = []
    results = []
    for index, case in enumerate(cases):
        protein_total, pred_pts, pred_diags, breakdown, view_pts = _case_step(
            r_dict, case, index, w, shared_breakdown,
        )
        protein_totals.append(protein_total)
        results.append((pred_pts, pred_diags, breakdown, view_pts))

    if USE_TDA_LOSS and w > 0:
        total = total + torch.stack(protein_totals).mean()

    for _, _, breakdown, _ in results:
        breakdown["batch_total"] = total.detach()

    return total, results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_cases = _load_cases(device, n=N_PROTEINS)
    train_cases, eval_cases = _split_train_eval(all_cases, EVAL_FRACTION, EVAL_SEED)
    print(
        f"split {len(all_cases)} proteins -> "
        f"{len(train_cases)} train, {len(eval_cases)} eval "
        f"(eval_fraction={EVAL_FRACTION})"
    )
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

    optimizer = torch.optim.Adam(
        [p for p in runner.model.parameters() if p.requires_grad],
        lr=LR,
    )
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)
    structure_loss_fn = AlphaFoldLoss(CONFIG_OF.loss)

    train(
        runner,
        train_cases,
        optimizer,
        scheduler,
        structure_loss_fn,
        batch_size=BATCH_SIZE,
        log_every_nth=LOG_EVERY_NTH_PROTEIN,
        group_name="train",
    )

    if eval_cases:
        evaluate(
            runner,
            eval_cases,
            structure_loss_fn,
            batch_size=BATCH_SIZE,
            log_every_nth=LOG_EVERY_NTH_PROTEIN,
            group_name="eval",
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
