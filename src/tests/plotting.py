import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import torch

from tests.test_utils import _to_numpy, _scalar

from proteintda.config import CONFIG_OF, LOSS_CONFIG, RUN_CONFIG

SHOW_PLOTS = True

_PLOT_GROUPS: list[dict] = []
_PLOT_METRICS: dict = {}

PD_MARKERS = ("o", "^", "s")
_TDA_BREAKDOWN_KEYS = ("wasserstein_h0", "wasserstein_h1", "wasserstein_h2", "vpd_h0", "vpd_h1", "vpd_h2")

def _tda_terms_msg(breakdown: dict) -> str:
    parts = []
    for key in _TDA_BREAKDOWN_KEYS:
        if key in breakdown:
            parts.append(f"{key}={_scalar(breakdown[key]):.4f}")
    return "  ".join(parts)

def _extra_loss_msg(breakdown):
    parts = []
    if "distogram" in breakdown:
        parts.append(f"disto={_scalar(breakdown['distogram']):.4f}")
    if "structure" in breakdown:
        parts.append(f"struct={_scalar(breakdown['structure']):.4f}")
    if "tm_score" in breakdown:
        parts.append(f"tm_score={breakdown['tm_score']:.4f}")
    if "plddt" in breakdown:
        parts.append(f"plddt={breakdown['plddt']:.4f}")
    return f"  {'  '.join(parts)}" if parts else ""

def _serialize_value(value):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return _scalar(value.detach())
        return value.detach().cpu().numpy().copy()
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _serialize_plot_cases(cases):
    hom_dim = LOSS_CONFIG.pd.hom_dim
    serialized = []
    for case in cases:
        history = []
        for frame in case["history"]:
            entry = {
                "step": int(frame["step"]),
                "loss": float(frame["loss"]),
                "pred": [np.asarray(p).copy() for p in frame["pred"]],
                "pts": np.asarray(frame["pts"]).copy(),
                "view_pts": np.asarray(frame["view_pts"]).copy(),
                "breakdown": _serialize_value(frame.get("breakdown", {})),
            }
            if "label" in frame:
                entry["label"] = frame["label"]
            history.append(entry)
        serialized.append(
            {
                "name": case["name"],
                "target_diags": [
                    _to_numpy(case["target_diags"], d).copy() for d in range(hom_dim)
                ],
                "target_pts": np.asarray(case["target_pts"]).copy(),
                "history": history,
            }
        )
    return serialized

def tda_weight(step: int) -> float:
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
    for dim in range(LOSS_CONFIG.pd.hom_dim):
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

def _make_history_frame(step, pred_pts, pred_diags, breakdown, view_pts):
    hom_dim = LOSS_CONFIG.pd.hom_dim
    return {
        "step": step,
        "loss": _scalar(breakdown["total"]),
        "pred": [_to_numpy(pred_diags, d).copy() for d in range(hom_dim)],
        "pts": pred_pts.detach().cpu().numpy().copy(),
        "view_pts": np.asarray(view_pts, dtype=float).copy(),
        "breakdown": _serialize_value(breakdown),
    }

def _logged_protein_indices(n_proteins: int, every_nth: int) -> list[int]:
    stride = max(1, every_nth)
    return list(range(0, n_proteins, stride))

def show_history(cases, title, *, show: bool | None = None, record: bool = True):
    if not cases or not cases[0]["history"]:
        return

    if show is None:
        show = SHOW_PLOTS

    if record:
        cases = _serialize_plot_cases(cases)
        _PLOT_GROUPS.append({"title": title, "cases": cases})

    if not show:
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
    active_homs = range(LOSS_CONFIG.pd.hom_dim)

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

        for dim in active_homs:
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
        frame_label = frame.get("label")
        if frame_label is None:
            frame_label = f"step {frame['step']}"
        step_text.set_text(
            f"{title}{protein_msg}  frame {step_idx + 1}/{n_steps}  "
            f"{frame_label}  loss={frame['loss']:.4f}  "
            f"{_tda_terms_msg(frame.get('breakdown', {}))}"
            f"{_extra_loss_msg(frame.get('breakdown', {}))}"
        )
        fig.canvas.draw_idle()

    def show(step_idx, protein_idx):
        clear_dynamic()
        draw_case(step_idx, protein_idx)

    go = _attach_controls(fig, show, n_steps, n_proteins=n_proteins)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, wspace=0.25)
    go(0, 0)
    plt.show(block=True)
