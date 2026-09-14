from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LRScheduler, StepLR
from tqdm import tqdm

from proteintda.config import LOSS_CONFIG, RUN_CONFIG
from proteintda.shared.checkpoint import (
    clear_fold_checkpoint,
    fold_checkpoint_path,
    fold_curves_path,
    fold_history_path,
    history_row,
    load_fold_checkpoint,
    save_fold_checkpoint,
    write_history_csv,
    write_metric_curves,
)
from proteintda.utils.conversions import SideChainAtom
from proteintda.utils.dataset import make_loader, set_seed


@dataclass(frozen=True)
class Backbone:
    name: str
    build_runner: Callable[..., Any]
    build_loss_fn: Callable[..., Any]
    train_kwargs: Mapping[str, Any] = field(default_factory=dict)
    eval_kwargs: Mapping[str, Any] = field(default_factory=dict)
    runner_kwargs: Mapping[str, Any] = field(default_factory=dict)


def prepare_vpd_kernels():
    from proteintda.config import HEAT_RFF_CONFIG
    from proteintda.tda.vpd_kernels import create_vpd_kernels

    print("Preparing VPD kernels...", flush=True)
    return create_vpd_kernels(LOSS_CONFIG, HEAT_RFF_CONFIG)


def tda_atom_from_config(loss_config=LOSS_CONFIG) -> SideChainAtom:
    return (
        SideChainAtom.CA
        if str(loss_config.tda.get("atom", "CB")).upper() == "CA"
        else SideChainAtom.CB
    )


def build_lr_scheduler(optimizer: torch.optim.Optimizer) -> LRScheduler | None:
    sched_cfg = RUN_CONFIG.training.get("scheduler", {})
    if not sched_cfg.get("enabled", False):
        return None
    return StepLR(
        optimizer,
        step_size=int(sched_cfg.get("step_size", 5)),
        gamma=float(sched_cfg.get("gamma", 0.9)),
    )


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


_METRIC_KEYS = ("plddt", "tm_score")


def _format_metrics_line(name: str, metrics: dict[str, float]) -> str:
    loss_keys: list[str] = []
    if "total" in metrics:
        loss_keys.append("total")
    for key in sorted(metrics):
        if key in _METRIC_KEYS or key in loss_keys:
            continue
        loss_keys.append(key)

    loss_str = "  ".join(f"{key}={metrics[key]:.4f}" for key in loss_keys if key in metrics)
    metric_str = "  ".join(f"{key}={metrics[key]:.4f}" for key in _METRIC_KEYS if key in metrics)
    return f"  {name}:  loss: {loss_str}  metrics: {metric_str}"


def format_epoch_metrics(
    *,
    epoch: int,
    epochs: int,
    fold: int,
    n_splits: int,
    train: dict[str, float],
    val: dict[str, float],
    lr: float | None = None,
) -> str:
    header = f"epoch {epoch}/{epochs}  fold {fold + 1}/{n_splits}"
    if lr is not None:
        header += f"  lr={lr:.6g}"
    lines = [header, _format_metrics_line("train", train), _format_metrics_line("val", val)]
    return "\n".join(lines) + "\n"


def write_log_file(
    log_file: Path,
    fold_plddt_scores: list[float],
    fold_tm_scores: list[float],
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        for fold_idx, (plddt, tm) in enumerate(zip(fold_plddt_scores, fold_tm_scores), start=1):
            handle.write(f"fold {fold_idx}: mean_plddt={plddt:.4f} mean_tm={tm:.4f}\n")
        handle.write(
            f"mean_plddt mean={np.mean(fold_plddt_scores):.4f} var={np.var(fold_plddt_scores):.4f}\n"
        )
        handle.write(
            f"mean_tm mean={np.mean(fold_tm_scores):.4f} var={np.var(fold_tm_scores):.4f}\n"
        )


def train_epoch(
    runner,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    *,
    epoch: int = 0,
    use_amp: bool = False,
    grad_clip_norm: float | None = 1.0,
    scaler: torch.amp.GradScaler,
    **batch_kwargs,
) -> dict[str, float]:
    totals = defaultdict(float)
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch_totals, batch_n = runner.run_batch(
            batch,
            loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            backward=True,
            include_loss=True,
            include_metrics=False,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
            epoch=epoch,
            **batch_kwargs,
        )
        for key, value in batch_totals.items():
            totals[key] += value
        n += batch_n

    if n == 0:
        return dict(totals)
    return {key: value / n for key, value in totals.items()}


def evaluate_loader(
    runner,
    loader,
    loss_fn=None,
    *,
    include_loss: bool = True,
    **batch_kwargs,
) -> dict[str, float]:
    totals = defaultdict(float)
    n = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        batch_totals, batch_n = runner.run_batch(
            batch,
            loss_fn,
            backward=False,
            include_loss=include_loss and loss_fn is not None,
            include_metrics=True,
            **batch_kwargs,
        )
        for key, value in batch_totals.items():
            totals[key] += value
        n += batch_n

    if n == 0:
        return dict(totals)
    return {key: value / n for key, value in totals.items()}


def run_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    proteins: list,
    device: torch.device,
    backbone: Backbone,
    n_splits: int,
    loss_fn=None,
    train_kwargs: Mapping[str, Any] | None = None,
    eval_kwargs: Mapping[str, Any] | None = None,
    **runner_kwargs,
) -> tuple[float, float]:
    training = RUN_CONFIG.training
    set_seed(training.seed + fold)
    train = loss_fn is not None
    train_kwargs = dict(train_kwargs or {})
    eval_kwargs = dict(eval_kwargs or {})

    print(f"Fold {fold + 1}/{n_splits}: loading {backbone.name} on {device}...")
    runner = backbone.build_runner(device, train=train, **runner_kwargs)
    test_loader = make_loader(
        [proteins[i] for i in test_idx],
        training.batch_size,
        shuffle=False,
    )

    if not train:
        metrics = evaluate_loader(
            runner,
            test_loader,
            loss_fn=None,
            include_loss=False,
            **eval_kwargs,
        )
        print(
            f"fold {fold + 1}/{n_splits}  "
            f"mean_plddt={metrics['plddt']:.4f}  mean_tm={metrics['tm_score']:.4f}"
        )
        return metrics["plddt"], metrics["tm_score"]

    trainable, total = runner.trainable_parameter_count
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    backbone_key = str(RUN_CONFIG.runtime.get("backbone", "minifold")).lower()
    ckpt_path = fold_checkpoint_path(backbone_key, fold)
    history_path = fold_history_path(backbone_key, fold)
    curves_path = fold_curves_path(backbone_key, fold)

    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.25, random_state=training.seed + fold
    )
    train_proteins = [proteins[i] for i in train_idx]
    val_proteins = [proteins[i] for i in val_idx]
    fold_rng = np.random.default_rng(training.seed + fold)

    optimizer = torch.optim.AdamW(
        [p for p in runner.model.parameters() if p.requires_grad],
        lr=training.lr,
        weight_decay=training.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=training.amp and runner.device.type == "cuda"
    )
    best_model_weights = None
    max_val_tm = 0.0
    patience = 0
    history: list[dict[str, Any]] = []
    start_epoch = 0
    val_subset_idx: list[int] | None = None

    if ckpt_path.is_file():
        ckpt = load_fold_checkpoint(ckpt_path, map_location="cpu")
        runner.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        best_model_weights = ckpt.get("best_model_state")
        max_val_tm = float(ckpt.get("max_val_tm", 0.0))
        patience = int(ckpt.get("patience", 0))
        history = list(ckpt.get("history", []))
        start_epoch = int(ckpt["epoch"])
        if ckpt.get("val_subset_idx") is not None:
            val_subset_idx = [int(i) for i in ckpt["val_subset_idx"]]
        if ckpt.get("rng_state") is not None:
            fold_rng.bit_generator.state = ckpt["rng_state"]
        print(
            f"Resuming fold {fold + 1}/{n_splits} from epoch {start_epoch + 1} "
            f"({ckpt_path})",
            flush=True,
        )

    if training.val_proteins_per_epoch is not None:
        if val_subset_idx is None:
            n = min(int(training.val_proteins_per_epoch), len(val_proteins))
            val_subset_idx = fold_rng.choice(
                len(val_proteins), size=n, replace=False
            ).tolist()
        val_for_loader = [val_proteins[i] for i in val_subset_idx]
    else:
        val_for_loader = val_proteins

    val_loader = make_loader(
        val_for_loader,
        training.batch_size,
        shuffle=False,
    )

    training_done = start_epoch >= training.epochs or patience > training.patience
    if training_done and start_epoch > 0:
        print(
            f"Fold {fold + 1}/{n_splits}: training already finished; running test eval.",
            flush=True,
        )

    for epoch in range(start_epoch, training.epochs):
        if patience > training.patience:
            break
        train_loader = make_loader(
            train_proteins,
            training.batch_size,
            shuffle=True,
            max_proteins=training.train_proteins_per_epoch,
            rng=fold_rng,
        )
        metrics = train_epoch(
            runner,
            train_loader,
            optimizer,
            loss_fn,
            epoch=epoch,
            use_amp=training.amp,
            grad_clip_norm=training.grad_clip_norm,
            scaler=scaler,
            **train_kwargs,
        )
        val_metrics = evaluate_loader(
            runner,
            val_loader,
            loss_fn=loss_fn,
            include_loss=True,
            **eval_kwargs,
        )
        if val_metrics.get("tm_score", 0.0) > max_val_tm:
            max_val_tm = val_metrics["tm_score"]
            patience = 0
            best_model_weights = runner.snapshot_state_dict()
        else:
            patience += 1

        lr = _current_lr(optimizer)
        print(
            format_epoch_metrics(
                epoch=epoch + 1,
                epochs=training.epochs,
                fold=fold,
                n_splits=n_splits,
                train=metrics,
                val=val_metrics,
                lr=lr,
            )
        )

        history.append(
            history_row(epoch=epoch + 1, lr=lr, train=metrics, val=val_metrics)
        )
        write_history_csv(history_path, history)
        write_metric_curves(curves_path, history, fold=fold)
        save_fold_checkpoint(
            ckpt_path,
            fold=fold,
            epoch=epoch + 1,
            model_state=runner.snapshot_state_dict(),
            best_model_state=best_model_weights,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            scaler_state=scaler.state_dict(),
            max_val_tm=max_val_tm,
            patience=patience,
            history=history,
            rng_state=fold_rng.bit_generator.state,
            val_subset_idx=val_subset_idx,
        )

        if scheduler is not None:
            scheduler.step()

        if patience > training.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_weights is not None:
        runner.load_state_dict(best_model_weights)

    test_metrics = evaluate_loader(
        runner,
        test_loader,
        loss_fn=None,
        include_loss=False,
        **eval_kwargs,
    )
    print(
        f"fold {fold + 1}/{n_splits}  "
        f"mean_plddt={test_metrics['plddt']:.4f}  mean_tm={test_metrics['tm_score']:.4f}"
    )
    clear_fold_checkpoint(ckpt_path)
    return test_metrics["plddt"], test_metrics["tm_score"]
