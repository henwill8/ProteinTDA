"""MiniFold k-fold training, evaluation, and logging."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from proteintda.config import CONFIG_OF, HEAT_RFF_CONFIG, LOSS_CONFIG, RUN_CONFIG
from proteintda.minifold.loss import MiniFoldLoss
from proteintda.utils.dataset import make_loader, set_seed
from proteintda.minifold.runner import MiniFoldRunner
from proteintda.tda.vpd_kernels import create_vpd_kernels


def build_loss_fn() -> MiniFoldLoss:
    h0rff, h1rff = create_vpd_kernels(LOSS_CONFIG, HEAT_RFF_CONFIG)
    return MiniFoldLoss(CONFIG_OF, loss_config=LOSS_CONFIG, h0rff=h0rff, h1rff=h1rff)


def format_epoch_metrics(
    *,
    epoch: int,
    epochs: int,
    fold: int,
    n_splits: int,
    train: dict[str, float],
    val_plddt: float,
    val_tm: float,
) -> str:
    grad_keys = ("fold_grad_norm", "topo_grad_norm")
    ordered_loss: list[str] = []
    if train is not None and "total" in train:
        ordered_loss.append("total")
    for key in sorted(train):
        if key in grad_keys or key in ordered_loss:
            continue
        ordered_loss.append(key)

    def fmt_items(keys) -> str:
        return "  ".join(f"{key}={train[key]:.4f}" for key in keys if key in train)

    lines = [f"epoch {epoch}/{epochs}  fold {fold + 1}/{n_splits}"]
    if ordered_loss:
        lines.append(f"  train loss: {fmt_items(ordered_loss)}")
    if any(key in train for key in grad_keys):
        lines.append(f"  train grad: {fmt_items(grad_keys)}")
    lines.append(f"  val:        plddt={val_plddt:.4f}  tm={val_tm:.4f}")
    return "\n".join(lines)


def train_epoch(
    runner: MiniFoldRunner,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MiniFoldLoss,
    *,
    train_recycles: int | None = None,
    randomize_recycles: bool = True,
    use_amp: bool = False,
    grad_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    totals = defaultdict(float)
    n = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and runner.device.type == "cuda")

    for batch in tqdm(loader, desc="train", leave=False):
        batch_totals, batch_n = runner.training_step(
            batch,
            optimizer,
            loss_fn,
            scaler,
            train_recycles=train_recycles,
            randomize_recycles=randomize_recycles,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
        )
        for key, value in batch_totals.items():
            totals[key] += value
        n += batch_n

    if n == 0:
        return dict(totals)
    return {key: value / n for key, value in totals.items()}


def evaluate_loader(
    runner: MiniFoldRunner,
    loader,
    *,
    num_recycling: int,
) -> tuple[float, float]:
    plddt_scores: list[float] = []
    tm_scores: list[float] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch_plddt, batch_tm = runner.evaluation_step(batch, num_recycling=num_recycling)
        plddt_scores.extend(batch_plddt)
        tm_scores.extend(batch_tm)

    return float(np.mean(plddt_scores)), float(np.mean(tm_scores))


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


def run_baseline_fold(
    fold: int,
    _train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    proteins: list,
    cache_dir: Path,
    device: torch.device,
    model_size: str,
    n_splits: int,
) -> tuple[float, float]:
    runtime = RUN_CONFIG.runtime
    training = RUN_CONFIG.training
    set_seed(training.seed + fold)

    print(f"Fold {fold + 1}/{n_splits}: loading MiniFold on {device}...")
    runner = MiniFoldRunner(
        cache_dir,
        model_size=model_size,
        device=device,
    )

    test_proteins = [proteins[i] for i in test_idx]
    test_loader = make_loader(test_proteins, training.batch_size, shuffle=False)
    plddt, tm = evaluate_loader(runner, test_loader, num_recycling=runtime.infer_recycles)
    print(f"fold {fold + 1}/{n_splits}  mean_plddt={plddt:.4f}  mean_tm={tm:.4f}")
    return plddt, tm


def run_train_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    proteins: list,
    cache_dir: Path,
    device: torch.device,
    model_size: str,
    loss_fn: MiniFoldLoss,
    n_splits: int,
) -> tuple[float, float]:
    runtime = RUN_CONFIG.runtime
    training = RUN_CONFIG.training
    set_seed(training.seed + fold)

    print(f"Fold {fold + 1}/{n_splits}: loading fresh MiniFold on {device}...")
    runner = MiniFoldRunner(
        cache_dir,
        model_size=model_size,
        device=device,
        train=True,
        unfreeze_fold_blocks=training.unfreeze_fold_blocks,
        unfreeze_structure_module=training.unfreeze_structure_module,
    )
    trainable, total = runner.trainable_parameter_count
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    train_idx, val_idx = train_test_split(train_idx, test_size=0.25)
    train_proteins = [proteins[i] for i in train_idx]
    val_proteins = [proteins[i] for i in val_idx]
    test_proteins = [proteins[i] for i in test_idx]

    fold_rng = np.random.default_rng(training.seed + fold)
    val_loader = make_loader(
        val_proteins,
        training.batch_size,
        shuffle=False,
        max_proteins=training.val_proteins_per_epoch,
        rng=fold_rng,
    )
    test_loader = make_loader(test_proteins, training.batch_size, shuffle=False)

    optimizer = runner.build_optimizer(
        base_lr=training.base_lr,
        struct_lr=training.struct_lr,
    )
    best_model_weights = None
    max_val_tm = 0.0
    patience = 0

    for epoch in range(training.epochs):
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
            train_recycles=training.train_recycles,
            randomize_recycles=training.randomize_recycles,
            use_amp=training.amp,
            grad_clip_norm=training.grad_clip_norm,
        )

        # TODO: switch to checking on the loss instead of the metric?
        val_plddt_score, val_tm_score = evaluate_loader(
            runner,
            val_loader,
            num_recycling=runtime.infer_recycles,
        )
        if val_tm_score > max_val_tm:
            max_val_tm = val_tm_score
            patience = 0
            best_model_weights = runner.snapshot_state_dict()
        else:
            patience += 1

        print(
            format_epoch_metrics(
                epoch=epoch + 1,
                epochs=training.epochs,
                fold=fold,
                n_splits=n_splits,
                train=metrics,
                val_plddt=val_plddt_score,
                val_tm=val_tm_score,
            )
        )

        if patience > training.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_weights is not None:
        runner.load_state_dict(best_model_weights)

    plddt_score, tm_score = evaluate_loader(
        runner,
        test_loader,
        num_recycling=runtime.infer_recycles,
    )
    print(f"fold {fold + 1}/{n_splits}  mean_plddt={plddt_score:.4f}  mean_tm={tm_score:.4f}")
    return plddt_score, tm_score
