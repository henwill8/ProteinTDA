"""Mid-fold training checkpoints and epoch metric curves."""

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from proteintda.config import RUN_CONFIG

_METRIC_PLOT_KEYS = ("total", "tm_score", "plddt")


def fold_work_dir(backbone: str, fold: int, *, baseline: bool | None = None) -> Path:
    if baseline is None:
        baseline = bool(RUN_CONFIG.runtime.baseline)
    mode = "baseline" if baseline else "finetune"
    return Path(RUN_CONFIG.kfold.checkpoint_dir) / f"{backbone}_{mode}" / f"fold_{fold}"


def fold_checkpoint_path(backbone: str, fold: int) -> Path:
    return fold_work_dir(backbone, fold) / "checkpoint.pt"


def fold_history_path(backbone: str, fold: int) -> Path:
    return fold_work_dir(backbone, fold) / "history.csv"


def fold_curves_path(backbone: str, fold: int) -> Path:
    return fold_work_dir(backbone, fold) / "curves.png"


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save_fold_checkpoint(
    path: Path,
    *,
    fold: int,
    epoch: int,
    model_state: dict[str, torch.Tensor],
    best_model_state: dict[str, torch.Tensor] | None,
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any] | None,
    scaler_state: dict[str, Any],
    max_val_tm: float,
    patience: int,
    history: list[dict[str, Any]],
    rng_state: dict[str, Any],
    val_subset_idx: list[int] | None = None,
) -> None:
    _atomic_torch_save(
        {
            "fold": fold,
            "epoch": epoch,
            "model_state": model_state,
            "best_model_state": best_model_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "scaler_state": scaler_state,
            "max_val_tm": max_val_tm,
            "patience": patience,
            "history": history,
            "rng_state": rng_state,
            "val_subset_idx": val_subset_idx,
        },
        path,
    )


def load_fold_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def clear_fold_checkpoint(path: Path) -> None:
    if path.is_file():
        path.unlink()
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.is_file():
        tmp.unlink()


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in history:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_metric_curves(path: Path, history: list[dict[str, Any]], *, fold: int) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    plot_keys = [key for key in _METRIC_PLOT_KEYS if any(f"train_{key}" in row or f"val_{key}" in row for row in history)]
    if not plot_keys:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(plot_keys), figsize=(4.5 * len(plot_keys), 3.5), squeeze=False)
    for ax, key in zip(axes[0], plot_keys):
        train_y = [row.get(f"train_{key}") for row in history]
        val_y = [row.get(f"val_{key}") for row in history]
        if any(v is not None for v in train_y):
            ax.plot(epochs, train_y, label="train", marker="o", markersize=3)
        if any(v is not None for v in val_y):
            ax.plot(epochs, val_y, label="val", marker="o", markersize=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Fold {fold + 1}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def history_row(
    *,
    epoch: int,
    lr: float,
    train: dict[str, float],
    val: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {"epoch": epoch, "lr": lr}
    for key, value in train.items():
        row[f"train_{key}"] = float(value)
    for key, value in val.items():
        row[f"val_{key}"] = float(value)
    return row
