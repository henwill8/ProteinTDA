"""Resumable k-fold evaluation and training."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from proteintda.config import RUN_CONFIG


def _checkpoint_path(baseline: bool) -> Path:
    name = "baseline" if baseline else "finetune"
    return Path(RUN_CONFIG.kfold.checkpoint_dir) / f"{name}.json"


def _run_config(*, num_proteins: int) -> dict[str, Any]:
    data = RUN_CONFIG.data
    training = RUN_CONFIG.training
    kfold = RUN_CONFIG.kfold
    runtime = RUN_CONFIG.runtime
    sched = training.get("scheduler", {})
    return {
        "baseline": runtime.baseline,
        "n_splits": kfold.n_splits,
        "seed": training.seed,
        "num_proteins": num_proteins,
        "casp_version": data.casp_version,
        "casp_thinning": data.casp_thinning,
        "max_proteins": data.max_proteins,
        "model_size": runtime.model_size,
        "infer_recycles": runtime.infer_recycles,
        "lr": training.lr,
        "scheduler": dict(sched) if sched else None,
    }


class KFoldRunner:
    """Runs k-fold splits, skipping folds already stored in the checkpoint file."""

    def __init__(self, proteins: list) -> None:
        self.proteins = proteins
        self.baseline = RUN_CONFIG.runtime.baseline
        self.n_splits = RUN_CONFIG.kfold.n_splits
        self.seed = RUN_CONFIG.training.seed
        self.path = _checkpoint_path(self.baseline)
        self.run_config = _run_config(num_proteins=len(proteins))
        self._folds: dict[str, dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        saved_config = saved.get("run_config", {})
        saved_baseline = saved_config.get("baseline")
        if saved_baseline != self.baseline:
            raise ValueError(
                f"K-fold checkpoint at {self.path} was created with baseline={saved_baseline!r}, "
                f"but RUN_CONFIG.runtime.baseline is {self.baseline!r}. "
                "Delete the checkpoint to start a new run."
            )
        if saved_config != self.run_config:
            raise ValueError(
                f"K-fold checkpoint at {self.path} does not match the current run settings. Delete the previous checkpoint to start a new run."
            )
        self._folds = {str(k): v for k, v in saved.get("folds", {}).items()}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"run_config": self.run_config, "folds": self._folds},
                handle,
                indent=2,
            )

    def splits(self) -> list[tuple[int, np.ndarray, np.ndarray]]:
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        return [
            (fold, train_idx, test_idx)
            for fold, (train_idx, test_idx) in enumerate(kf.split(self.proteins))
        ]

    def run(
        self,
        fold_fn: Callable[[int, np.ndarray, np.ndarray], tuple[float, float]],
    ) -> tuple[list[float], list[float]]:
        for fold, train_idx, test_idx in self.splits():
            key = str(fold)
            if key in self._folds:
                result = self._folds[key]
                print(
                    f"fold {fold + 1}/{self.n_splits}  "
                    f"mean_plddt={result['mean_plddt']:.4f}  mean_tm={result['mean_tm']:.4f}  (cached)"
                )
                continue

            plddt, tm = fold_fn(fold, train_idx, test_idx)
            self._folds[key] = {"mean_plddt": plddt, "mean_tm": tm}
            self._save()

        plddt_scores = [self._folds[str(fold)]["mean_plddt"] for fold in range(self.n_splits)]
        tm_scores = [self._folds[str(fold)]["mean_tm"] for fold in range(self.n_splits)]
        return plddt_scores, tm_scores
