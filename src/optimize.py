import copy
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold, train_test_split

from proteintda.config import CONFIG_OF, HEAT_RFF_CONFIG, LOSS_CONFIG, RUN_CONFIG
from proteintda.minifold.loss import MiniFoldLoss
from proteintda.minifold.pipeline import evaluate_loader, train_epoch
from proteintda.minifold.runner import MiniFoldRunner
from proteintda.tda.vpd_kernels import create_vpd_kernels
from proteintda.utils.dataset import load_dataset, make_loader, set_seed

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def resolve_device() -> torch.device:
    device = RUN_CONFIG.runtime.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def suggest_params(trial: optuna.Trial) -> dict:
    optuna_cfg = RUN_CONFIG.optuna
    params: dict = {}

    if optuna_cfg.tune_wasserstein:
        params["w_wasserstein_h0"] = trial.suggest_float(
            "w_wasserstein_h0", optuna_cfg.w_min, optuna_cfg.w_max, log=True
        )
        params["w_wasserstein_h1"] = trial.suggest_float(
            "w_wasserstein_h1", optuna_cfg.w_min, optuna_cfg.w_max, log=True
        )

    if optuna_cfg.tune_vpd:
        params["w_vpd_h0"] = trial.suggest_float(
            "w_vpd_h0", optuna_cfg.w_min, optuna_cfg.w_max, log=True
        )
        params["w_vpd_h1"] = trial.suggest_float(
            "w_vpd_h1", optuna_cfg.w_min, optuna_cfg.w_max, log=True
        )

    return params


def build_loss_cfg(params: dict):
    loss_cfg = copy.deepcopy(LOSS_CONFIG)

    if RUN_CONFIG.optuna.tune_wasserstein:
        loss_cfg.wasserstein_h0.weight = params["w_wasserstein_h0"]
        loss_cfg.wasserstein_h1.weight = params["w_wasserstein_h1"]

    if RUN_CONFIG.optuna.tune_vpd:
        loss_cfg.vpd_h0.weight = params["w_vpd_h0"]
        loss_cfg.vpd_h1.weight = params["w_vpd_h1"]

    return loss_cfg


def build_heat_cfg(params: dict):
    heat_cfg = copy.deepcopy(HEAT_RFF_CONFIG)

    if RUN_CONFIG.optuna.tune_vpd:
        heat_cfg.h0rff.t = params["t_h0"]
        heat_cfg.h1rff.t = params["t_h1"]

    return heat_cfg


def build_loss_fn(loss_cfg, heat_cfg) -> MiniFoldLoss:
    h0rff = h1rff = None
    if loss_cfg.vpd_h0.enabled or loss_cfg.vpd_h1.enabled:
        h0rff, h1rff = create_vpd_kernels(loss_cfg, heat_cfg)
    return MiniFoldLoss(CONFIG_OF, loss_config=loss_cfg, h0rff=h0rff, h1rff=h1rff)


def run_optuna_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    proteins: list,
    device: torch.device,
    cache_dir: Path,
    loss_fn: MiniFoldLoss,
    trial: optuna.Trial,
) -> float:
    training = RUN_CONFIG.training
    runtime = RUN_CONFIG.runtime
    set_seed(training.seed + fold)

    runner = MiniFoldRunner(
        cache_dir,
        model_size=runtime.model_size,
        device=device,
        train=True,
        unfreeze_fold_blocks=training.unfreeze_fold_blocks,
        unfreeze_structure_module=training.unfreeze_structure_module,
    )

    train_idx, val_idx = train_test_split(train_idx, test_size=0.25)
    train_proteins = [proteins[i] for i in train_idx]
    val_proteins = [proteins[i] for i in val_idx]
    test_proteins = [proteins[i] for i in test_idx]

    fold_rng = np.random.default_rng(training.seed + fold)
    val_loader = make_loader(val_proteins, training.batch_size, shuffle=False)
    test_loader = make_loader(test_proteins, training.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        [p for p in runner.model.parameters() if p.requires_grad],
        lr=training.lr,
        weight_decay=training.weight_decay,
    )
    best_model_weights = None
    max_val_tm = 0.0
    patience = 0
    report_num = 0

    for epoch in range(training.epochs):
        report_num += 1
        train_loader = make_loader(
            train_proteins,
            training.batch_size,
            shuffle=True,
            rng=fold_rng,
        )
        train_epoch(
            runner,
            train_loader,
            optimizer,
            loss_fn,
            train_recycles=training.train_recycles,
            randomize_recycles=training.randomize_recycles,
            use_amp=training.amp,
            grad_clip_norm=training.grad_clip_norm,
        )
        val_metrics = evaluate_loader(
            runner,
            val_loader,
            loss_fn,
            num_recycling=runtime.infer_recycles,
        )
        val_tm_score = val_metrics["tm"]

        best_val_tm = max(max_val_tm, val_tm_score)
        trial.report(best_val_tm, report_num)
        if trial.should_prune():
            del runner
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise optuna.TrialPruned()

        if val_tm_score > max_val_tm:
            max_val_tm = val_tm_score
            patience = 0
            best_model_weights = runner.snapshot_state_dict()
        else:
            patience += 1

        if patience > training.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_weights is not None:
        runner.load_state_dict(best_model_weights)

    test_metrics = evaluate_loader(
        runner,
        test_loader,
        loss_fn=None,
        num_recycling=runtime.infer_recycles,
    )
    tm_score = test_metrics["tm"]

    del runner
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tm_score


def create_objective(device: torch.device, proteins: list):
    optuna_cfg = RUN_CONFIG.optuna
    training = RUN_CONFIG.training
    kfold = RUN_CONFIG.kfold

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        loss_cfg = build_loss_cfg(params)
        heat_cfg = build_heat_cfg(params)
        loss_fn = build_loss_fn(loss_cfg, heat_cfg)

        kf = KFold(n_splits=kfold.n_splits, shuffle=True, random_state=training.seed)
        fold_tm_scores: list[float] = []
        cache_dir = Path(RUN_CONFIG.runtime.minifold_cache_dir)

        for fold, (train_idx, test_idx) in enumerate(kf.split(proteins)):
            if fold >= optuna_cfg.n_folds:
                break
            tm_score = run_optuna_fold(
                fold,
                train_idx,
                test_idx,
                proteins=proteins,
                device=device,
                cache_dir=cache_dir,
                loss_fn=loss_fn,
                trial=trial,
            )
            fold_tm_scores.append(tm_score)

        print(fold_tm_scores)
        return float(np.mean(fold_tm_scores)) if fold_tm_scores else 0.0

    return objective


def main() -> int:
    optuna.logging.set_verbosity(optuna.logging.INFO)

    optuna_cfg = RUN_CONFIG.optuna
    training = RUN_CONFIG.training
    if not optuna_cfg.tune_wasserstein and not optuna_cfg.tune_vpd:
        raise ValueError("Enable tune_wasserstein and/or tune_vpd")

    device = resolve_device()
    set_seed(training.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tuned = []
    if optuna_cfg.tune_wasserstein:
        tuned.append("wasserstein")
    if optuna_cfg.tune_vpd:
        tuned.append("vpd")
    print(f"Optuna tuning: {', '.join(tuned)}")

    proteins = load_dataset()

    study = optuna.create_study(
        study_name=optuna_cfg.study_name,
        storage=optuna_cfg.storage,
        sampler=TPESampler(seed=training.seed),
        pruner=SuccessiveHalvingPruner(),
        direction="maximize",
        load_if_exists=True,
    )

    study.optimize(
        create_objective(device, proteins),
        show_progress_bar=True,
        n_trials=optuna_cfg.n_trials,
    )

    log_file = Path(RUN_CONFIG.logging.optuna_log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("==================== New Study Results ====================\n")
        handle.write("========== Best Values ==========\n")
        handle.write(f"best_mean_val_tm={study.best_value:.6f}\n")
        for key, value in study.best_params.items():
            handle.write(f"{key}={value:.8g}\n")
        handle.write("========== Parameter Importances ==========\n")
        handle.write(str(optuna.importance.get_param_importances(study)))

    print(f"Wrote results to {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
