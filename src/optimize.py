"""Hard coded for LightRoseTTA currently."""

import argparse
import copy
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_collections as mlc
import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.trial import TrialState
from sklearn.model_selection import KFold, train_test_split

from proteintda.config import HEAT_RFF_CONFIG, LOSS_CONFIG, RUN_CONFIG, apply_set_overrides
from proteintda.lightrosetta.loss import LightRoseTTALoss
from proteintda.lightrosetta.runner import LightRoseTTARunner
from proteintda.shared.pipeline import (
    build_lr_scheduler,
    evaluate_loader,
    format_epoch_metrics,
    tda_atom_from_config,
    train_epoch,
)
from proteintda.tda.vpd_kernels import create_vpd_kernels
from proteintda.utils.dataset import load_dataset, make_loader, set_seed
from proteintda.utils.device import resolve_device


@dataclass(frozen=True)
class TuneSpec:
    """
      --tune key=kind,low,high
      --tune key=kind,v1|v2|...
    """

    key: str
    kind: str
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] | None = None


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_tune_spec(raw: str) -> TuneSpec:
    if "=" not in raw:
        raise ValueError(f"Tune spec must be key=kind,..., got {raw!r}")
    key, spec = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Empty tune key in {raw!r}")
    parts = [p.strip() for p in spec.split(",")]
    if not parts:
        raise ValueError(f"Empty tune spec for {key!r}")
    kind = parts[0].lower()
    if kind in {"float", "log", "uniform"}:
        if len(parts) != 3:
            raise ValueError(f"{kind} tune needs key={kind},low,high; got {raw!r}")
        low, high = float(parts[1]), float(parts[2])
        if low >= high:
            raise ValueError(f"Tune {key}: low must be < high ({low} >= {high})")
        return TuneSpec(key=key, kind="log" if kind == "log" else "float", low=low, high=high)
    if kind == "int":
        if len(parts) != 3:
            raise ValueError(f"int tune needs key=int,low,high; got {raw!r}")
        low_i, high_i = int(float(parts[1])), int(float(parts[2]))
        if low_i >= high_i:
            raise ValueError(f"Tune {key}: low must be < high ({low_i} >= {high_i})")
        return TuneSpec(key=key, kind="int", low=float(low_i), high=float(high_i))
    if kind == "categorical":
        if len(parts) != 2:
            raise ValueError(
                f"categorical tune needs key=categorical,v1|v2|...; got {raw!r}"
            )
        choices = tuple(_parse_value(c.strip()) for c in parts[1].split("|") if c.strip())
        if not choices:
            raise ValueError(f"Tune {key}: categorical needs at least one choice")
        return TuneSpec(key=key, kind="categorical", choices=choices)
    raise ValueError(
        f"Unknown tune kind {kind!r} in {raw!r}; use float, log, int, or categorical"
    )


def _root_and_parts(key: str) -> tuple[mlc.ConfigDict, list[str]]:
    parts = key.split(".")
    if not parts or any(not p for p in parts):
        raise ValueError(f"Invalid tune key {key!r}")
    if parts[0] == "loss":
        if len(parts) < 2:
            raise ValueError("loss tune key needs a field path")
        return LOSS_CONFIG, parts[1:]
    if parts[0] == "run":
        if len(parts) < 2:
            raise ValueError("run tune key needs a field path")
        return RUN_CONFIG, parts[1:]
    return RUN_CONFIG, parts


def _ensure_path(root: mlc.ConfigDict, parts: list[str]) -> None:
    node = root
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], mlc.ConfigDict):
            raise KeyError(f"Unknown config path segment {part!r} under tune target")
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f"Unknown config leaf {'.'.join(parts)!r}")


def validate_tune_specs(specs: list[TuneSpec]) -> None:
    seen: set[str] = set()
    for spec in specs:
        if spec.key in seen:
            raise ValueError(f"Duplicate --tune key {spec.key!r}")
        seen.add(spec.key)
        root, parts = _root_and_parts(spec.key)
        _ensure_path(root, parts)


def suggest_params(trial: optuna.Trial, specs: list[TuneSpec]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for spec in specs:
        name = spec.key
        if spec.kind == "float":
            params[name] = trial.suggest_float(name, float(spec.low), float(spec.high))
        elif spec.kind == "log":
            params[name] = trial.suggest_float(
                name, float(spec.low), float(spec.high), log=True
            )
        elif spec.kind == "int":
            params[name] = trial.suggest_int(name, int(spec.low), int(spec.high))
        elif spec.kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec.choices or ()))
        else:
            raise ValueError(f"Unhandled tune kind {spec.kind!r}")
    return params


def _set_path(root: mlc.ConfigDict, parts: list[str], value: Any) -> None:
    with root.unlocked():
        node = root
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value


def apply_params(params: dict[str, Any]) -> mlc.ConfigDict:
    """Apply tuned values. Returns a LOSS_CONFIG deepcopy used to build the loss."""
    loss_cfg = copy.deepcopy(LOSS_CONFIG)
    for key, value in params.items():
        root, parts = _root_and_parts(key)
        if root is LOSS_CONFIG:
            _set_path(loss_cfg, parts, value)
        else:
            _set_path(RUN_CONFIG, parts, value)
    return loss_cfg


def build_loss_fn(loss_cfg) -> LightRoseTTALoss:
    h0rff = h1rff = h2rff = None
    if loss_cfg.tda.enabled:
        terms = loss_cfg.tda.terms
        needs_vpd = any(
            terms[name].enabled for name in terms if str(name).startswith("vpd_")
        )
        if needs_vpd:
            h0rff, h1rff, h2rff = create_vpd_kernels(
                loss_cfg, copy.deepcopy(HEAT_RFF_CONFIG)
            )
    return LightRoseTTALoss(
        loss_cfg,
        h0rff=h0rff,
        h1rff=h1rff,
        h2rff=h2rff,
        tda_atom=tda_atom_from_config(loss_cfg),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna search over arbitrary ProteinTDA config leaves"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Fixed dotted override (same as main.py).",
    )
    parser.add_argument(
        "--tune",
        dest="tunes",
        action="append",
        default=[],
        metavar="KEY=KIND,...",
        help=(
            "Searchable leaf. Examples: loss.tda.weight=log,1e-6,1 | "
            "training.lr=float,1e-5,1e-3 | loss.tda.atom=categorical,CA|CB"
        ),
    )
    return parser.parse_args(argv)


def run_optuna_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    proteins: list,
    device: torch.device,
    loss_fn: LightRoseTTALoss,
    trial: optuna.Trial,
) -> float:
    training = RUN_CONFIG.training
    base_seed = int(training.seed) + int(fold)
    if bool(RUN_CONFIG.optuna.get("vary_trial_seed", False)):
        base_seed += 10007 * int(trial.number)
    set_seed(base_seed)

    runner = LightRoseTTARunner(device, train=True)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.25, random_state=base_seed
    )
    train_proteins = [proteins[i] for i in train_idx]
    val_proteins = [proteins[i] for i in val_idx]
    test_proteins = [proteins[i] for i in test_idx]

    fold_rng = np.random.default_rng(base_seed)
    if training.val_proteins_per_epoch is not None:
        n_val = min(int(training.val_proteins_per_epoch), len(val_proteins))
        val_proteins = [
            val_proteins[i]
            for i in fold_rng.choice(len(val_proteins), size=n_val, replace=False)
        ]
    val_loader = make_loader(val_proteins, training.batch_size, shuffle=False)
    test_loader = make_loader(test_proteins, training.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        [p for p in runner.model.parameters() if p.requires_grad],
        lr=training.lr,
        weight_decay=training.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=training.amp and device.type == "cuda"
    )

    best_model_weights = None
    max_val_tm = 0.0
    patience = 0
    n_splits = RUN_CONFIG.kfold.n_splits
    print(f"trial {trial.number}  fold {fold + 1}/{n_splits}", flush=True)

    for epoch in range(training.epochs):
        train_loader = make_loader(
            train_proteins,
            training.batch_size,
            shuffle=True,
            max_proteins=training.train_proteins_per_epoch,
            rng=fold_rng,
        )
        train_metrics = train_epoch(
            runner,
            train_loader,
            optimizer,
            loss_fn,
            epoch=epoch,
            use_amp=training.amp,
            grad_clip_norm=training.grad_clip_norm,
            scaler=scaler,
        )
        val_metrics = evaluate_loader(
            runner,
            val_loader,
            loss_fn,
            include_loss=True,
        )
        val_tm = float(val_metrics.get("tm_score", 0.0))
        lr = float(optimizer.param_groups[0]["lr"])
        print(
            format_epoch_metrics(
                epoch=epoch + 1,
                epochs=training.epochs,
                fold=fold,
                n_splits=n_splits,
                train=train_metrics,
                val=val_metrics,
                lr=lr,
            ),
            flush=True,
        )

        trial.report(val_tm, epoch + 1)
        if trial.should_prune():
            print(
                f"Pruned trial {trial.number} fold {fold + 1} "
                f"at epoch {epoch + 1}  val_tm={val_tm:.4f}  best_val_tm={max_val_tm:.4f}",
                flush=True,
            )
            del runner
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise optuna.TrialPruned()

        if val_tm > max_val_tm:
            max_val_tm = val_tm
            patience = 0
            best_model_weights = runner.snapshot_state_dict()
        else:
            patience += 1

        if scheduler is not None:
            scheduler.step()
        if patience > training.patience:
            print(f"Early stopping at epoch {epoch + 1}", flush=True)
            break

    if best_model_weights is not None:
        runner.load_state_dict(best_model_weights)

    test_metrics = evaluate_loader(
        runner,
        test_loader,
        loss_fn=None,
        include_loss=False,
    )
    tm_score = float(test_metrics.get("tm_score", 0.0))
    print(
        f"trial {trial.number}  fold {fold + 1}/{n_splits}  "
        f"test tm={tm_score:.4f}  best_val_tm={max_val_tm:.4f}",
        flush=True,
    )
    del runner
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tm_score


def create_objective(device: torch.device, proteins: list, specs: list[TuneSpec]):
    optuna_cfg = RUN_CONFIG.optuna
    training = RUN_CONFIG.training
    kfold = RUN_CONFIG.kfold

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, specs)
        print(f"trial {trial.number} params={params}", flush=True)
        loss_cfg = apply_params(params)
        loss_fn = build_loss_fn(loss_cfg)

        fold_seed = int(training.seed)
        if bool(optuna_cfg.get("vary_trial_seed", False)):
            fold_seed += 10007 * int(trial.number)
        kf = KFold(
            n_splits=kfold.n_splits,
            shuffle=True,
            random_state=fold_seed,
        )
        fold_scores: list[float] = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(proteins)):
            if fold >= int(optuna_cfg.n_folds):
                break
            fold_scores.append(
                run_optuna_fold(
                    fold,
                    train_idx,
                    test_idx,
                    proteins=proteins,
                    device=device,
                    loss_fn=loss_fn,
                    trial=trial,
                )
            )
            print(
                f"trial {trial.number}: fold scores so far = "
                f"{[round(s, 4) for s in fold_scores]}",
                flush=True,
            )

        mean_tm = float(np.mean(fold_scores)) if fold_scores else 0.0
        print(f"trial {trial.number}: mean test tm={mean_tm:.4f}", flush=True)
        return mean_tm

    return objective


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.overrides:
        apply_set_overrides(args.overrides)
    if not args.tunes:
        raise ValueError(
            "Pass at least one --tune KEY=KIND,... "
        )
    specs = [parse_tune_spec(raw) for raw in args.tunes]
    validate_tune_specs(specs)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna_cfg = RUN_CONFIG.optuna
    training = RUN_CONFIG.training

    if str(RUN_CONFIG.runtime.get("backbone", "")).lower() != "lightrosetta":
        raise ValueError("optimize.py currently expects runtime.backbone=lightrosetta")

    device = resolve_device()
    set_seed(training.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    proteins = load_dataset()
    sampler_name = str(optuna_cfg.get("sampler", "tpe")).lower()
    pruner_name = str(optuna_cfg.get("pruner", "successive_halving")).lower()
    n_repeats = int(optuna_cfg.get("n_repeats", 1))
    vary_trial_seed = bool(optuna_cfg.get("vary_trial_seed", False))
    print(
        f"Optuna study={optuna_cfg.study_name!r} device={device} "
        f"n_proteins={len(proteins)} n_trials={optuna_cfg.n_trials} "
        f"sampler={sampler_name} pruner={pruner_name} n_repeats={n_repeats} "
        f"vary_trial_seed={vary_trial_seed} "
        f"tuning={[s.key for s in specs]}",
        flush=True,
    )

    if sampler_name == "random":
        sampler = RandomSampler(seed=training.seed)
    elif sampler_name == "tpe":
        sampler = TPESampler(seed=training.seed)
    else:
        raise ValueError(f"Unknown optuna.sampler={sampler_name!r}; use tpe or random")

    if pruner_name in {"none", "nop", "off"}:
        pruner = NopPruner()
    elif pruner_name in {"successive_halving", "sha"}:
        pruner = SuccessiveHalvingPruner()
    elif pruner_name == "median":
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    else:
        raise ValueError(
            f"Unknown optuna.pruner={pruner_name!r}; use none, successive_halving, or median"
        )

    study = optuna.create_study(
        study_name=str(optuna_cfg.study_name),
        storage=str(optuna_cfg.storage),
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    all_categorical = all(spec.kind == "categorical" for spec in specs)
    if n_repeats > 1 and all_categorical and specs:
        axes = [
            [(spec.key, choice) for choice in (spec.choices or ())]
            for spec in specs
        ]
        enqueued = 0
        for combo in itertools.product(*axes):
            params = dict(combo)
            for _ in range(n_repeats):
                study.enqueue_trial(params)
                enqueued += 1
        n_trials = max(int(optuna_cfg.n_trials), enqueued)
        print(f"Enqueued {enqueued} balanced categorical trials", flush=True)
    else:
        n_trials = int(optuna_cfg.n_trials)

    study.optimize(
        create_objective(device, proteins, specs),
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(RuntimeError, torch.cuda.OutOfMemoryError),
    )

    complete = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    by_params: dict[str, list[float]] = defaultdict(list)
    for trial in complete:
        label = ", ".join(f"{k}={v}" for k, v in sorted(trial.params.items()))
        by_params[label].append(float(trial.value))

    print("===== Complete trials by params =====", flush=True)
    for label, values in sorted(by_params.items()):
        arr = np.asarray(values, dtype=float)
        print(
            f"{label}: n={len(arr)} mean={arr.mean():.4f} std={arr.std(ddof=0):.4f} "
            f"values={[round(v, 4) for v in values]}",
            flush=True,
        )

    log_file = Path(str(optuna_cfg.log_file))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("==================== New Study Results ====================\n")
        handle.write(f"tuned={','.join(s.key for s in specs)}\n")
        handle.write(
            f"sampler={sampler_name} pruner={pruner_name} n_repeats={n_repeats} "
            f"vary_trial_seed={vary_trial_seed}\n"
        )
        if complete:
            handle.write(f"best_mean_test_tm={study.best_value:.6f}\n")
            for key, value in study.best_params.items():
                handle.write(f"{key}={value}\n")
        for label, values in sorted(by_params.items()):
            arr = np.asarray(values, dtype=float)
            handle.write(
                f"arm[{label}] n={len(arr)} mean={arr.mean():.6f} "
                f"std={arr.std(ddof=0):.6f} values={values}\n"
            )
        try:
            handle.write(
                "importances="
                + str(optuna.importance.get_param_importances(study))
                + "\n"
            )
        except Exception as exc:
            handle.write(f"importances_unavailable={exc}\n")

    print(f"Wrote results to {log_file}", flush=True)
    if complete:
        print(f"Best value={study.best_value:.6f} params={study.best_params}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
