from sklearn.model_selection import KFold, train_test_split

import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import transformers
transformers.logging.set_verbosity_error()       
transformers.utils.logging.disable_progress_bar() 

import argparse
import sys
import copy
from pathlib import Path

import sidechainnet as scn

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer 
import numpy as np

from esmfold_finetune import (
    build_model,
    train_one_epoch,
    test_model,
    trainable_parameter_count,
)
from loss import ESMFoldLoss
from config import HEAT_RFF_CONFIG, LOSS_CONFIG
from vpd_macros import create_vpd_kernels

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ESMFold with Wasserstein TDA loss.")
    parser.add_argument("--model", default="facebook/esmfold_v1")
    parser.add_argument("--casp-version", default="debug")
    parser.add_argument("--casp-thinning", type=int, default=30)
    parser.add_argument("--allow-incomplete", type=bool, default=False)
    parser.add_argument("--scn-dir", type=Path, default=Path("data/sidechainnet"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--unfreeze-trunk-blocks", type=int, default=2)
    parser.add_argument("--unfreeze-structure-module", type=bool, default=True)
    parser.add_argument("--unfreeze-esm-layers", type=int, default=0)
    parser.add_argument("--train-recycles", type=int, default=8)
    parser.add_argument("--trunk-chunk-size", type=int, default=64)
    parser.add_argument("--gradient-checkpointing", type=bool, default=True)
    parser.add_argument("--amp", type=bool, default=True)
    parser.add_argument("--optimized-param", type=str, default="tau")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-file", type=Path, default=Path("out/optuna_output.txt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-min", type=float, default=0.005)
    parser.add_argument("--w-max", type=float, default=2)
    parser.add_argument("--t-min", type=float, default=1e-25)
    parser.add_argument("--t-max", type=float, default=1e-5)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--n-trials", type=int, default=2)
    return parser.parse_args(argv)

def suggest_params(trial: optuna.Trial, args) -> dict:
    params = {
            "w_wasserstein_h0": trial.suggest_float("w_wasserstein_h0", args.w_min, args.w_max, log=True),
            "w_wasserstein_h1": trial.suggest_float("w_wasserstein_h1", args.w_min, args.w_max, log=True),
            #"w_vpd_h0": trial.suggest_float("w_vpd_h0", args.w_min, args.w_max, log=True),
            #"w_vpd_h1": trial.suggest_float("w_vpd_h1", args.w_min, args.w_max, log=True),
            #"tau_h0": trial.suggest_float("tau_h0", args.t_min, args.t_max, log=True),
            #"tau_h1": trial.suggest_float("tau_h1", args.t_min, args.t_max, log=True)
    }
    return params

def build_params(params:dict):
    loss_cfg = copy.deepcopy(LOSS_CONFIG)
    heat_cfg = copy.deepcopy(HEAT_RFF_CONFIG)

    loss_cfg.wasserstein_h0.weight = params["w_wasserstein_h0"]
    loss_cfg.wasserstein_h1.weight = params["w_wasserstein_h1"]
    #loss_cfg.vpd_h0.weight = params["w_vpd_h0"]
    #loss_cfg.vpd_h1.weight = params["w_vpd_h1"]

    #heat_cfg.h0rff.tau = params["tau_h0"]
    #heat_cfg.h1rff.tau = params["tau_h1"]

    return loss_cfg, heat_cfg

def create_obj_function(args, device, tokenizer, dataset):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, args)
        loss_cfg, heat_cfg = build_params(params)
        h0rff, h1rff = create_vpd_kernels(loss_cfg, heat_cfg)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_tm_scores: list[float] = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
            if fold >= args.n_folds:
                break
            set_seed(42 + fold)
        
            model = build_model(
                args.model,
                device,
                unfreeze_esm_layers=args.unfreeze_esm_layers,
                unfreeze_trunk_blocks=args.unfreeze_trunk_blocks,
                unfreeze_structure_module=args.unfreeze_structure_module,
                trunk_chunk_size=args.trunk_chunk_size,
                gradient_checkpointing=args.gradient_checkpointing,
            )
            trainable, total = trainable_parameter_count(model)
        
            train_idx, val_idx = train_test_split(train_idx, test_size=0.25)

            train_dataset = [dataset[i] for i in train_idx]
            val_dataset = [dataset[i] for i in val_idx]
            test_dataset = [dataset[i] for i in test_idx]

            train_loader = DataLoader(
                dataset = train_dataset,
                batch_size = args.batch_size,
                shuffle = True,
                collate_fn = lambda x: x,
            )

            val_loader = DataLoader(
                dataset = val_dataset,
                batch_size = args.batch_size,
                shuffle = False,
                collate_fn = lambda x: x,
            )

            test_loader = DataLoader(
                dataset = test_dataset,
                batch_size = args.batch_size,
                shuffle = False,
                collate_fn = lambda x: x,
            )

            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

            best_model_weights = None
            max_val_tm = 0.0 
            patience = 0

            report_num = 0

            for epoch in range(args.epochs):
                report_num += 1
                metrics = train_one_epoch(
                    model,
                    tokenizer,
                    train_loader,
                    optimizer,
                    device,
                    loss_fn=ESMFoldLoss(config=LOSS_CONFIG, h0rff=h0rff, h1rff=h1rff),
                    unfreeze_esm_layers=args.unfreeze_esm_layers,
                    unfreeze_trunk_blocks=args.unfreeze_trunk_blocks,
                    unfreeze_structure_module=args.unfreeze_structure_module,
                    train_recycles=args.train_recycles,
                    use_amp=args.amp,
                )
                _, val_tm_score = test_model(
                    model,
                    tokenizer,
                    val_loader,
                    device,
                )

                best_val_tm = max(max_val_tm, val_tm_score)
    
                trial.report(best_val_tm, report_num)

                if trial.should_prune():
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned()

                if val_tm_score > max_val_tm:
                    max_val_tm = val_tm_score
                    patience = 0
                    best_model_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1

                if patience > args.patience:
                    print(f"Early stoppping at epoch {epoch}")
                    break

            if best_model_weights is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})

            _, tm_score = test_model(
                model,
                tokenizer,
                test_loader,
                device,
            )
            fold_tm_scores.append(tm_score)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(fold_tm_scores)
        return (np.mean(fold_tm_scores)) if fold_tm_scores else 0.0

    return objective


def main(argv: list[str] | None = None) -> int:
    optuna.logging.set_verbosity(optuna.logging.INFO)

    args = parse_args(argv)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = scn.load(
        casp_version = args.casp_version,
        casp_thinning = args.casp_thinning,
        scn_dataset = True,
        scn_dir = str(args.scn_dir),
        force_download = False,
        complete_structures_only = not args.allow_incomplete,
    )

    # The last 5 proteins in the dataset were large enough to  significantly impact memory usage and time complexity.
    dataset = dataset[:-5]

    study = optuna.create_study(
        study_name="TDA Optimizer",
        storage="sqlite:///optuna_tda.db",
        sampler=TPESampler(seed=args.seed),
        pruner=SuccessiveHalvingPruner(),
        direction="maximize",
        load_if_exists=True
    )

    study.optimize(create_obj_function(args, device, tokenizer, dataset), show_progress_bar=True, n_trials=args.n_trials)

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("a") as f:
        f.write("==================== New Study Results ====================\n")
        f.write("========== Best Values= =========\n")
        f.write(f"best_mean_val_tm={study.best_value:.6f}\n")
        for k, v in study.best_params.items():
            f.write(f"{k}={v:.8g}\n")
        f.write("========== Parameter Importances ==========\n")
        f.write(str(optuna.importance.get_param_importances(study)))

    return 0

if __name__ == "__main__":
    sys.exit(main())
