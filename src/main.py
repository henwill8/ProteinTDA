import sys
from functools import partial
from pathlib import Path

import torch

from proteintda.config import RUN_CONFIG
from proteintda.minifold.pipeline import (
    build_loss_fn,
    run_baseline_fold,
    run_train_fold,
    write_log_file,
)
from proteintda.utils.dataset import load_dataset, set_seed
from proteintda.utils.kfold import KFoldRunner


def resolve_device() -> torch.device:
    device = RUN_CONFIG.runtime.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def main() -> int:
    runtime = RUN_CONFIG.runtime
    training = RUN_CONFIG.training
    device = resolve_device()
    set_seed(training.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    proteins = load_dataset()
    cache_dir = Path(runtime.minifold_cache_dir)
    runner = KFoldRunner(proteins, baseline=runtime.baseline)

    if runtime.baseline:
        fold_fn = partial(
            run_baseline_fold,
            proteins=proteins,
            cache_dir=cache_dir,
            device=device,
            model_size=runtime.model_size,
            n_splits=RUN_CONFIG.kfold.n_splits,
        )
        log_path = RUN_CONFIG.logging.minifold_log_file
    else:
        loss_fn = build_loss_fn()
        fold_fn = partial(
            run_train_fold,
            proteins=proteins,
            cache_dir=cache_dir,
            device=device,
            model_size=runtime.model_size,
            loss_fn=loss_fn,
            n_splits=RUN_CONFIG.kfold.n_splits,
        )
        log_path = RUN_CONFIG.logging.finetune_log_file

    fold_plddt_scores, fold_tm_scores = runner.run(fold_fn)

    log_file = Path(log_path)
    write_log_file(log_file, fold_plddt_scores, fold_tm_scores)
    print(f"Wrote results to {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
