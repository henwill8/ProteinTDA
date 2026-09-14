import sys
from functools import partial
from pathlib import Path

import torch

from proteintda.config import RUN_CONFIG
from proteintda.utils.dataset import load_dataset, select_backbone, set_seed
from proteintda.utils.device import resolve_device
from proteintda.utils.kfold import KFoldRunner
from proteintda.shared.pipeline import run_fold, write_log_file


def main() -> int:
    runtime = RUN_CONFIG.runtime
    training = RUN_CONFIG.training
    device = resolve_device()
    set_seed(training.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    backbone_name, backbone = select_backbone()
    proteins = load_dataset()

    mode = "baseline" if runtime.baseline else "training"
    print(f"Starting {backbone.name} k-fold {mode}...", flush=True)

    loss_fn = None if runtime.baseline else backbone.build_loss_fn()
    fold_fn = partial(
        run_fold,
        proteins=proteins,
        device=device,
        backbone=backbone,
        loss_fn=loss_fn,
        n_splits=RUN_CONFIG.kfold.n_splits,
        train_kwargs=dict(backbone.train_kwargs),
        eval_kwargs=dict(backbone.eval_kwargs),
        **dict(backbone.runner_kwargs),
    )
    suffix = "baseline" if runtime.baseline else "kfold"
    log_file = Path(f"logs/{backbone_name}_{suffix}.log")

    fold_plddt_scores, fold_tm_scores = KFoldRunner(proteins).run(fold_fn)
    write_log_file(log_file, fold_plddt_scores, fold_tm_scores)
    print(f"Wrote results to {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
