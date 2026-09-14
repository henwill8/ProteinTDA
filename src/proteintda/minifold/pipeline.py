from pathlib import Path

import torch

from proteintda.config import LOSS_CONFIG, MINIFOLD_CONFIG_OF, RUN_CONFIG
from proteintda.minifold.loss import MiniFoldLoss
from proteintda.minifold.runner import MiniFoldRunner
from proteintda.shared.pipeline import Backbone, prepare_vpd_kernels, tda_atom_from_config


def build_loss_fn() -> MiniFoldLoss:
    h0rff, h1rff = prepare_vpd_kernels()
    return MiniFoldLoss(
        MINIFOLD_CONFIG_OF,
        loss_config=LOSS_CONFIG,
        h0rff=h0rff,
        h1rff=h1rff,
        tda_atom=tda_atom_from_config(),
    )


def build_runner(device: torch.device, *, train: bool = False, **_extra) -> MiniFoldRunner:
    training = RUN_CONFIG.training
    minifold = RUN_CONFIG.minifold
    return MiniFoldRunner(
        Path(minifold.cache_dir),
        model_size=minifold.model_size,
        device=device,
        train=train,
        unfreeze_fold_blocks=training.unfreeze_fold_blocks if train else 0,
        unfreeze_structure_module=training.unfreeze_structure_module if train else False,
    )


BACKBONE = Backbone(
    name="MiniFold",
    build_runner=build_runner,
    build_loss_fn=build_loss_fn,
    train_kwargs={
        "num_recycling": RUN_CONFIG.training.train_recycles or 0,
        "randomize_recycles": RUN_CONFIG.training.randomize_recycles,
    },
    eval_kwargs={"num_recycling": RUN_CONFIG.runtime.infer_recycles or 0},
)
