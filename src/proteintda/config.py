import ml_collections as mlc
from minifold.data.config import model_config

from vpd import _cpp

_EPS = 1e-8

# Shared OpenFold / MiniFold config for feature pipeline, model, and loss.
CONFIG_OF = model_config(
    "finetuning",
    train=True,
    low_prec=False,
    long_sequence_inference=False,
)

with CONFIG_OF.unlocked():
    # CONFIG_OF.model.heads.tm.enabled = True
    # CONFIG_OF.loss.tm.enabled = True
    # CONFIG_OF.loss.tm.weight = 0.1
    # CONFIG_OF.loss.violation.weight = 1.0
    # CONFIG_OF.loss.experimentally_resolved.weight = 0.01
    CONFIG_OF.data.train.crop_size = None

RUN_CONFIG = mlc.ConfigDict(
    {
        "data": {
            "casp_version": "debug",
            "casp_thinning": 30,
            "allow_incomplete": False,
            "scn_dir": "./data/sidechainnet",
            "max_proteins": 1000,
            "max_protein_length": None,
        },
        "runtime": {
            "baseline": False,  # True = pretrained eval only, False = fine-tune
            "device": None,  # 'cuda', 'cpu', or None for auto-detection
            "infer_recycles": 3,
            "minifold_cache_dir": "cache/minifold",
            "model_size": "12L",  # '48L' or '12L'
        },
        "kfold": {
            "n_splits": 5,
            "checkpoint_dir": "logs/kfold",
        },
        "training": {
            "seed": 42,
            "lr": 1e-5,
            "weight_decay": 0.01,
            "batch_size": 1,
            "length_bucketing": True,
            "length_bucket_size": 10,
            "train_proteins_per_epoch": None,
            "val_proteins_per_epoch": None,
            "unfreeze_fold_blocks": 12,
            "unfreeze_structure_module": True,
            "train_recycles": 3,
            "randomize_recycles": True,
            "epochs": 300,
            "patience": 5,
            "amp": True,
            "grad_clip_norm": 1.0,
            "dropout": False,
            "scheduler": {
                "enabled": True,
                "step_size": 5,
                "gamma": 0.9,
            },
        },
        "logging": {
            "baseline_log_file": "logs/esmfold_baseline.log",
            "finetune_log_file": "logs/kfold_test_scores.log",
            "minifold_log_file": "logs/minifold_kfold.log",
        },
    }
)

HEAT_RFF_CONFIG = mlc.ConfigDict(
    {
        "h0rff": {
            "n": 1,
            "axis_dim": 10,
            "resolution": 100,
            "R": 1000,
            "t": 10,
            "s": 100,
            "seed": 42,
            "device": _cpp.Device.CPU
        },
        "h1rff": {
            "n": 2,
            "axis_dim": 10,
            "resolution": 10,
            "R": 1000,
            "t": 10,
            "s": 100,
            "seed": 42,
            "device": _cpp.Device.CUDA
        }
    }
)

LOSS_CONFIG = mlc.ConfigDict(
    {
        "distogram": {
            "weight": 0.8,
            "enabled": True,
        },
        "structure": {
            "weight": 0.2,
            "enabled": True,
        },
        "tda": {
            "weight": 1.0,
            "enabled": True,
        },
        "pd": {
            "max_dimension": 2,
            "hom_dim": 2,
            "max_edge_length": 10,
        },
        "wasserstein_h0": {
            "weight": 0.01,
            "enabled": True,
        },
        "wasserstein_h1": {
            "weight": 0.9,
            "enabled": True,
        },
        "vpd_h0": {
            "weight": 1.0,
            "enabled": True,
        },
        "vpd_h1": {
            "weight": 1.0,
            "enabled": True,
        },
        "eps": _EPS,
    }
)
