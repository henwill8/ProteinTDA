import ml_collections as mlc

_EPS = 1e-8

RUN_CONFIG = mlc.ConfigDict(
    {
        "data": {
            "casp_version": "debug",
            "casp_thinning": 30,
            "allow_incomplete": False,
            "scn_dir": "data/sidechainnet",
            "max_proteins": 1000,
            "max_protein_length": None,
        },
        "runtime": {
            "device": None,  # 'cuda', 'cpu', or None for auto-detection
            "infer_recycles": 3,
            "minifold_cache_dir": "cache/minifold",
            "model_size": "48L",  # '48L' or '12L'
        },
        "kfold": {
            "n_splits": 5,
            "checkpoint_dir": "logs/kfold",
        },
        "training": {
            "seed": 42,
            "lr": 1e-4,
            "batch_size": 1,
            "train_recycles": None,
            "epochs": 300,
            "patience": 5,
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
            "resolution": 1000,
            "R": 1000,
            "t": 7e-9,
            "s": 1.0,
            "seed": 42
        },
        "h1rff": {
            "n": 2,
            "axis_dim": 10,
            "resolution": 10,
            "R": 1000,
            "t": 1e-10,
            "s": 1.0,
            "seed": 42
        }
    }
)

LOSS_CONFIG = mlc.ConfigDict(
    {
        "pd": {
            "max_dimension": 2,
            "hom_dim": 2,
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
