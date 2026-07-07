import ml_collections as mlc
from vpd import _cpp

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
        "model": {
            "name": "facebook/esmfold_v1",
        },
        "runtime": {
            "baseline": False,
            "device": None, # 'cuda', 'cpu', or None for auto-detection
            "use_esm_cache": True,
            "esm_cache_dir": "cache/esm_embeddings",
            # 0 = cache ESM embeddings only, N = cache first N trunk blocks, -N = cache last N trunk blocks
            "esm_cache_trunk_blocks": -2,
            "trunk_chunk_size": 1024,
            "infer_recycles": None,
        },
        "kfold": {
            "n_splits": 5,
            "checkpoint_dir": "logs/kfold",
        },
        "training": {
            "seed": 42,
            "lr": 1e-4,
            "batch_size": 1,
            "train_proteins_per_epoch": None,
            "val_proteins_per_epoch": None,
            "unfreeze_trunk_blocks": 2,
            "unfreeze_structure_module": True,
            "train_recycles": None,
            "epochs": 300,
            "patience": 5,
            "gradient_checkpointing": True,
            "amp": True
        },
        "logging": {
            "baseline_log_file": "logs/esmfold_baseline.log",
            "finetune_log_file": "logs/kfold_test_scores.log",
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

# Copied from openfold.config
LOSS_CONFIG = mlc.ConfigDict(
    {
        "distogram": {
            "min_bin": 2.3125,
            "max_bin": 21.6875,
            "no_bins": 64,
            "eps": _EPS,
            "weight": 0.3,
            "enabled": True,
        },
        "fape": {
            "backbone": {
                "clamp_distance": 10.0,
                "loss_unit_distance": 10.0,
                "use_clamped_fape": 0.9,
                "weight": 0.5,
            },
            "sidechain": {
                "clamp_distance": 10.0,
                "length_scale": 10.0,
                "use_clamped_fape": 0.9,
                "weight": 0.5,
            },
            "eps": 1e-4,
            "weight": 1.0,
            "enabled": True,
        },
        "plddt_loss": {
            "min_resolution": 0.1,
            "max_resolution": 3.0,
            "cutoff": 15.0,
            "no_bins": 50,
            "eps": _EPS,
            "weight": 0.01,
            "enabled": True,
        },
        "masked_msa": {
            "num_classes": 23,
            "eps": _EPS,
            "weight": 2.0,
            "enabled": True,
        },
        "supervised_chi": {
            "chi_weight": 0.5,
            "angle_norm_weight": 0.01,
            "eps": _EPS,
            "weight": 1.0,
            "enabled": True,
        },
        "violation": { # if violation loss is enabled, then stereo_chemical_props.txt must be present in the openfold installation
            "violation_tolerance_factor": 12.0,
            "clash_overlap_tolerance": 1.5,
            "average_clashes": False,
            "eps": _EPS,
            "weight": 1.0,
            "enabled": False,
        },
        "tm": {
            "max_bin": 31,
            "no_bins": 64,
            "min_resolution": 0.1,
            "max_resolution": 3.0,
            "eps": _EPS,
            "weight": 0.0,
            "enabled": False,
        },
        "chain_center_of_mass": {
            "clamp_distance": -4.0,
            "weight": 0.0,
            "eps": _EPS,
            "enabled": False,
        },
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
