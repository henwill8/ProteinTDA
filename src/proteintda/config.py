import json
from enum import Enum, auto
from typing import Any

import ml_collections as mlc
from minifold.data.config import model_config

from vpd import _cpp

_EPS = 1e-8

# OpenFold-style MiniFold config for feature pipeline, model, and AlphaFold loss.
MINIFOLD_CONFIG_OF = model_config(
    "finetuning",
    train=True,
    low_prec=False,
    long_sequence_inference=False,
)

with MINIFOLD_CONFIG_OF.unlocked():
    # MINIFOLD_CONFIG_OF.model.heads.tm.enabled = True
    # MINIFOLD_CONFIG_OF.loss.tm.enabled = True
    # MINIFOLD_CONFIG_OF.loss.tm.weight = 0.1
    # MINIFOLD_CONFIG_OF.loss.violation.weight = 1.0
    # MINIFOLD_CONFIG_OF.loss.experimentally_resolved.weight = 0.01
    MINIFOLD_CONFIG_OF.data.train.crop_size = None


class SamplingMethod(Enum):
    RANDOM = auto()
    REJECTION = auto()
    MCMC = auto()
    MALA = auto()


class GraphRepresentation(Enum):
    COMPLETE = auto()
    LATTICE = auto()


RUN_CONFIG = mlc.ConfigDict(
    {
        "data": {
            "casp_version": "11",
            "casp_thinning": 30,
            "allow_incomplete": False,
            "scn_dir": "./data/sidechainnet",
            "max_proteins": 1000,
            "max_protein_length": None,
            # Keep proteins the original (non-TDA) model is weak on (TM <= threshold).
            # LightRoseTTA needs data.baseline_checkpoint when this is set.
            "max_baseline_tm": None,
            "baseline_tm_scores_dir": "cache/baseline_tm_scores",
            # Non-TDA weights for LightRoseTTA TM filter (MiniFold uses its pretrained ckpt).
            "baseline_checkpoint": None,
        },
        "runtime": {
            "backbone": "lightrosetta",  # 'minifold' or 'lightrosetta'
            "baseline": False,  # True = eval only, False = train
            "device": None,  # 'cuda', 'cpu', or None for auto-detection (also 'cuda:n' for GPU n)
            "infer_recycles": 3,
        },
        "kfold": {
            "n_splits": 5,
            "checkpoint_dir": "logs/kfold",
        },
        "training": {
            "seed": 42,
            "lr": 5e-4,
            "weight_decay": 5e-4,
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
        "minifold": {
            "model_size": "12L",  # '48L' or '12L'
            "cache_dir": "cache/minifold",
        },
        "lightrosetta": {
            # Pair/SE3 activations are O(L^2); 256 fits a 22GB L4 in float32.
            "max_seq_length": 256,
            "model": {
                "n_module": 4,
                "n_module_str": 1,
                "n_layer": 1,
                "d_msa": 32,
                "d_pair": 32,
                "d_templ": 32,
                "d_hidden": 32,
                "p_drop": 0.1,
                "use_templ": True,
            },
        },
    }
)

HEAT_RFF_CONFIG = mlc.ConfigDict(
    {
        "h0rff": {
            "n": 1,
            "axis_dim": 10,
            "resolution": 5,
            "R": 1000,
            "t": 2,
            "s": 0.4,
            "seed": 42,
            "device": _cpp.Device.CUDA,
            "sampling_method": SamplingMethod.MALA,
            "graph_representation_type": GraphRepresentation.LATTICE,
        },
        "h1rff": {
            "n": 2,
            "axis_dim": 10,
            "resolution": 2,
            "R": 1000,
            "t": 2,
            "s": 0.4,
            "seed": 42,
            "device": _cpp.Device.CUDA,
            "sampling_method": SamplingMethod.MALA,
            "graph_representation_type": GraphRepresentation.LATTICE,
        },
        "h2rff": {
            "n": 2,
            "axis_dim": 10,
            "resolution": 1,
            "R": 1000,
            "t": 2,
            "s": 0.4,
            "seed": 42,
            "device": _cpp.Device.CUDA,
            "sampling_method": SamplingMethod.MALA,
            "graph_representation_type": GraphRepresentation.LATTICE,
        },
    }
)

LOSS_CONFIG = mlc.ConfigDict(
    {
        "minifold": {
            "distogram": {
                "weight": 0.8,
                "enabled": True,
            },
            "structure": {
                "weight": 0.2,
                "enabled": True,
            },
        },
        "lightrosetta": {
            "distogram": {"weight": 0.3, "enabled": True},
            "omega": {"weight": 0.5, "enabled": True},
            "theta": {"weight": 0.5, "enabled": True},
            "phi": {"weight": 0.5, "enabled": True},
            "coor": {"weight": 0.5, "enabled": True},
            "plddt_loss": {"weight": 0.01, "enabled": True},
            "bond": {"weight": 1.0, "epoch_scale": 0.005, "enabled": True},
            "angle": {"weight": 1.0, "epoch_scale": 0.005, "enabled": True},
            "dihedral": {"weight": 1.0, "epoch_scale": 0.005, "enabled": True},
        },
        "tda": {
            "weight": 1.0,
            "enabled": True,
            "atom": "CB",
            "pd": {
                "max_dimension": 2,
                "hom_dim": 2,
                "max_edge_length": 10,
            },
            "terms": {
                "wasserstein_h0": {
                    "weight": 0.1,
                    "enabled": True,
                },
                "wasserstein_h1": {
                    "weight": 0.8,
                    "enabled": True,
                },
                "wasserstein_h2": {
                    "weight": 0.4,
                    "enabled": False,
                },
                "vpd_h0": {
                    "weight": 0.001,
                    "enabled": True,
                },
                "vpd_h1": {
                    "weight": 0.00001,
                    "enabled": True,
                },
                "vpd_h2": {
                    "weight": 0.00001,
                    "enabled": False,
                },
            }
        },
        "eps": _EPS,
    }
)


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_set_overrides(pairs: list[str]) -> None:
    """Apply dotted overrides like runtime.device=cuda:1 or loss.tda.weight=0.5."""
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override must be key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        parts = key.split(".")
        if not parts or any(not part for part in parts):
            raise ValueError(f"Invalid override key {key!r}")
        if parts[0] == "loss":
            root = LOSS_CONFIG
            parts = parts[1:]
            if not parts:
                raise ValueError("loss override needs a field path")
        else:
            root = RUN_CONFIG
            if parts[0] == "run":
                parts = parts[1:]
                if not parts:
                    raise ValueError("run override needs a field path")
        value = _parse_override_value(raw)
        with root.unlocked():
            node = root
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], mlc.ConfigDict):
                    raise KeyError(f"Unknown config path {key!r}")
                node = node[part]
            leaf = parts[-1]
            if leaf not in node:
                raise KeyError(f"Unknown config path {key!r}")
            node[leaf] = value
