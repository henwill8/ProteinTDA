"""
Local OpenFold loss config for ESMFold fine-tuning.

Copied from ``openfold.config`` with Wasserstein TDA loss sections added.
"""

import ml_collections as mlc

_EPS = 1e-8

LOSS_CONFIG = mlc.ConfigDict(
    {
        "distogram": {
            "min_bin": 2.3125,
            "max_bin": 21.6875,
            "no_bins": 64,
            "eps": _EPS,
            "weight": 0.3,
        },
        "experimentally_resolved": {
            "eps": _EPS,
            "min_resolution": 0.1,
            "max_resolution": 3.0,
            "weight": 0.0,
        },
        "fape": {
            "backbone": {
                "clamp_distance": 10.0,
                "loss_unit_distance": 10.0,
                "weight": 0.5,
            },
            "sidechain": {
                "clamp_distance": 10.0,
                "length_scale": 10.0,
                "weight": 0.5,
            },
            "eps": 1e-4,
            "weight": 1.0,
        },
        "plddt_loss": {
            "min_resolution": 0.1,
            "max_resolution": 3.0,
            "cutoff": 15.0,
            "no_bins": 50,
            "eps": _EPS,
            "weight": 0.01,
        },
        "masked_msa": {
            "num_classes": 23,
            "eps": _EPS,
            "weight": 2.0,
        },
        "supervised_chi": {
            "chi_weight": 0.5,
            "angle_norm_weight": 0.01,
            "eps": _EPS,
            "weight": 1.0,
        },
        "violation": {
            "violation_tolerance_factor": 12.0,
            "clash_overlap_tolerance": 1.5,
            "average_clashes": False,
            "eps": _EPS,
            "weight": 0.0,
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
        "wasserstein": {
            "max_dimension": 2,
            "hom_dim": 2,
        },
        "wasserstein_h0": {
            "weight": 1.0,
            "enabled": True,
        },
        "wasserstein_h1": {
            "weight": 1.0,
            "enabled": True,
        },
        "eps": _EPS,
    }
)


def loss_config(
    *,
    wasserstein_h0_weight: float = 1.0,
    wasserstein_h1_weight: float = 1.0,
    max_rips_dimension: int = 2,
    hom_dim: int = 2,
) -> mlc.ConfigDict:
    """Return a copy of the project loss config with Wasserstein overrides applied."""
    config = LOSS_CONFIG.copy_and_resolve_references()
    config.wasserstein.max_dimension = max_rips_dimension
    config.wasserstein.hom_dim = hom_dim
    config.wasserstein_h0.weight = wasserstein_h0_weight
    config.wasserstein_h0.enabled = wasserstein_h0_weight > 0.0 and hom_dim >= 1
    config.wasserstein_h1.weight = wasserstein_h1_weight
    config.wasserstein_h1.enabled = wasserstein_h1_weight > 0.0 and hom_dim >= 2
    return config
