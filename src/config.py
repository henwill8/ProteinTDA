import ml_collections as mlc

_EPS = 1e-8

HEAT_RFF_CONFIG = mlc.ConfigDict(
    {
        "h0rff": {
            "n": 1,
            "axis_dim": 10,
            "resolution": 1000,
            "R": 1000,
            "tau": 0.00001,
            "mask": None,
            "seed": 42
        },
        "h1rff": {
            "n": 2,
            "axis_dim": 10,
            "resolution": 100,
            "R": 1000,
            "tau": 0.00001,
            "mask": None,
            "seed": 42
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
            "enabled": True,
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
            "weight": 1.0,
            "enabled": True,
        },
        "wasserstein_h1": {
            "weight": 1.0,
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
