from proteintda.config import LOSS_CONFIG
from proteintda.lightrosetta.loss import LightRoseTTALoss
from proteintda.lightrosetta.runner import LightRoseTTARunner
from proteintda.shared.pipeline import Backbone, prepare_vpd_kernels, tda_atom_from_config


def build_loss_fn() -> LightRoseTTALoss:
    h0rff, h1rff = prepare_vpd_kernels()
    return LightRoseTTALoss(
        LOSS_CONFIG, h0rff=h0rff, h1rff=h1rff, tda_atom=tda_atom_from_config()
    )


BACKBONE = Backbone(
    name="LightRoseTTA",
    build_runner=LightRoseTTARunner,
    build_loss_fn=build_loss_fn,
)
