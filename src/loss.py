import torch
import random
from vpd import _cpp

from openfold.utils.loss import (
    AlphaFoldLoss,
    chain_center_of_mass_loss,
    compute_renamed_ground_truth,
    distogram_loss,
    fape_loss,
    find_structural_violations,
    lddt_loss,
    supervised_chi_loss,
    tm_loss,
    violation_loss,
)
from persistence import pd_from_graph, wasserstein_distance

def wasserstein_loss(
    pred_adj: torch.Tensor,
    target_adj: torch.Tensor,
    *,
    max_dimension: int = 2,
    hom_dim: int = 2,
) -> dict[str, torch.Tensor]:
    """Wasserstein distances between predicted and target persistence diagrams."""
    target_diags = pd_from_graph(target_adj, max_dimension, hom_dim)
    pred_diags = pd_from_graph(pred_adj, max_dimension, hom_dim)
    terms = wasserstein_distance(pred_diags, target_diags, hom_dim)
    zero = torch.zeros((), device=pred_adj.device, dtype=pred_adj.dtype)
    return {f"h{i}": terms[i] if i < len(terms) else zero for i in range(hom_dim)}


class ESMFoldLoss(AlphaFoldLoss):
    """AlphaFoldLoss without masked MSA or experimentally-resolved terms."""

    def __init__(self, config, h0rff, h1rff):
        super().__init__(config)
        self.original_fape_config = config.fape
        self.h0rff = h0rff
        self.h1rff = h1rff

    def loss(self, out, batch, _return_breakdown=False):
        if self.config.violation.enabled and "violation" not in out:
            out["violation"] = find_structural_violations(
                batch,
                out["sm"]["positions"][-1],
                **self.config.violation,
            )

        if "renamed_atom14_gt_positions" not in out:
            batch.update(
                compute_renamed_ground_truth(
                    batch,
                    out["sm"]["positions"][-1],
                )
            )

        if random.random() < 0.1:
            self.config.fape.backbone.use_clamped_fape = None
            self.config.fape.sidechain.use_clamped_fape = None
        else:
            self.config.fape = self.original_fape_config

        loss_fns = {
            "distogram": lambda: distogram_loss(
                logits=out["distogram_logits"],
                **{**batch, **self.config.distogram},
            ),
            "fape": lambda: fape_loss(
                out,
                batch,
                self.config.fape,
            ),
            "plddt_loss": lambda: lddt_loss(
                logits=out["lddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                **{**batch, **self.config.plddt_loss},
            ),
            "supervised_chi": lambda: supervised_chi_loss(
                out["sm"]["angles"],
                out["sm"]["unnormalized_angles"],
                **{**batch, **self.config.supervised_chi},
            ),
        }

        if self.config.violation.enabled:
            loss_fns["violation"] = lambda: violation_loss(
                out["violation"],
                **{**batch, **self.config.violation},
            )

        if self.config.tm.enabled:
            loss_fns["tm"] = lambda: tm_loss(
                logits=out["tm_logits"],
                **{**batch, **out, **self.config.tm},
            )

        if self.config.chain_center_of_mass.enabled:
            loss_fns["chain_center_of_mass"] = lambda: chain_center_of_mass_loss(
                all_atom_pred_pos=out["final_atom_positions"],
                **{**batch, **self.config.chain_center_of_mass},
            )

        wasserstein_terms = wasserstein_loss(
            pred_adj=out["adj"],
            target_adj=batch["adj"],
            **self.config.wasserstein,
        )

        if self.config.wasserstein_h0.enabled:
            loss_fns["wasserstein_h0"] = lambda: wasserstein_terms["h0"]

        if self.config.wasserstein_h1.enabled:
            loss_fns["wasserstein_h1"] = lambda: wasserstein_terms["h1"]

        cum_loss = 0.0
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0.0, requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()
        losses["unscaled_loss"] = cum_loss.detach().clone()

        # Scale the loss by the square root of the minimum of the crop size and
        # the (average) sequence length. See subsection 1.9.
        seq_len = torch.mean(batch["seq_length"].float())
        crop_len = batch["aatype"].shape[-1]
        cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))

        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses
