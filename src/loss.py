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
    pred_diags: list[torch.Tensor],
    target_diags: list[torch.Tensor],
    *,
    hom_dim: int = 2,
) -> dict[str, torch.Tensor]:
    """Wasserstein distances between predicted and target persistence diagrams."""
    terms = wasserstein_distance(pred_diags, target_diags, hom_dim)
    ref = pred_diags[0] if pred_diags else target_diags[0]
    zero = torch.zeros((), device=ref.device, dtype=ref.dtype)
    return {f"h{i}": terms[i] if i < len(terms) else zero for i in range(hom_dim)}


class ESMFoldLoss(AlphaFoldLoss):
    """AlphaFoldLoss without masked MSA or experimentally-resolved terms."""

    def __init__(self, config, h0rff=None, h1rff=None):
        super().__init__(config)
        self.original_fape_config = config.fape
        self.h0rff = h0rff
        self.h1rff = h1rff

    def _build_loss_fns(self, out, batch):
        loss_fns = {}
        cfg = self.config

        def add(name, fn):
            if cfg[name].enabled:
                loss_fns[name] = fn

        add("distogram", lambda: distogram_loss(
            logits=out["distogram_logits"],
            **batch,
            **cfg.distogram,
        ))
        add("fape", lambda: fape_loss(
            out,
            batch,
            cfg.fape
        ))
        add("plddt_loss", lambda: lddt_loss(
            logits=out["lddt_logits"],
            all_atom_pred_pos=out["final_atom_positions"],
            **batch,
            **cfg.plddt_loss,
        ))
        add("supervised_chi", lambda: supervised_chi_loss(
            out["sm"]["angles"],
            out["sm"]["unnormalized_angles"],
            **batch,
            **cfg.supervised_chi,
        ))
        add("violation", lambda: violation_loss(
            out["violation"],
            **batch,
            **cfg.violation,
        ))
        add("tm", lambda: tm_loss(
            logits=out["tm_logits"],
            **out,
            **batch,
            **cfg.tm,
        ))
        add("chain_center_of_mass", lambda: chain_center_of_mass_loss(
            all_atom_pred_pos=out["final_atom_positions"],
            **batch,
            **cfg.chain_center_of_mass,
        ))

        needs_diags = (
            cfg.wasserstein_h0.enabled
            or cfg.wasserstein_h1.enabled
            or cfg.vpd_h0.enabled
            or cfg.vpd_h1.enabled
        )

        if needs_diags:
            target_diags = pd_from_graph(batch["adj"], **cfg.pd)
            pred_diags = pd_from_graph(out["adj"], **cfg.pd)

        if cfg.wasserstein_h0.enabled or cfg.wasserstein_h1.enabled:
            wasserstein_terms = wasserstein_loss(
                pred_diags=pred_diags,
                target_diags=target_diags,
                hom_dim=cfg.pd.hom_dim,
            )

            if cfg.wasserstein_h0.enabled:
                add("wasserstein_h0", lambda: wasserstein_terms["h0"])
            if cfg.wasserstein_h1.enabled:
                add("wasserstein_h1", lambda: wasserstein_terms["h1"])

        if cfg.vpd_h0.enabled:
            if self.h0rff is None:
                raise ValueError("vpd_h0 loss is enabled but h0rff was not provided")
            add("vpd_h0", lambda: self.h0rff.vpd_loss(pred_diags[0], target_diags[0]))

        if cfg.vpd_h1.enabled:
            if self.h1rff is None:
                raise ValueError("vpd_h1 loss is enabled but h1rff was not provided")
            add("vpd_h1", lambda: self.h1rff.vpd_loss(pred_diags[1], target_diags[1]))

        return loss_fns

    def loss(self, out, batch, _return_breakdown=False):
        if self.config.violation.enabled and "violation" not in out:
            out["violation"] = find_structural_violations(
                batch,
                out["sm"]["positions"][-1],
                **self.config.violation,
            )

        if self.config.fape.enabled and "renamed_atom14_gt_positions" not in batch:
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

        loss_fns = self._build_loss_fns(out, batch)
        if not loss_fns:
            raise ValueError("No loss terms are enabled in LOSS_CONFIG")

        cum_loss = None
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0.0, requires_grad=True)
            term = weight * loss
            cum_loss = term if cum_loss is None else cum_loss + term
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
