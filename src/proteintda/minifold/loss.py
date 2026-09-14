"""MiniFold-specific losses (distogram + AlphaFold structure + shared TDA)."""

import random

import torch
import torch.nn.functional as F
from minifold.model.model import MiniFoldModel
from minifold.train.loss import AlphaFoldLoss
from minifold.utils.residue_constants import atom_order
from minifold.utils.tensor_utils import tensor_tree_map

from proteintda.shared.loss import TDALoss, _as_scalar
from proteintda.utils.conversions import SideChainAtom


class MiniFoldLoss:
    """Distogram, AlphaFold structure, and TDA losses for MiniFold fine-tuning."""

    def __init__(
        self,
        config_of,
        *,
        loss_config,
        h0rff=None,
        h1rff=None,
        tda_atom: SideChainAtom = SideChainAtom.CB,
    ) -> None:
        self.config_of = config_of
        self.loss_config = loss_config
        self.minifold = loss_config.minifold
        self.tda_atom = tda_atom
        self.structure_loss = AlphaFoldLoss(config_of.loss)
        self._tda = (
            TDALoss(loss_config, h0rff=h0rff, h1rff=h1rff, tda_atom=tda_atom)
            if loss_config.tda.enabled
            else None
        )

    @property
    def tda_enabled(self) -> bool:
        return self._tda is not None and bool(self._tda._enabled)

    @staticmethod
    def _distogram_loss(
        preds: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        boundaries: torch.Tensor,
        *,
        no_bins: int,
    ) -> torch.Tensor:
        """Cross-entropy distogram loss. Adapted from minifold.train.model.MiniFold.training_step."""
        coords = coords[:, :, 1, :]
        dists = torch.cdist(coords, coords)
        boundaries = boundaries.to(device=preds.device, dtype=preds.dtype)
        labels = F.one_hot((dists.unsqueeze(-1) > boundaries).sum(dim=-1), no_bins).to(preds)
        errors = -torch.sum(labels * F.log_softmax(preds, dim=-1), dim=-1)

        square_mask = mask[:, None] * mask[:, :, None]
        square_mask = square_mask * (1 - torch.eye(dists.shape[1], device=dists.device))[None]

        denom = 1e-5 + torch.sum(square_mask, dim=(-1, -2))
        mean = errors * square_mask
        mean = torch.sum(mean, dim=-1)
        mean = mean / denom[..., None]
        mean = torch.sum(mean, dim=-1)
        return torch.mean(mean)

    def compute(
        self,
        model: MiniFoldModel,
        batch: dict,
        *,
        num_recycling: int = 0,
        include_metrics: bool = False,
    ) -> dict[str, torch.Tensor] | None:
        """Based on minifold.train.model.MiniFold.training_step."""
        try:
            r_dict = model(batch, num_recycling=num_recycling)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("OOM during MiniFold forward pass; skipping protein.")
            return None

        preds = r_dict["preds"]
        total = preds.new_zeros(())
        log: dict[str, float] = {}

        if self.minifold.distogram.enabled:
            disto_loss = self._distogram_loss(
                preds,
                batch["coords"],
                batch["mask"],
                model.boundaries,
                no_bins=preds.shape[-1],
            )
            weighted = self.minifold.distogram.weight * disto_loss
            log["distogram"] = _as_scalar(weighted)
            total = total + weighted

        needs_structure_outputs = self.minifold.structure.enabled or self.tda_enabled
        if needs_structure_outputs:
            if not model.use_structure_module:
                raise ValueError(
                    "Structure module outputs are required for structure and TDA losses."
                )
            batch_of = tensor_tree_map(lambda t: t[..., -1], batch["batch_of"])

            if self.minifold.structure.enabled:
                loss_of, of_breakdown = self.structure_loss(r_dict, batch_of, _return_breakdown=True)
                weighted_structure = self.minifold.structure.weight * loss_of
                total = total + weighted_structure
                for name, value in of_breakdown.items():
                    if name in ("loss", "unscaled_loss"):
                        continue
                    of_weight = self.config_of.loss[name].weight
                    log_key = "tm_loss" if name == "tm" else name # prevent tm loss and tm score from merging
                    log[log_key] = _as_scalar(self.minifold.structure.weight * of_weight * value)

            if self.tda_enabled:
                tda_loss, tda_breakdown = self._tda(
                    r_dict,
                    batch_of,
                    _return_breakdown=True,
                )
                weighted_tda = self.loss_config.tda.weight * tda_loss
                total = total + weighted_tda
                for name, value in tda_breakdown.items():
                    if name == "loss":
                        continue
                    term_weight = self.loss_config.tda.terms[name].weight
                    log[name] = _as_scalar(self.loss_config.tda.weight * term_weight * value)

        log["total"] = _as_scalar(total)
        result: dict[str, torch.Tensor | float | dict[str, float]] = {"total": total, "log": log}
        if include_metrics:
            if "plddt" in r_dict:
                result["plddt"] = r_dict["plddt"].detach()
            if "final_atom_positions" in r_dict:
                ca_idx = atom_order["CA"]
                result["pred_ca"] = (
                    r_dict["final_atom_positions"][:, :, ca_idx].detach().float().cpu()
                )
        return result

    @staticmethod
    def sample_recycles(max_recycles: int) -> int:
        """Random recycling count for training."""
        if max_recycles <= 0:
            return 0
        return random.randint(0, max_recycles)
