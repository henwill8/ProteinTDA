import random

import torch
import torch.nn.functional as F
from minifold.model.model import MiniFoldModel
from minifold.train.loss import AlphaFoldLoss
from minifold.utils.tensor_utils import tensor_tree_map

from proteintda.tda.persistence import pd_from_graph, wasserstein_distance
from proteintda.utils.conversions import Atom37, SideChainAtom, atom_positions_from_atom37


def _distance_matrix(positions: torch.Tensor) -> torch.Tensor:
    """Full pairwise distance matrix, shape (n, n)."""
    dists = torch.cdist(positions, positions).clone()
    dists.fill_diagonal_(0.0)
    return dists


def _wasserstein_terms(
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


class TDALoss:
    """Wasserstein and VPD losses on persistence diagrams from distance matrices."""

    def __init__(self, config, h0rff=None, h1rff=None):
        self.config = config
        self.h0rff = h0rff
        self.h1rff = h1rff

    def _build_loss_fns(self, out, batch):
        loss_fns = {}
        cfg = self.config

        def add(name, fn):
            if cfg[name].enabled:
                loss_fns[name] = fn

        target_diags = pd_from_graph(batch["adj"], **cfg.pd)
        pred_diags = pd_from_graph(out["adj"], **cfg.pd)

        wasserstein = _wasserstein_terms(
            pred_diags=pred_diags,
            target_diags=target_diags,
            hom_dim=cfg.pd.hom_dim,
        )
        add("wasserstein_h0", lambda: wasserstein["h0"])
        add("wasserstein_h1", lambda: wasserstein["h1"])

        if self.h0rff is None and cfg.vpd_h0.enabled:
            raise ValueError("vpd_h0 loss is enabled but h0rff was not provided")
        add("vpd_h0", lambda: self.h0rff.vpd_loss(pred_diags[0], target_diags[0]))

        if self.h1rff is None and cfg.vpd_h1.enabled:
            raise ValueError("vpd_h1 loss is enabled but h1rff was not provided")
        add("vpd_h1", lambda: self.h1rff.vpd_loss(pred_diags[1], target_diags[1]))

        return loss_fns

    def __call__(self, out, batch, _return_breakdown=False):
        loss_fns = self._build_loss_fns(out, batch)
        if not loss_fns:
            zero = torch.zeros((), device=out["adj"].device, dtype=out["adj"].dtype, requires_grad=True)
            if _return_breakdown:
                return zero, {}
            return zero

        cum_loss = 0.0
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if not torch.isfinite(loss).all():
                print(f"{loss_name} loss is NaN or Inf. Skipping...")
                loss = loss.new_zeros((), requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()

        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss
        return cum_loss, losses


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
        self.h0rff = h0rff
        self.h1rff = h1rff
        self.tda_atom = tda_atom
        self.structure_loss = AlphaFoldLoss(config_of.loss)
        self._tda = TDALoss(loss_config, h0rff=h0rff, h1rff=h1rff) if loss_config.tda.enabled else None

    @property
    def tda_enabled(self) -> bool:
        return self._tda is not None and any(
            self._tda.config[name].enabled
            for name in ("wasserstein_h0", "wasserstein_h1", "vpd_h0", "vpd_h1")
        )

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

    def _add_tda_fields(self, r_dict: dict, batch_of: dict) -> None:
        atom = Atom37.CB if self.tda_atom is SideChainAtom.CB else Atom37.CA
        pred_positions = r_dict["final_atom_positions"][0]
        pred_mask = r_dict["final_atom_mask"][0]
        target_positions = batch_of["all_atom_positions"][0]
        target_mask = batch_of["all_atom_mask_true"][0]

        pred_pts = atom_positions_from_atom37(pred_positions, pred_mask, atom)
        target_pts = atom_positions_from_atom37(target_positions, target_mask, atom)
        r_dict["adj"] = _distance_matrix(pred_pts)
        batch_of["adj"] = _distance_matrix(target_pts).detach()

    def compute(
        self,
        model: MiniFoldModel,
        batch: dict,
        *,
        num_recycling: int = 0,
    ) -> dict[str, torch.Tensor] | None:
        """Based on minifold.train.model.MiniFold.training_step."""
        try:
            r_dict = model(batch, num_recycling=num_recycling)
        except torch.cuda.OutOfMemoryError:
            print("OOM during MiniFold forward pass; skipping protein.")
            return None

        preds = r_dict["preds"]
        disto_loss = self._distogram_loss(
            preds,
            batch["coords"],
            batch["mask"],
            model.boundaries,
            no_bins=preds.shape[-1],
        )
        total = 0.0
        losses: dict[str, torch.Tensor] = {}
        if self.loss_config.distogram.enabled:
            total = total + self.loss_config.distogram.weight * disto_loss
            losses["distogram"] = disto_loss.detach()

        if not model.use_structure_module or not self.loss_config.structure.enabled:
            losses["total"] = total
            return losses

        batch_of = tensor_tree_map(lambda t: t[..., -1], batch["batch_of"])
        loss_of, of_breakdown = self.structure_loss(r_dict, batch_of, _return_breakdown=True)
        total = total + self.loss_config.structure.weight * loss_of
        losses["structure"] = loss_of.detach()
        for name, value in of_breakdown.items():
            if name != "loss":
                losses[f"of_{name}"] = value

        if self.tda_enabled and self.loss_config.tda.enabled:
            self._add_tda_fields(r_dict, batch_of)
            tda_loss, tda_breakdown = self._tda(r_dict, batch_of, _return_breakdown=True)
            total = total + self.loss_config.tda.weight * tda_loss
            losses.update(tda_breakdown)

        losses["total"] = total
        return losses

    @staticmethod
    def sample_recycles(max_recycles: int) -> int:
        """Random recycling count for training."""
        if max_recycles <= 0:
            return 0
        return random.randint(0, max_recycles)
