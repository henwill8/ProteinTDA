import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from minifold.model.model import MiniFoldModel
from minifold.train.loss import AlphaFoldLoss
from minifold.utils.residue_constants import atom_order
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


def _as_tensor(value: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return ref.new_tensor(value)


def _as_scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, float):
        return value
    return float(value.detach())


class TDALoss:
    """Wasserstein and VPD losses on persistence diagrams from distance matrices."""

    def __init__(
        self,
        config,
        h0rff=None,
        h1rff=None,
        *,
        tda_atom: SideChainAtom = SideChainAtom.CB,
    ):
        self.config = config
        self.h0rff = h0rff
        self.h1rff = h1rff
        self.tda_atom = tda_atom
        self._atom37 = Atom37.CB if tda_atom is SideChainAtom.CB else Atom37.CA
        terms = config.tda.terms
        self._enabled = tuple(name for name in terms if terms[name].enabled)
        vpd_dims = [
            int(name.rsplit("_h", 1)[1])
            for name in self._enabled
            if name.startswith("vpd_")
        ]
        if 0 in vpd_dims and h0rff is None:
            raise ValueError("vpd_h0 loss is enabled but h0rff was not provided")
        if any(dim > 0 for dim in vpd_dims) and h1rff is None:
            raise ValueError("vpd_h1+ loss is enabled but h1rff was not provided")

    def _create_adjs(
        self,
        out: dict,
        batch: dict,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_positions = out["final_atom_positions"][index]
        pred_mask = out["final_atom_mask"][index]
        target_positions = batch["all_atom_positions"][index]
        target_mask = batch["all_atom_mask_true"][index]

        pred_pts = atom_positions_from_atom37(pred_positions, pred_mask, self._atom37)
        target_pts = atom_positions_from_atom37(target_positions, target_mask, self._atom37)
        return _distance_matrix(pred_pts), _distance_matrix(target_pts).detach()

    def _term_losses(
        self,
        pred_adj: torch.Tensor,
        target_adj: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        target_diags = pd_from_graph(target_adj, **cfg.tda.pd)
        pred_diags = pd_from_graph(pred_adj, **cfg.tda.pd)
        wasserstein = _wasserstein_terms(
            pred_diags,
            target_diags,
            hom_dim=cfg.tda.pd.hom_dim,
        )

        terms: dict[str, torch.Tensor] = {}
        for name in self._enabled:
            if name.startswith("wasserstein_"):
                dim = int(name.rsplit("_h", 1)[1])
                terms[name] = wasserstein[f"h{dim}"]
            elif name.startswith("vpd_"):
                dim = int(name.rsplit("_h", 1)[1])
                rff = self.h0rff if dim == 0 else self.h1rff
                terms[name] = rff.vpd_loss(pred_diags[dim], target_diags[dim])
            else:
                raise ValueError(f"Unknown TDA term: {name}")
        return terms

    def _loss_from_adjs(
        self,
        pred_adj: torch.Tensor,
        target_adj: torch.Tensor,
        *,
        _return_breakdown: bool = False,
    ):
        if not self._enabled:
            zero = pred_adj.new_zeros((), requires_grad=True)
            if _return_breakdown:
                return zero, {}
            return zero

        ref = pred_adj
        cum_loss = ref.new_zeros(())
        losses: dict[str, torch.Tensor] = {}
        for name, loss in self._term_losses(pred_adj, target_adj).items():
            loss = _as_tensor(loss, ref)
            if not torch.isfinite(loss).all():
                print(f"{name} loss is NaN or Inf. Skipping...")
                loss = loss.new_zeros((), requires_grad=True)
            cum_loss = cum_loss + self.config.tda.terms[name].weight * loss
            losses[name] = loss.detach().clone()

        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss
        return cum_loss, losses

    def __call__(self, out, batch, _return_breakdown=False):
        batch_size = out["final_atom_positions"].shape[0]
        tda_losses: list[torch.Tensor] = []
        breakdown: defaultdict[str, list[torch.Tensor]] = defaultdict(list)

        for index in range(batch_size):
            pred_adj, target_adj = self._create_adjs(out, batch, index)
            loss_i, breakdown_i = self._loss_from_adjs(pred_adj, target_adj, _return_breakdown=True)
            tda_losses.append(loss_i)
            for name, value in breakdown_i.items():
                if name == "loss":
                    continue
                breakdown[name].append(value)

        tda_loss = torch.stack(tda_losses).mean()
        if not _return_breakdown:
            return tda_loss

        tda_breakdown = {
            name: torch.stack(values).mean()
            for name, values in breakdown.items()
        }
        tda_breakdown["loss"] = tda_loss.detach().clone()
        return tda_loss, tda_breakdown


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

        if self.loss_config.distogram.enabled:
            disto_loss = self._distogram_loss(
                preds,
                batch["coords"],
                batch["mask"],
                model.boundaries,
                no_bins=preds.shape[-1],
            )
            weighted = self.loss_config.distogram.weight * disto_loss
            log["distogram"] = _as_scalar(weighted)
            total = total + weighted

        needs_structure_outputs = (
            self.loss_config.structure.enabled
            or (self.tda_enabled and self.loss_config.tda.enabled)
        )
        if needs_structure_outputs:
            if not model.use_structure_module:
                raise ValueError(
                    "Structure module outputs are required for structure and TDA losses."
                )
            batch_of = tensor_tree_map(lambda t: t[..., -1], batch["batch_of"])

            if self.loss_config.structure.enabled:
                loss_of, of_breakdown = self.structure_loss(r_dict, batch_of, _return_breakdown=True)
                weighted_structure = self.loss_config.structure.weight * loss_of
                total = total + weighted_structure
                for name, value in of_breakdown.items():
                    if name in ("loss", "unscaled_loss"):
                        continue
                    of_weight = self.config_of.loss[name].weight
                    log_key = "tm_loss" if name == "tm" else name # prevent tm loss and tm score from merging
                    log[log_key] = _as_scalar(self.loss_config.structure.weight * of_weight * value)

            if self.tda_enabled and self.loss_config.tda.enabled:
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
