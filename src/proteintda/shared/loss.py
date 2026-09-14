"""Shared TDA loss (Wasserstein / VPD)."""

from collections import defaultdict

import torch

from proteintda.tda.persistence import pd_from_graph, wasserstein_distance
from proteintda.utils.conversions import Atom37, SideChainAtom, atom_positions_from_atom37


def _distance_matrix(positions: torch.Tensor) -> torch.Tensor:
    """Full pairwise distance matrix, shape (n, n)."""
    dists = torch.cdist(positions, positions)
    eye = torch.eye(dists.shape[0], device=dists.device, dtype=dists.dtype)
    return dists * (1.0 - eye)


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
        pred_diags: list[torch.Tensor],
        target_diags: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
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
        pred_diags = pd_from_graph(pred_adj)
        target_diags = pd_from_graph(target_adj)
        for name, loss in self._term_losses(pred_diags, target_diags).items():
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

    def from_points(
        self,
        pred_xyz: torch.Tensor,
        target_xyz: torch.Tensor,
        *,
        _return_breakdown: bool = False,
    ):
        """TDA loss from coordinate clouds (n, 3)."""
        n = min(pred_xyz.shape[0], target_xyz.shape[0])
        pred_adj = _distance_matrix(pred_xyz[:n])
        target_adj = _distance_matrix(target_xyz[:n]).detach()
        return self._loss_from_adjs(pred_adj, target_adj, _return_breakdown=_return_breakdown)

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
