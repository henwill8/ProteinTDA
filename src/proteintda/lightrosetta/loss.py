import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from proteintda.lightrosetta.path import ensure_lightrosetta_on_path
from proteintda.shared.loss import TDALoss, _as_scalar
from proteintda.utils.conversions import SideChainAtom

ensure_lightrosetta_on_path()

from utils.LDDT_torch import lddt as calculate_lddt
from utils.loss_func import (
    harmonic_angle_force_loss,
    harmonic_bond_force_loss,
    periodic_torsion_force_loss,
)
from utils.rigid_transform import transform_pred_coor_to_label_coor


def _to_tensor(x) -> torch.Tensor:
    while isinstance(x, (list, tuple)):
        x = x[0]
    return x if torch.is_tensor(x) else torch.as_tensor(x)


def _cross_loss_mask(pred, true, mask, device: torch.device) -> torch.Tensor:
    """CE with LxL crop + invalid-bin clamp (official .pt quirks)."""
    if pred.dim() == 4:
        L = min(pred.shape[1], true.shape[-1], mask.shape[-1])
        pred = pred[:, :L, :L]
        true = true[..., :L, :L]
        mask = mask[..., :L, :L]
    elif pred.dim() == 3:
        L = min(pred.shape[0], true.shape[-1], mask.shape[-1])
        pred = pred[:L, :L]
        true = true[..., :L, :L]
        mask = mask[..., :L, :L]

    n_classes = pred.shape[-1]
    pred = pred.reshape(-1, n_classes)
    true = torch.flatten(true).to(device=device).long()
    mask = torch.flatten(mask).float().to(device=device)
    n = min(pred.shape[0], true.numel(), mask.numel())
    pred, true, mask = pred[:n], true.view(-1)[:n], mask.view(-1)[:n]

    invalid = (true < 0) | (true >= n_classes)
    if invalid.any():
        true = true.clone()
        mask = mask.clone()
        true[invalid] = 0
        mask[invalid] = 0.0

    loss = F.cross_entropy(pred, true, reduction="none")
    return (mask * loss).sum() / mask.sum().clamp_min(1.0)


def _pair_labels(data, device: torch.device):
    pp = data.pair_prob_s_label
    if isinstance(pp[0], (list, tuple)):
        pp = pp[0]
    labels = [_to_tensor(pp[i]).to(device=device, dtype=torch.long) for i in range(4)]
    pm = data.pair_masks
    if isinstance(pm, (list, tuple)):
        pm = pm[0]
    mask = _to_tensor(pm).to(device=device)
    return labels, mask


def _ca_coords_from_data(data, device: torch.device) -> torch.Tensor:
    if getattr(data, "ca_coords", None) is not None:
        return data.ca_coords.to(device)
    return torch.index_select(data.pos, 0, data.CA_atom_index.long()).to(device)


class LightRoseTTALoss:
    """Native LightRoseTTA terms plus optional TDA via shared TDALoss."""

    def __init__(
        self,
        loss_config,
        h0rff=None,
        h1rff=None,
        *,
        tda_atom: SideChainAtom = SideChainAtom.CA,
    ):
        self.loss_config = loss_config
        self.native = loss_config.lightrosetta
        self._tda = (
            TDALoss(loss_config, h0rff=h0rff, h1rff=h1rff, tda_atom=tda_atom)
            if loss_config.tda.enabled
            else None
        )
        self.tda_atom = tda_atom

    @property
    def tda_enabled(self) -> bool:
        return self._tda is not None and bool(self._tda._enabled)

    def _enabled(self, name: str) -> bool:
        term = self.native.get(name)
        return term is not None and bool(term.enabled)

    def _weight(self, name: str) -> float:
        return float(self.native[name].weight)

    def compute(
        self,
        xyz: torch.Tensor,
        lddt_pred: torch.Tensor,
        logits: list[torch.Tensor],
        data,
        *,
        device: torch.device,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]] | None:
        args = SimpleNamespace(device=device)
        log: dict[str, float] = {}
        total = xyz.new_zeros(())

        dis_pred, omega_pred, theta_pred, phi_pred = logits
        (dis_label, omega_label, theta_label, phi_label), dis_mask = _pair_labels(data, device)

        for name, pred, label in (
            ("distogram", dis_pred, dis_label),
            ("omega", omega_pred, omega_label),
            ("theta", theta_pred, theta_label),
            ("phi", phi_pred, phi_label),
        ):
            if self._enabled(name):
                loss = _cross_loss_mask(pred.float(), label, dis_mask, device)
                weighted = self._weight(name) * loss
                log[name] = _as_scalar(weighted)
                total = total + weighted

        ca_idx = data.CA_atom_index.long().view(-1).to(device)
        pos = data.pos.to(device)
        n_pos = pos.shape[0]
        if ca_idx.numel() == 0 or (ca_idx - 1).min().item() < 0 or (ca_idx + 1).max().item() >= n_pos:
            print(f"skip bad CA_atom_index (n_pos={n_pos})")
            return None

        n_target = torch.index_select(pos, 0, ca_idx - 1)
        ca_target = torch.index_select(pos, 0, ca_idx)
        c_target = torch.index_select(pos, 0, ca_idx + 1)
        bb_pos = torch.stack([n_target, ca_target, c_target], dim=1).reshape(-1, 3)
        xyz_flat = xyz.reshape(-1, 3)
        if xyz_flat.shape[0] != bb_pos.shape[0]:
            print(
                f"skip coord len mismatch pred={xyz_flat.shape[0]} lab={bb_pos.shape[0]}"
            )
            return None
        if (not torch.isfinite(xyz_flat).all()) or (not torch.isfinite(bb_pos).all()):
            print("skip nonfinite coords")
            return None

        try:
            new_out_coor = transform_pred_coor_to_label_coor(xyz_flat, bb_pos, args)
        except ValueError as exc:
            print(f"skip rigid transform ({exc})")
            return None

        if self._enabled("coor"):
            loss = torch.sqrt(F.mse_loss(new_out_coor, bb_pos) + 1e-8)
            weighted = self._weight("coor") * loss
            log["coor"] = _as_scalar(weighted)
            total = total + weighted

        for name, fn, args_t in (
            (
                "bond",
                harmonic_bond_force_loss,
                (new_out_coor, data.bond_value.to(device), data.bond_index.T.to(device)),
            ),
            (
                "angle",
                harmonic_angle_force_loss,
                (new_out_coor, data.angle_value.to(device), data.angle_index.to(device)),
            ),
        ):
            if self._enabled(name):
                loss = fn(*args_t)
                scale = (epoch + 1) * float(self.native[name].get("epoch_scale", 0.005))
                weighted = self._weight(name) * scale * loss
                log[name] = _as_scalar(weighted)
                total = total + weighted

        if self._enabled("dihedral") and data.dihedral_index.numel() > 0:
            loss = periodic_torsion_force_loss(
                new_out_coor, bb_pos, data.dihedral_index.to(device)
            )
            scale = (epoch + 1) * float(self.native.dihedral.get("epoch_scale", 0.005))
            weighted = self._weight("dihedral") * scale * loss
            log["dihedral"] = _as_scalar(weighted)
            total = total + weighted

        if self._enabled("plddt_loss"):
            ca_out = new_out_coor[1 : new_out_coor.shape[0] - 1 : 3]
            lddt_true = calculate_lddt(
                ca_out.unsqueeze(0), ca_target.unsqueeze(0), args, 15.0, True
            ).to(device)
            loss = torch.sqrt(
                F.mse_loss(lddt_pred.squeeze().float(), lddt_true.squeeze()) + 1e-8
            )
            weighted = self._weight("plddt_loss") * loss
            log["plddt_loss"] = _as_scalar(weighted)
            total = total + weighted

        if self.tda_enabled:
            ca_out = new_out_coor[1::3]
            target_xyz = _ca_coords_from_data(data, device)
            tda_loss, tda_breakdown = self._tda.from_points(
                ca_out, target_xyz, _return_breakdown=True
            )
            weighted_tda = self.loss_config.tda.weight * tda_loss
            total = total + weighted_tda
            for name, value in tda_breakdown.items():
                if name == "loss":
                    continue
                term_weight = self.loss_config.tda.terms[name].weight
                log[name] = _as_scalar(self.loss_config.tda.weight * term_weight * value)

        if not log:
            raise ValueError("No LightRoseTTA loss terms are enabled")

        L = float(data.msa.shape[-1])
        total = total * math.sqrt(L)
        log["total"] = _as_scalar(total)
        return total, log
