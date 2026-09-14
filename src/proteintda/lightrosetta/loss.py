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
    cross_loss_mask,
    harmonic_angle_force_loss,
    harmonic_bond_force_loss,
    periodic_torsion_force_loss,
)
from utils.rigid_transform import transform_pred_coor_to_label_coor


class LightRoseTTALoss:
    """Native LightRoseTTA terms plus optional TDA via shared TDALoss."""

    def __init__(self, loss_config, h0rff=None, h1rff=None, *, tda_atom: SideChainAtom = SideChainAtom.CB):
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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        args = SimpleNamespace(device=device)
        log: dict[str, float] = {}
        total = xyz.new_zeros(())

        dis_pred, omega_pred, theta_pred, phi_pred = logits
        dis_label = data.pair_prob_s_label[0][0].to(device=device, dtype=torch.long)
        omega_label = data.pair_prob_s_label[1][0].to(device=device, dtype=torch.long)
        theta_label = data.pair_prob_s_label[2][0].to(device=device, dtype=torch.long)
        phi_label = data.pair_prob_s_label[3][0].to(device=device, dtype=torch.long)
        dis_mask = data.pair_masks[0].to(device=device)

        for name, pred, label in (
            ("distogram", dis_pred, dis_label),
            ("omega", omega_pred, omega_label),
            ("theta", theta_pred, theta_label),
            ("phi", phi_pred, phi_label),
        ):
            if self._enabled(name):
                loss = cross_loss_mask(pred.float(), label, dis_mask, device)
                weighted = self._weight(name) * loss
                log[name] = _as_scalar(weighted)
                total = total + weighted

        n_target = torch.index_select(data.pos, 0, data.CA_atom_index - 1)
        ca_target = torch.index_select(data.pos, 0, data.CA_atom_index)
        c_target = torch.index_select(data.pos, 0, data.CA_atom_index + 1)
        bb_pos = torch.stack([n_target, ca_target, c_target], dim=1).reshape(-1, 3)
        new_out_coor = transform_pred_coor_to_label_coor(xyz, bb_pos, args)

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
            loss = torch.sqrt(F.mse_loss(lddt_pred.squeeze().float(), lddt_true.squeeze()) + 1e-8)
            weighted = self._weight("plddt_loss") * loss
            log["plddt_loss"] = _as_scalar(weighted)
            total = total + weighted

        if self.tda_enabled:
            target_xyz = (
                data.ca_coords if self.tda_atom is SideChainAtom.CA else data.cb_coords
            ).to(device)
            pred_xyz = new_out_coor[1::3]
            tda_loss, tda_breakdown = self._tda.from_points(
                pred_xyz, target_xyz, _return_breakdown=True
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

        total = total * math.sqrt(float(data.msa.shape[-1]))
        log["total"] = _as_scalar(total)
        return total, log
