import torch
# import openfold.data.data_transforms as data_transforms
# from transformers.models.esm.modeling_esmfold import EsmForProteinFoldingOutput
# from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from sidechainnet.dataloaders.SCNProtein import SCNProtein
# from openfold.utils.tensor_utils import batched_gather
# from openfold.np import residue_constants as rc

from enum import Enum

# OpenFold atom14 indices
class Atom14(Enum):
    CA = 1
    CB = 4


class SideChainAtom(Enum):
    CA = 1
    CB = 5


def atom_positions_from_atom14(
    positions: torch.Tensor,
    atom: Atom14,
    atom_exists: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extracts the positions of the given atom from the ESMFold atom14 output."""
    coords: list[torch.Tensor] = []
    length = positions.shape[0]
    for i in range(length):
        atom_pos = positions[i, atom.value]
        if atom_exists is not None and atom_exists[i, atom.value] < 0.5:
            atom_pos = positions[i, Atom14.CA.value]
        coords.append(atom_pos)
    if not coords:
        raise ValueError("No valid atom coordinates in model output.")
    return torch.stack(coords)


def atom_positions_from_sidechainnet(
    protein: SCNProtein,
    atom: SideChainAtom,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Extracts the positions of the given atom from SidechainNet coordinates."""
    coords = protein.coords
    if isinstance(coords, torch.Tensor):
        coords = coords.detach()
    else:
        coords = torch.as_tensor(coords, dtype=dtype)

    mask = str(protein.mask)
    positions: list[torch.Tensor] = []
    for i, char in enumerate(mask):
        if char != "+":
            continue
        atom_pos = coords[i, atom.value]
        if torch.isnan(atom_pos).any():
            atom_pos = coords[i, SideChainAtom.CA.value]
        positions.append(atom_pos)

    if not positions:
        raise ValueError("No valid atom coordinates in SidechainNet protein.")
    out = torch.stack(positions).to(dtype=dtype)
    if device is not None:
        out = out.to(device)
    return out


def sidechainnet_to_atom37(
    protein: SCNProtein,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = protein.coords
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().to(dtype=dtype)
    else:
        coords = torch.as_tensor(coords, dtype=dtype)

    mask = str(protein.mask)
    atom_names_per_residue = protein.get_atom_names()
    length = len(mask)
    all_atom_positions = torch.zeros(length, 37, 3, dtype=dtype, device=device)
    all_atom_mask = torch.zeros(length, 37, dtype=dtype, device=device)
    seq_mask = torch.zeros(length, dtype=dtype, device=device)

    for res_idx, (mask_char, atom_names) in enumerate(zip(mask, atom_names_per_residue)):
        if mask_char != "+":
            continue
        res_coords = coords[res_idx]
        for atom_idx, atom_name in enumerate(atom_names):
            # PAD is SidechainNet's filler for empty slots (the coordinate array is fixed)
            if atom_name == "PAD" or atom_name not in rc.atom_order:
                continue
            atom_pos = res_coords[atom_idx]
            if torch.isnan(atom_pos).any():
                continue
            atom37_idx = rc.atom_order[atom_name]
            all_atom_positions[res_idx, atom37_idx] = atom_pos.to(device)
            all_atom_mask[res_idx, atom37_idx] = 1.0
            seq_mask[res_idx] = 1.0

    return all_atom_positions, all_atom_mask, seq_mask


# def atom37_to_atom14(atom37: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
#     atom14_data = batched_gather(
#                 atom37,
#                 batch["residx_atom14_to_atom37"],
#                 dim=-2,
#                 no_batch_dims=len(atom37.shape[:-2]),
#             )
#     atom14_data = atom14_data * batch["atom14_atom_exists"][..., None]

#     return atom14_data


def distance_matrix(positions: torch.Tensor) -> torch.Tensor:
    """Full pairwise distance matrix, shape (n, n)."""
    dists = torch.cdist(positions, positions).clone() # Do we need the full matrix or can we use the upper triangle?
    dists.fill_diagonal_(0.0)
    return dists


# def _loss_field_requirements(loss_config) -> dict[str, bool]:
#     cfg = loss_config
#     needs_tda = (
#         cfg.wasserstein_h0.enabled
#         or cfg.wasserstein_h1.enabled
#         or cfg.vpd_h0.enabled
#         or cfg.vpd_h1.enabled
#     )
#     needs_openfold_batch = any(
#         cfg[name].enabled
#         for name in (
#             "distogram",
#             "fape",
#             "plddt_loss",
#             "supervised_chi",
#             "violation",
#             "tm",
#             "chain_center_of_mass",
#         )
#     )
#     needs_final_atom_positions = cfg.plddt_loss.enabled or cfg.chain_center_of_mass.enabled
#     return {
#         "tda": needs_tda,
#         "openfold_batch": needs_openfold_batch,
#         "distogram": cfg.distogram.enabled,
#         "fape": cfg.fape.enabled,
#         "plddt": cfg.plddt_loss.enabled,
#         "chi": cfg.supervised_chi.enabled,
#         "violation": cfg.violation.enabled,
#         "tm": cfg.tm.enabled,
#         "needs_sm_positions": (
#             cfg.fape.enabled
#             or cfg.violation.enabled
#             or needs_final_atom_positions
#         ),
#         "needs_sm_frames": cfg.fape.enabled,
#         "needs_sm_angles": cfg.supervised_chi.enabled,
#         "needs_fape_frames_batch": cfg.fape.enabled,
#         "needs_chi_batch": cfg.supervised_chi.enabled,
#         "needs_resolution": cfg.plddt_loss.enabled,
#         "needs_pseudo_beta": cfg.distogram.enabled or cfg.tm.enabled,
#         "needs_final_atom_positions": needs_final_atom_positions,
#     }


# def _base_batch(sc_protein: SCNProtein, device: torch.device) -> dict[str, torch.Tensor]:
#     length = len(sc_protein.seq)
#     return {
#         "aatype": torch.tensor(
#             [rc.restype_order_with_x.get(aa, rc.restype_num) for aa in sc_protein.seq],
#             dtype=torch.long,
#             device=device,
#         ),
#         "seq_length": torch.tensor(length, device=device),
#         "residue_index": torch.arange(length, device=device),
#     }


# def _add_tda_fields(
#     out: dict,
#     batch: dict[str, torch.Tensor],
#     esm_out: EsmForProteinFoldingOutput,
#     sc_protein: SCNProtein,
#     *,
#     device: torch.device,
#     tda_atom: SideChainAtom,
# ) -> None:
#     atom_exists = esm_out.atom14_atom_exists[0] if esm_out.atom14_atom_exists is not None else None
#     atom14_atom = Atom14.CB if tda_atom is SideChainAtom.CB else Atom14.CA
#     pred_positions = atom_positions_from_atom14(
#         esm_out.positions[-1][0],
#         atom14_atom,
#         atom_exists,
#     )
#     target_positions = atom_positions_from_sidechainnet(
#         sc_protein,
#         tda_atom,
#         device=device,
#     )
#     out["adj"] = distance_matrix(pred_positions)
#     batch["adj"] = distance_matrix(target_positions).detach()


# def pre_loss_conversion(
#     esm_out: EsmForProteinFoldingOutput,
#     sc_protein: SCNProtein,
#     *,
#     device: torch.device,
#     loss_config,
#     tda_atom: SideChainAtom = SideChainAtom.CB,
# ):
#     req = _loss_field_requirements(loss_config)
#     batch = _base_batch(sc_protein, device)
#     out: dict = {}

#     if req["openfold_batch"]:
#         batch = data_transforms.make_atom14_masks(batch)
#         batch["all_atom_positions"], batch["all_atom_mask"], batch["seq_mask"] = sidechainnet_to_atom37(
#             sc_protein, device
#         )
#         batch = data_transforms.make_atom14_positions(batch)
#         if req["needs_fape_frames_batch"]:
#             batch = data_transforms.atom37_to_frames(batch)
#             batch = data_transforms.get_backbone_frames(batch)
#         if req["needs_chi_batch"]:
#             batch = data_transforms.atom37_to_torsion_angles("")(batch)
#             batch = data_transforms.get_chi_angles(batch)
#         if req["needs_resolution"]:
#             batch["resolution"] = torch.tensor(
#                 sc_protein.resolution if sc_protein.resolution is not None else 0.0,
#                 device=device,
#             )
#         if req["needs_pseudo_beta"]:
#             batch = data_transforms.make_pseudo_beta("")(batch)

#         batch = {key: value.unsqueeze(0) for key, value in batch.items()}

#         if req["needs_sm_positions"] or req["needs_sm_frames"] or req["needs_sm_angles"]:
#             out["sm"] = {}
#             if req["needs_sm_positions"]:
#                 out["sm"]["positions"] = esm_out.positions
#             if req["needs_sm_frames"]:
#                 out["sm"]["frames"] = esm_out.frames
#                 out["sm"]["sidechain_frames"] = esm_out.sidechain_frames
#                 out["final_affine_tensor"] = esm_out.frames[-1]
#             if req["needs_sm_angles"]:
#                 out["sm"]["angles"] = esm_out.angles
#                 out["sm"]["unnormalized_angles"] = esm_out.unnormalized_angles

#         if req["distogram"]:
#             out["distogram_logits"] = esm_out.distogram_logits
#         if req["plddt"]:
#             out["lddt_logits"] = esm_out.lddt_head[-1, :, :, 1, :]
#         if req["tm"]:
#             out["tm_logits"] = esm_out.ptm_logits
#         if req["needs_final_atom_positions"]:
#             out["final_atom_positions"] = atom14_to_atom37(out["sm"]["positions"][-1], batch)
#     else:
#         batch = {key: value.unsqueeze(0) for key, value in batch.items()}

#     if req["tda"]:
#         _add_tda_fields(out, batch, esm_out, sc_protein, device=device, tda_atom=tda_atom)

#     return out, batch
