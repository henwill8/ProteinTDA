import torch
import openfold.data.data_transforms as data_transforms
from transformers.models.esm.modeling_esmfold import EsmForProteinFoldingOutput
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from openfold.utils.tensor_utils import batched_gather
from openfold.np import residue_constants as rc 

from enum import Enum


class SideChainAtom(Enum):
    CA = 1
    CB = 5


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

def distance_matrix(positions: torch.Tensor) -> torch.Tensor:
    """Full pairwise distance matrix, shape (n, n)."""
    return torch.cdist(positions, positions).requires_grad_()

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


def atom37_to_atom14(atom_37: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    atom14_data = batched_gather(
                atom_37,
                batch["residx_atom14_to_atom37"],
                dim=-2,
                no_batch_dims=len(atom37.shape[:-2]),
            )
    atom14_data = atom14_data * batch["atom14_atom_exists"][..., None]

    return atom14_data

def out_conversion(
            esm_out: EsmForProteinFoldingOutput,
            sc_protein: SCNProtein,
            device
            ):
    out = {}
    out["sm"] = {}
    out["sm"]["frames"] = esm_out.frames
    out["sm"]["sidechain_frames"] = esm_out.sidechain_frames
    out["sm"]["positions"] = esm_out.positions
    out["sm"]["angles"] = esm_out.angles
    out["sm"]["unnormalized_angles"] = esm_out.unnormalized_angles
    out["tm_logits"] = esm_out.ptm_logits
    out["lddt_logits"] = esm_out.lddt_head[-1,:,:,1,:] # <- Looking at lddt loss in openfold, they extract C_Alpha which is why there is the 1. The -1 extracts the final iteration.
    out["distogram_logits"] = esm_out.distogram_logits
    out["final_affine_tensor"] = out["sm"]["frames"][-1]
    # ? experimentally_resolved_logits
    # ? masked_msa_logits

    batch = {}
    # batch from sc_protein
    batch["aatype"] = torch.tensor(
            [rc.restype_order_with_x.get(aa, rc.restype_num) for aa in sc_protein.seq], #rc.restype_num is 20, line 878 of residue constants - seems to be a fallback value
            dtype=torch.long,
            )
    batch["seq_length"] = torch.tensor(len(sc_protein.seq))
    batch["residue_index"] = torch.arange(end=batch["seq_length"], device=device)
    batch = data_transforms.make_atom14_masks(batch) # <- This will make residx atom14_to_atom37 and atom37_to_atom14 along with atom37_atom_exists and atom14_atom_exists
    batch["all_atom_positions"], batch["all_atom_mask"], batch["seq_mask"] = sidechainnet_to_atom37(sc_protein, device)
    batch = data_transforms.make_atom14_positions(batch) # <- Makes atom14_gt_positions, atom14_gt_exists, atom14_alt_gt_exists, atom14_alt_gt_positions, atom14_atom_is_ambiguous
    batch = data_transforms.atom37_to_frames(batch) # <- Gets input for sidechain FAPE loss 
    batch = data_transforms.get_backbone_frames(batch) # <- Gets input for backbone FAPE loss
    batch = data_transforms.atom37_to_torsion_angles(batch) # <- Prepares us for chi angles
    batch = data_transforms.get_chi_angles(batch) # <- Everything for supervised chi loss
    batch["resolution"] = torch.tensor(sc_protein.resolution) # <- Used in pLDDT loss
    batch = data_transforms.make_pseudo_beta(protein) # <- Preparation for Distorgram Loss, pTM loss


    out["final_atom_positions"] = atom14_to_atom37(out["sm"]["positions"][-1], batch)
    return out, batch
