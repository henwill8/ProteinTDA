"""
Build distance matrices from SidechainNet protein residues (SCNProtein).

Residues without a valid CB (e.g. glycine) use CA instead.
Matrix entries are Euclidean distances (angstroms) between included residues.
"""

import torch
from sidechainnet.dataloaders.SCNProtein import SCNProtein

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
        # TODO: ensure this actually gets CA if CB is missing
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
) -> tuple[torch.Tensor, torch.Tensor]:
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

    return all_atom_positions, all_atom_mask


def distance_matrix(positions: torch.Tensor) -> torch.Tensor:
    """Full pairwise distance matrix, shape (n, n)."""
    return torch.cdist(positions, positions).requires_grad_()
