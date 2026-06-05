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


def distance_matrix(positions: torch.Tensor) -> torch.Tensor:
    """Full pairwise distance matrix, shape (n, n)."""
    return torch.cdist(positions, positions).requires_grad_()
