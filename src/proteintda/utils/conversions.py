from enum import Enum

import torch
from minifold.utils.residue_constants import atom_order
from sidechainnet.dataloaders.SCNProtein import SCNProtein


class Atom14(Enum):
    CA = 1
    CB = 4

class Atom37(Enum):
    CA = 1
    CB = 3

class SideChainAtom(Enum):
    CA = 1
    CB = 5


def atom_positions_from_atom14(
    positions: torch.Tensor,
    atom: Atom14,
    atom_exists: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract atom14 positions, falling back to CA when the atom is missing."""
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


def atom_positions_from_atom37(
    positions: torch.Tensor,
    atom_mask: torch.Tensor,
    atom: Atom37,
) -> torch.Tensor:
    """Extract atom37 positions, falling back to CA when the atom is missing."""
    atom_idx = atom.value
    fallback_idx = Atom37.CA.value
    coords: list[torch.Tensor] = []
    for res_idx in range(positions.shape[0]):
        if atom_mask[res_idx, atom_idx] > 0.5:
            coords.append(positions[res_idx, atom_idx])
        else:
            coords.append(positions[res_idx, fallback_idx])
    if not coords:
        raise ValueError("No valid atom coordinates in atom37 tensor.")
    return torch.stack(coords)


def atom_positions_from_sidechainnet(
    protein: SCNProtein,
    atom: SideChainAtom,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Extract SidechainNet atom positions, falling back to CA when missing."""
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
    dtype: torch.dtype = torch.float32,
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
            if atom_name == "PAD" or atom_name not in atom_order:
                continue
            atom_pos = res_coords[atom_idx]
            if torch.isnan(atom_pos).any():
                continue
            atom37_idx = atom_order[atom_name]
            all_atom_positions[res_idx, atom37_idx] = atom_pos.to(device)
            all_atom_mask[res_idx, atom37_idx] = 1.0
            seq_mask[res_idx] = 1.0

    return all_atom_positions, all_atom_mask, seq_mask
