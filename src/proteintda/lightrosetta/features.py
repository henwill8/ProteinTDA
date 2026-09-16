from functools import lru_cache

import torch
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from torch_geometric.data import Data

from proteintda.lightrosetta.path import ensure_lightrosetta_on_path
from proteintda.utils.conversions import SideChainAtom, atom_positions_from_sidechainnet

ensure_lightrosetta_on_path()

from utils.kinematics import c6d_to_bins2, xyz_to_c6d  # noqa: E402

_MSA_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
_AA_TO_MSA = {c: i for i, c in enumerate(_MSA_ALPHABET)}
_ATOM_ONE_HOT = {"C": [1, 0, 0, 0], "N": [0, 1, 0, 0], "O": [0, 0, 1, 0], "S": [0, 0, 0, 1]}

_RIGID_BB = {
    0: [[-0.525, 1.363, 0.000], [0.000, 0.000, 0.000], [1.526, -0.000, -0.000]],
    1: [[-0.524, 1.362, -0.000], [0.000, 0.000, 0.000], [1.525, -0.000, -0.000]],
    2: [[-0.536, 1.357, 0.000], [0.000, 0.000, 0.000], [1.526, -0.000, -0.000]],
    3: [[-0.525, 1.362, -0.000], [0.000, 0.000, 0.000], [1.527, 0.000, -0.000]],
    4: [[-0.522, 1.362, -0.000], [0.000, 0.000, 0.000], [1.524, 0.000, 0.000]],
    5: [[-0.526, 1.361, -0.000], [0.000, 0.000, 0.000], [1.526, 0.000, 0.000]],
    6: [[-0.528, 1.361, 0.000], [0.000, 0.000, 0.000], [1.526, -0.000, -0.000]],
    7: [[-0.572, 1.337, 0.000], [0.000, 0.000, 0.000], [1.517, -0.000, -0.000]],
    8: [[-0.527, 1.360, 0.000], [0.000, 0.000, 0.000], [1.525, 0.000, 0.000]],
    9: [[-0.493, 1.373, -0.000], [0.000, 0.000, 0.000], [1.527, -0.000, -0.000]],
    10: [[-0.520, 1.363, 0.000], [0.000, 0.000, 0.000], [1.525, -0.000, -0.000]],
    11: [[-0.526, 1.362, -0.000], [0.000, 0.000, 0.000], [1.526, 0.000, 0.000]],
    12: [[-0.521, 1.364, -0.000], [0.000, 0.000, 0.000], [1.525, 0.000, 0.000]],
    13: [[-0.518, 1.363, 0.000], [0.000, 0.000, 0.000], [1.524, 0.000, -0.000]],
    14: [[-0.566, 1.351, -0.000], [0.000, 0.000, 0.000], [1.527, -0.000, 0.000]],
    15: [[-0.529, 1.360, -0.000], [0.000, 0.000, 0.000], [1.525, -0.000, -0.000]],
    16: [[-0.517, 1.364, 0.000], [0.000, 0.000, 0.000], [1.526, 0.000, -0.000]],
    17: [[-0.521, 1.363, 0.000], [0.000, 0.000, 0.000], [1.525, -0.000, 0.000]],
    18: [[-0.522, 1.362, 0.000], [0.000, 0.000, 0.000], [1.524, -0.000, -0.000]],
    19: [[-0.494, 1.373, -0.000], [0.000, 0.000, 0.000], [1.527, -0.000, -0.000]],
}


def _one_hot(index: int, total: int) -> list[int]:
    return [1 if i == index else 0 for i in range(total)]


def _seq_to_msa_tokens(seq: str) -> torch.Tensor:
    tokens = [_AA_TO_MSA.get(c, 20) for c in seq]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


@lru_cache(maxsize=1)
def _load_residue_templates() -> dict[str, dict]:
    from atom_graph.generate_protein_graph import read_residue_graph

    root = ensure_lightrosetta_on_path() / "atom_graph" / "amino acid"
    return read_residue_graph(str(root))


def _build_atom_graph(seq: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    from atom_graph.generate_protein_graph import build_graph

    templates = _load_residue_templates()
    node_lines, edge_lines = build_graph(seq, templates, "all atom")
    if not node_lines:
        raise ValueError(f"Failed to build atom graph for sequence length {len(seq)}")

    features: list[list[float]] = []
    atom_names: list[str] = []
    ca_indices: list[int] = []
    node_count = 0
    for j, line in enumerate(node_lines):
        parts = line.strip().split(" ")
        atom_name = parts[0]
        atom_names.append(atom_name)
        if atom_name == "N":
            node_count = 0
        else:
            node_count += 1
        if atom_name == "CA":
            ca_indices.append(j)

        init_xyz = [float(v) for v in parts[1].split(",")]
        residue_type = int(parts[2])
        backbone_flag = int(parts[3])
        element = atom_name[0] if atom_name[0] in _ATOM_ONE_HOT else "C"
        feat = list(_ATOM_ONE_HOT[element])
        feat.extend(_one_hot(residue_type, 20))
        feat.extend(_one_hot(backbone_flag, 2))
        feat.extend(_one_hot(node_count, 14))
        feat.extend(init_xyz)
        features.append(feat)

    src: list[int] = []
    dst: list[int] = []
    for line in edge_lines:
        line = line.strip()
        if not line:
            continue
        vals = [int(v) for v in line.split()]
        src.append(vals[0])
        dst.append(vals[1])

    x = torch.tensor(features, dtype=torch.float32)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    ca_atom_index = torch.tensor(ca_indices, dtype=torch.long)
    return x, edge_index, ca_atom_index, atom_names


def _dummy_templates(msa: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq = msa[0]
    L = seq.numel()
    unfolded = []
    for token in seq.tolist():
        key = token if token in _RIGID_BB else 0
        unfolded.append(_RIGID_BB[key])
    xyz_t = torch.tensor([unfolded], dtype=torch.float32)
    t0d = torch.ones(1, 3, dtype=torch.float32)
    t1d = torch.ones(1, L, 3, dtype=torch.float32)
    return xyz_t, t0d, t1d


def _scn_coords(protein: SCNProtein) -> tuple[torch.Tensor, list[list[str]]]:
    coords = protein.coords
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().float()
    else:
        coords = torch.as_tensor(coords, dtype=torch.float32)
    return coords, protein.get_atom_names()


def _all_atom_pos_from_scn(
    protein: SCNProtein,
    atom_names: list[str],
) -> torch.Tensor:
    coords, names_per_res = _scn_coords(protein)
    mask = str(protein.mask)
    pos = torch.zeros(len(atom_names), 3, dtype=torch.float32)

    res_idx = -1
    for atom_i, atom_name in enumerate(atom_names):
        if atom_name == "N":
            res_idx += 1
            while res_idx < len(mask) and mask[res_idx] != "+":
                res_idx += 1
        name_map = {n: j for j, n in enumerate(names_per_res[res_idx])}
        if atom_name in name_map:
            atom_pos = coords[res_idx, name_map[atom_name]]
            if not torch.isnan(atom_pos).any():
                pos[atom_i] = atom_pos
    return pos


def _fill_missing_pos_from_features(pos: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    missing = pos.abs().sum(dim=-1) == 0
    if missing.any():
        pos = pos.clone()
        pos[missing] = x[missing, -3:]
    return pos


def _backbone_ncac(pos: torch.Tensor, ca_atom_index: torch.Tensor) -> torch.Tensor:
    n = pos[ca_atom_index - 1]
    ca = pos[ca_atom_index]
    c = pos[ca_atom_index + 1]
    return torch.stack([n, ca, c], dim=1)


def _pair_labels_from_backbone(bb: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    c6d, mask = xyz_to_c6d(bb.unsqueeze(0))
    bins = c6d_to_bins2(c6d)[0]  # (L, L, 4)
    pair_prob_s_label = [[bins[..., i].long()] for i in range(4)]
    pair_masks = [mask[0]]
    return pair_prob_s_label, pair_masks


def _simple_bpe_from_backbone(bb_flat: torch.Tensor) -> dict[str, torch.Tensor]:
    n_atoms = bb_flat.shape[0]
    src = list(range(n_atoms - 1))
    dst = list(range(1, n_atoms))
    bond_index = torch.tensor([src, dst], dtype=torch.long).T
    bond_value = torch.norm(bb_flat[bond_index[:, 0]] - bb_flat[bond_index[:, 1]], dim=-1) / 10.0

    angle_index = []
    angle_value = []
    for i in range(1, n_atoms - 1):
        v1 = bb_flat[i - 1] - bb_flat[i]
        v2 = bb_flat[i + 1] - bb_flat[i]
        cos = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
        angle_index.append([i - 1, i, i + 1])
        angle_value.append(torch.acos(torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)))
    angle_index_t = torch.tensor(angle_index, dtype=torch.long) if angle_index else torch.zeros(0, 3, dtype=torch.long)
    angle_value_t = torch.stack(angle_value) if angle_value else torch.zeros(0, dtype=torch.float32)

    dihedral_index = []
    for i in range(n_atoms - 3):
        dihedral_index.append([i, i + 1, i + 2, i + 3])
    dihedral_index_t = (
        torch.tensor(dihedral_index, dtype=torch.long) if dihedral_index else torch.zeros(0, 4, dtype=torch.long)
    )
    return {
        "bond_index": bond_index,
        "bond_value": bond_value,
        "angle_index": angle_index_t,
        "angle_value": angle_value_t,
        "dihedral_index": dihedral_index_t,
    }


def protein_to_lightrosetta_data(protein: SCNProtein) -> Data:
    seq = "".join(c for c, m in zip(str(protein.seq), str(protein.mask)) if m == "+")
    if not seq:
        raise ValueError(f"Protein {getattr(protein, 'id', '?')} has no complete residues")

    msa = _seq_to_msa_tokens(seq)
    xyz_t, t0d, t1d = _dummy_templates(msa)
    x, edge_index, ca_atom_index, atom_names = _build_atom_graph(seq)
    pos = _fill_missing_pos_from_features(_all_atom_pos_from_scn(protein, atom_names), x)

    bb = _backbone_ncac(pos, ca_atom_index)
    pair_prob_s_label, pair_masks = _pair_labels_from_backbone(bb)
    bb_flat = bb.reshape(-1, 3)
    bpe = _simple_bpe_from_backbone(bb_flat)
    ca_coords = atom_positions_from_sidechainnet(protein, SideChainAtom.CA)
    cb_coords = atom_positions_from_sidechainnet(protein, SideChainAtom.CB)

    return Data(
        x=x,
        edge_index=edge_index,
        pos=pos,
        CA_atom_index=ca_atom_index,
        protein_name=getattr(protein, "id", "unknown"),
        msa=msa,
        xyz_t=xyz_t,
        t0d=t0d,
        t1d=t1d,
        pair_prob_s_label=pair_prob_s_label,
        pair_masks=pair_masks,
        bond_index=bpe["bond_index"],
        angle_index=bpe["angle_index"],
        dihedral_index=bpe["dihedral_index"],
        bond_value=bpe["bond_value"],
        angle_value=bpe["angle_value"],
        ca_coords=ca_coords,
        cb_coords=cb_coords,
        seq=seq,
        num_features=x.shape[1],
        num_classes=3,
    )
