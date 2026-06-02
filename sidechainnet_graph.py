"""
Build distance-weighted graphs from SidechainNet protein (SCNProtein).

Residues without a finite C_beta (e.g. glycine) use C_alpha instead.
Graph nodes are one per included residue; edge weights are Euclidean distances (angstroms).
"""

import networkx as nx
import numpy as np

from sidechainnet.dataloaders.SCNProtein import SCNProtein

_CA_ATOM_INDEX = 1
_CB_ATOM_INDEX = 5


def scn_protein_to_graph(
    protein: SCNProtein,
    *,
    contact_cutoff: float | None = None,
    allow_incomplete: bool = False,
) -> nx.Graph:
    """
    Build an undirected distance graph from an ``SCNProtein``.

    Parameters
    ----------
    protein
        An ``SCNProtein`` from ``load_sidechainnet()``.
    contact_cutoff
        If set, only add edges with distance <= this value (angstroms).
    allow_incomplete
        If False (default), raise when the mask contains any ``-`` residues.
    """
    if not allow_incomplete and "-" in str(protein.mask):
        raise ValueError("Protein is incomplete: mask contains unknown amino acid positions.")

    positions = read_cb_positions(protein)
    return positions_to_graph(
        positions,
        protein_id=getattr(protein, "id", None),
        contact_cutoff=contact_cutoff,
    )


def read_cb_positions(protein: SCNProtein) -> np.ndarray:
    """
    Compact C_beta/C_alpha coordinates for graph nodes, shape (m, 3).

    Only residues with mask ``+`` and a finite chosen position are included.
    """
    coords = protein.coords
    if hasattr(coords, "detach"):
        coords = coords.detach().cpu().numpy()
    coords = np.asarray(coords, dtype=float)

    mask = str(protein.mask)
    positions: list[np.ndarray] = []
    for i, char in enumerate(mask):
        if char != "+":
            continue
        pos = coords[i, _CB_ATOM_INDEX]
        if np.isnan(pos).any():
            pos = coords[i, _CA_ATOM_INDEX]
        if np.isnan(pos).any():
            continue
        positions.append(pos)

    if not positions:
        raise ValueError("No C_beta/C_alpha coordinates found (mask or coordinates all missing).")
    return np.asarray(positions, dtype=float)


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Euclidean distances between all pairs of 3D points, shape (m, m)."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (m, 3), got {positions.shape}.")
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def positions_to_graph(
    positions: np.ndarray,
    *,
    protein_id: str | None = None,
    contact_cutoff: float | None = None,
) -> nx.Graph:
    """
    Build an undirected graph from residue positions.

    Parameters
    ----------
    positions
        Shape (m, 3) coordinates in angstroms.
    protein_id
        Optional identifier stored in ``graph.graph["protein_id"]``.
    contact_cutoff
        If set, only add edges with distance <= this value.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        raise ValueError("positions must contain at least one residue.")

    distances = pairwise_distances(positions)

    graph = nx.Graph()
    if protein_id is not None:
        graph.graph["protein_id"] = protein_id
    graph.graph["distance_unit"] = "angstrom"

    for i, pos in enumerate(positions):
        graph.add_node(i, position=pos)

    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = distances[i, j]
            if contact_cutoff is None or dist <= contact_cutoff:
                graph.add_edge(i, j, weight=dist)

    return graph
