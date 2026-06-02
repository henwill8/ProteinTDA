"""
Convert ProteinNet text records to distance-weighted residue graphs.

ProteinNet tertiary structure stores backbone atoms (N, C_alpha, C_prime) in picometers
Record format: https://github.com/aqlaboratory/proteinnet/blob/master/docs/proteinnet_records.md
"""

from io import StringIO
from pathlib import Path
from typing import BinaryIO, TextIO, Union

import networkx as nx
import numpy as np

Source = Union[str, Path, TextIO, BinaryIO]

_CA_ATOM_INDEX = 1  # N=0, C_alpha=1, C_prime=2 within each residue triple

def proteinnet_record_to_graph(
    source: Source,
    *,
    contact_cutoff: float | None = None,
) -> nx.Graph:
    """
    Build an undirected graph: one node per residue C_alpha, edges weighted by distance.

    Parameters
    ----------
    source
        Path to a record file, open text stream, or string containing a record.
    contact_cutoff
        If set, only add edges with distance <= this value.
    """
    # If the protein is missing positions in the residues, skip it and throw a ValueError
    check_protein_completeness_from_mask(source, raise_on_incomplete=True)

    positions = read_ca_positions(source)
    distances = pairwise_distances(positions)

    graph = nx.Graph()
    for i, pos in enumerate(positions):
        graph.add_node(i, position=pos)

    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = distances[i, j]
            if contact_cutoff is None or dist <= contact_cutoff:
                graph.add_edge(i, j, weight=dist)

    return graph


def read_ca_positions(source: Source) -> np.ndarray:
    """
    Read C_alpha coordinates from the [TERTIARY] block of a ProteinNet record.

    Returns
    -------
    ndarray, shape (m, 3)
        One row per residue with non-zero C_alpha (missing residues are dropped).
    """
    x, y, z = _read_tertiary_axes(source)
    n_res = len(x) // 3
    if len(x) != 3 * n_res:
        raise ValueError(
            f"TERTIARY line length {len(x)} is not divisible by 3 (N, C_alpha, C_prime per residue)."
        )

    ca = np.column_stack(
        (
            x[_CA_ATOM_INDEX::3],
            y[_CA_ATOM_INDEX::3],
            z[_CA_ATOM_INDEX::3],
        )
    )
    present = np.linalg.norm(ca, axis=1) > 0
    ca = ca[present]

    if ca.size == 0:
        raise ValueError("No C_alpha coordinates found (all missing or zero).")

    return ca


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """
    Euclidean distances between all pairs of 3D points.

    Returns
    -------
    ndarray, shape (m, m)
        Symmetric matrix; diagonal is zero.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (m, 3), got {positions.shape}.")

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def check_protein_completeness_from_mask(
    source: Source,
    *,
    raise_on_incomplete: bool = True,
) -> bool:
    """
    Check the [MASK] section and validate that all residues are known.

    Returns
    -------
    bool
        True if the protein is complete (mask has no '-').
        False if incomplete and ``raise_on_incomplete`` is False.

    Raises
    ------
    ValueError
        If the protein is incomplete and ``raise_on_incomplete`` is True.
    """
    mask = _read_mask(source)
    incomplete = "-" in mask
    if incomplete:
        if raise_on_incomplete:
            raise ValueError("Protein is incomplete: mask contains unknown amino acid positions.")
        return False
    return True


def _read_tertiary_axes(source: Source) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    handle, close = _open_text(source)
    try:
        for line in handle:
            if _decode_line(line) != "[TERTIARY]":
                continue
            x = np.asarray([float(v) for v in _decode_line(handle.readline()).split()])
            y = np.asarray([float(v) for v in _decode_line(handle.readline()).split()])
            z = np.asarray([float(v) for v in _decode_line(handle.readline()).split()])
            if len(x) != len(y) or len(x) != len(z):
                raise ValueError("TERTIARY x/y/z lines have different lengths.")
            return x, y, z
    finally:
        if close:
            handle.close()

    raise ValueError("No [TERTIARY] section found.")


def _read_mask(source: Source) -> str:
    handle, close = _open_text(source)
    try:
        for line in handle:
            if _decode_line(line) != "[MASK]":
                continue
            return _decode_line(handle.readline()).strip()
    finally:
        if close:
            handle.close()

    raise ValueError("No [MASK] section found.")


def _open_text(source: Source) -> tuple[TextIO, bool]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists() and path.is_file():
            return path.open(encoding="utf-8"), True
        return StringIO(str(source)), False
    if hasattr(source, "read"):
        return source, False  # type: ignore[return-value]
    raise TypeError(f"Unsupported source type: {type(source)!r}")


def _decode_line(line: str | bytes) -> str:
    if isinstance(line, bytes):
        return line.decode("utf-8").rstrip("\n\r")
    return line.rstrip("\n\r")
