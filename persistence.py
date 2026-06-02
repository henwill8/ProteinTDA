"""Differentiable persistence diagrams and Wasserstein distance (Gudhi)."""

import gudhi as gd
import gudhi.wasserstein
import torch


def pd_from_graph(adj_tensor: torch.Tensor, max_dimension: int, hom_dim: int = 2) -> list[torch.Tensor]:
    """
    Persistence diagrams whose birth/death values remain attached to ``adj_tensor`` for autograd.

    ``adj_tensor`` is a symmetric distance matrix (diagonal zero).
    """
    diagrams: list[torch.Tensor] = []
    adj_matrix = adj_tensor.detach().cpu().numpy()

    rips_complex = gd.RipsComplex(distance_matrix=adj_matrix)
    st = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    st.compute_persistence()
    generators = st.flag_persistence_generators()

    for i in range(hom_dim):
        if i == 0:
            generators_i = generators[i]
        else:
            generators_i = generators[1][i - 1]
        if len(generators_i) == 0:
            diagrams.append(torch.empty((0, 2), device=adj_tensor.device, dtype=adj_tensor.dtype))
            continue

        hi_gens = torch.tensor(generators_i, device=adj_tensor.device)
        if i == 0:
            birth_values = adj_tensor[hi_gens[:, 0], hi_gens[:, 0]]
            death_values = adj_tensor[hi_gens[:, 1], hi_gens[:, 2]]
        else:
            birth_values = adj_tensor[hi_gens[:, 0], hi_gens[:, 1]]
            death_values = adj_tensor[hi_gens[:, 2], hi_gens[:, 3]]
        diagrams.append(torch.stack([birth_values, death_values], dim=-1))

    return diagrams


def wasserstein_distance(
    pred_diags: list[torch.Tensor],
    exp_diags: list[torch.Tensor],
    hom_dim: int = 2,
) -> list[torch.Tensor]:
    """Per-dimension Wasserstein distances with autodiff through ``pred_diags``."""
    losses: list[torch.Tensor] = []
    for i in range(hom_dim):
        loss_wd = gudhi.wasserstein.wasserstein_distance(
            pred_diags[i],
            exp_diags[i],
            matching=False,
            enable_autodiff=True,
            keep_essential_parts=False,
        )
        losses.append(loss_wd)
    return losses


def wasserstein_loss(
    pred_adj: torch.Tensor,
    target_adj: torch.Tensor,
    *,
    max_dimension: int = 2,
    hom_dim: int = 2,
) -> dict[str, torch.Tensor]:
    """
    Wasserstein distances between predicted and target persistence diagrams.

    Returns separate losses for H0 (``h0``) and H1 (``h1``) when ``hom_dim >= 2``.
    """
    target_diags = pd_from_graph(target_adj, max_dimension, hom_dim)
    pred_diags = pd_from_graph(pred_adj, max_dimension, hom_dim)
    terms = wasserstein_distance(pred_diags, target_diags, hom_dim)
    zero = torch.zeros((), device=pred_adj.device, dtype=pred_adj.dtype)
    losses: dict[str, torch.Tensor] = {}
    if hom_dim >= 1:
        losses["h0"] = terms[0] if len(terms) > 0 else zero
    if hom_dim >= 2:
        losses["h1"] = terms[1] if len(terms) > 1 else zero
    return losses


def distance_matrix(positions: torch.Tensor) -> torch.Tensor:
    """Full pairwise distance matrix, shape (n, n)."""
    return torch.cdist(positions, positions)
