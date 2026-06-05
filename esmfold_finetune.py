"""
Fine-tune facebook/esmfold_v1 with frozen weights except the last ESM encoder layers,
adding a topological (Wasserstein) loss on C_beta/C_alpha distance matrices (TDA).

Model: https://huggingface.co/facebook/esmfold_v1
"""

import numpy as np
import torch
import torch.nn.functional as F
from tmtools import tm_align
from tqdm import tqdm
from transformers import EsmForProteinFolding
from enum import Enum

from persistence import wasserstein_loss
from sidechainnet_graph import read_atom_positions, distance_matrix, SideChainAtom

# OpenFold atom37 indices
class Atom14(Enum):
    CA = 1
    CB = 4


def freeze_except_last_esm_layers(model: EsmForProteinFolding, n_layers: int = 2) -> None:
    """Freeze all parameters, then unfreeze the last ``n_layers`` ESM encoder blocks."""
    for param in model.parameters():
        param.requires_grad = False

    encoder_layers = model.esm.encoder.layer
    for layer in encoder_layers[-n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def atom_positions_from_atom14(
    positions: torch.Tensor,
    atom: Atom14,
    atom_exists: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compact C_beta/C_alpha coordinates from ESMFold atom14 output, shape (m, 3).

    ``positions`` is (L, 37, 3); ``mask`` is the SidechainNet mask string.
    """
    coords: list[torch.Tensor] = []
    length = positions.shape[0]
    for i in range(length):
        atom_pos = positions[i, atom.value]
        if atom_exists is not None and atom_exists[i, atom.value] < 0.5:
            atom_pos = positions[i, Atom14.CA.value]
        coords.append(atom_pos)
    if not coords:
        raise ValueError("No valid C_beta/C_alpha coordinates in model output.")
    return torch.stack(coords)


def target_atom_positions(protein, atom: SideChainAtom, device: torch.device) -> torch.Tensor:
    """Ground-truth compact C_beta/C_alpha positions from SidechainNet."""
    positions = read_atom_positions(protein, atom)
    return torch.tensor(positions, dtype=torch.float32, device=device)


def esmfold_loss(pred_cb: torch.Tensor, target_cb: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=pred_cb.device)


def compute_losses(
    model: EsmForProteinFolding,
    tokenizer,
    protein,
    device: torch.device,
    *,
    wasserstein_h0_weight: float,
    wasserstein_h1_weight: float,
    max_rips_dimension: int,
    hom_dim: int,
) -> dict[str, torch.Tensor]:
    sequence = str(protein.seq)
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model(**inputs)
    if outputs.positions is None:
        raise RuntimeError("ESMFold did not return positions.")

    
    atom_exists = outputs.atom14_atom_exists[0] if outputs.atom14_atom_exists is not None else None
    pred_cb = atom_positions_from_atom14(outputs["positions"][-1][0], Atom14.CB, atom_exists)
    target_cb = target_atom_positions(protein, SideChainAtom.CB, device)

    esmfold_loss_value = esmfold_loss(pred_cb, target_cb)

    pred_adj = distance_matrix(pred_cb)
    target_adj = distance_matrix(target_cb)
    target_adj.requires_grad = False

    topo = wasserstein_loss(
        pred_adj,
        target_adj,
        max_dimension=max_rips_dimension,
        hom_dim=hom_dim,
    )

    wass_h0 = topo["h0"]
    wass_h1 = topo["h1"]
    total = esmfold_loss_value + wasserstein_h0_weight * wass_h0 + wasserstein_h1_weight * wass_h1
    return {
        "total": total,
        "esmfold_loss": esmfold_loss_value,
        "wasserstein_h0": wass_h0,
        "wasserstein_h1": wass_h1,
    }


def train_one_epoch(
    model,
    tokenizer,
    loader,
    optimizer,
    device,
    *,
    wasserstein_h0_weight: float,
    wasserstein_h1_weight: float,
    max_rips_dimension: int,
    hom_dim: int,
) -> dict[str, float]:
    model.train()
    totals = {"total": 0.0, "esmfold_loss": 0.0, "wasserstein_h0": 0.0, "wasserstein_h1": 0.0}
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        for protein in batch:
            optimizer.zero_grad(set_to_none=True)
            try:
                losses = compute_losses(
                    model,
                    tokenizer,
                    protein,
                    device,
                    wasserstein_h0_weight=wasserstein_h0_weight,
                    wasserstein_h1_weight=wasserstein_h1_weight,
                    max_rips_dimension=max_rips_dimension,
                    hom_dim=hom_dim,
                )
            except (ValueError, RuntimeError):
                continue

            losses["total"].backward()
            optimizer.step()

            for key in totals:
                totals[key] += float(losses[key].detach())
            n += 1

    if n == 0:
        return totals
    return {key: value / n for key, value in totals.items()}

def test_model(
    model,
    tokenizer,
    loader,
    device,
):
    model.eval()
    with torch.no_grad():
        plddt_list = []
        tm_score_list = []
        for batch in tqdm(loader, desc="test", leave=False):
            for protein in batch:
                sequence = str(protein.seq)
                inputs = tokenizer(
                    sequence,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}

                output = model(**inputs)
                if output.positions is None:
                    raise RuntimeError("ESMFold did not return positions.")
                
                plddt = output.plddt.tolist()
                protein_mean_plddt = np.mean(plddt)
                plddt_list.append(protein_mean_plddt)
               
                # We are using c-alpha atoms to extract
                atom_exists = output.atom14_atom_exists[0] if output.atom14_atom_exists is not None else None
                pred_c_alpha = atom_positions_from_atom14(output["positions"][-1][0], Atom14.CA, atom_exists)

                exp_c_alpha = read_atom_positions(protein, SideChainAtom.CA)

                res = tm_align(pred_c_alpha.cpu().numpy(), exp_c_alpha, sequence, sequence)
                tm_score = res.tm_norm_chain2

                tm_score_list.append(tm_score)

        mean_plddt = np.mean(plddt_list) 
        mean_tm = np.mean(tm_score_list)
        return mean_plddt, mean_tm
