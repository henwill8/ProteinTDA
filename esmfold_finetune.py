"""
Fine-tune facebook/esmfold_v1 with frozen weights except the last ESM encoder layers,
adding a topological (Wasserstein) loss on C_beta/C_alpha distance matrices (TDA).

Model: https://huggingface.co/facebook/esmfold_v1
"""

import numpy as np
import torch
from tmtools import tm_align
from tqdm import tqdm
from transformers import EsmForProteinFolding
from collections import defaultdict

from data_conversions import (
    Atom14,
    atom_positions_from_atom14,
    atom_positions_from_sidechainnet,
    pre_loss_conversion,
    SideChainAtom,
)
from loss import ESMFoldLoss

def freeze_except_last_esm_layers(model: EsmForProteinFolding, n_layers: int = 2) -> None:
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


def compute_losses(
    model: EsmForProteinFolding,
    tokenizer,
    protein,
    device: torch.device,
    loss_fn: ESMFoldLoss,
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

    out, batch = pre_loss_conversion(outputs, protein, device=device)
    total, breakdown = loss_fn(out, batch, _return_breakdown=True)

    return {
        "total": total,
        **{key: value for key, value in breakdown.items()},
    }


def train_one_epoch(
    model,
    tokenizer,
    loader,
    optimizer,
    device,
    loss_fn: ESMFoldLoss,
) -> dict[str, float]:
    model.train()
    totals = defaultdict(lambda: 0.0)
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
                    loss_fn,
                )
            except (ValueError, RuntimeError) as e:
                print(e)
                continue

            losses["total"].backward()
            optimizer.step()

            for key, value in losses.items():
                totals[key] += float(value.detach())
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

                exp_c_alpha = atom_positions_from_sidechainnet(protein, SideChainAtom.CA).cpu().numpy()

                res = tm_align(pred_c_alpha.cpu().numpy(), exp_c_alpha, sequence, sequence)
                tm_score = res.tm_norm_chain2

                tm_score_list.append(tm_score)

        mean_plddt = np.mean(plddt_list)
        mean_tm = np.mean(tm_score_list)
        return mean_plddt, mean_tm
