"""
Fine-tune facebook/esmfold_v1 with frozen weights except the last ESM encoder layers,
adding a topological (Wasserstein) loss on C_beta/C_alpha distance matrices (TDA).

Model: https://huggingface.co/facebook/esmfold_v1
"""

import types
from collections import defaultdict

import numpy as np
import torch
from tmtools import tm_align
from tqdm import tqdm
from transformers import EsmForProteinFolding
from transformers.models.esm.modeling_esmfold import (
    EsmForProteinFoldingOutput,
    categorical_lddt,
)
from transformers.models.esm.openfold_utils import (
    compute_predicted_aligned_error,
    compute_tm,
    make_atom14_masks,
)

from data_conversions import (
    Atom14,
    atom_positions_from_atom14,
    atom_positions_from_sidechainnet,
    pre_loss_conversion,
    SideChainAtom,
)
from loss import ESMFoldLoss


def _forward_preserve_esm_grad(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    masking_pattern: torch.Tensor | None = None,
    num_recycles: int | None = None,
    output_hidden_states: bool | None = False,
    **kwargs,
) -> EsmForProteinFoldingOutput:
    """
    ESMFold forward that keeps ESM representations in the autograd graph while training.

    https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/modeling_esmfold.py#L2044
    """
    cfg = self.config.esmfold_config

    aa = input_ids
    batch_size = aa.shape[0]
    seq_len = aa.shape[1]
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(aa, device=device)
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=device).expand_as(input_ids)

    esmaa = self.af2_idx_to_esm_idx(aa, attention_mask)

    if masking_pattern is not None:
        masked_aa, esmaa, mlm_targets = self.bert_mask(aa, esmaa, attention_mask, masking_pattern)
    else:
        masked_aa = aa
        mlm_targets = None

    esm_s = self.compute_language_model_representations(esmaa)
    esm_s = esm_s.to(self.esm_s_combine.dtype)

    if cfg.esm_ablate_sequence:
        esm_s = esm_s * 0

    if not self.training:
        esm_s = esm_s.detach()

    esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
    s_s_0 = self.esm_s_mlp(esm_s)

    s_z_0 = s_s_0.new_zeros(batch_size, seq_len, seq_len, cfg.trunk.pairwise_state_dim)

    if self.config.esmfold_config.embed_aa:
        s_s_0 += self.embedding(masked_aa)

    structure: dict = self.trunk(s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles)
    structure = {
        key: value
        for key, value in structure.items()
        if key
        in [
            "s_z",
            "s_s",
            "frames",
            "sidechain_frames",
            "unnormalized_angles",
            "angles",
            "positions",
            "states",
        ]
    }

    if mlm_targets:
        structure["mlm_targets"] = mlm_targets

    disto_logits = self.distogram_head(structure["s_z"])
    disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
    structure["distogram_logits"] = disto_logits

    lm_logits = self.lm_head(structure["s_s"])
    structure["lm_logits"] = lm_logits

    structure["aatype"] = aa
    make_atom14_masks(structure)
    for key in [
        "atom14_atom_exists",
        "atom37_atom_exists",
    ]:
        structure[key] *= attention_mask.unsqueeze(-1)
    structure["residue_index"] = position_ids

    lddt_head = self.lddt_head(structure["states"]).reshape(
        structure["states"].shape[0], batch_size, seq_len, -1, self.lddt_bins
    )
    structure["lddt_head"] = lddt_head
    structure["plddt"] = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)

    ptm_logits = self.ptm_head(structure["s_z"])
    structure["ptm_logits"] = ptm_logits
    structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=self.distogram_bins)
    structure.update(compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=self.distogram_bins))

    return EsmForProteinFoldingOutput(**structure)


def patch_forward_for_training(model: EsmForProteinFolding) -> None:
    """Replace ESMFold forward so losses can backprop into trainable ESM layers."""
    model.forward = types.MethodType(_forward_preserve_esm_grad, model)


def freeze_except_last_esm_layers(model: EsmForProteinFolding, n_layers: int = 2) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if n_layers <= 0:
        return

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
