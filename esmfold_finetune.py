"""
Fine-tune facebook/esmfold_v1 adding a topological (Wasserstein)
loss on C_beta/C_alpha distance matrices (TDA).

Model: https://huggingface.co/facebook/esmfold_v1
"""

import random
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


def _esm_encoder_is_trainable(model: EsmForProteinFolding) -> bool:
    return any(p.requires_grad for p in model.esm.encoder.layer.parameters())


def _compute_esm_representations(model: EsmForProteinFolding, esmaa: torch.Tensor) -> torch.Tensor:
    if _esm_encoder_is_trainable(model):
        return model.compute_language_model_representations(esmaa)
    with torch.no_grad():
        return model.compute_language_model_representations(esmaa)


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

    esm_s = _compute_esm_representations(self, esmaa)
    esm_s = esm_s.to(self.esm_s_combine.dtype)

    if cfg.esm_ablate_sequence:
        esm_s = esm_s * 0

    # Keep ESM in the graph only if encoder layers are being trained, otherwise
    # detach to avoid backprop through the full model.
    if not self.training or not _esm_encoder_is_trainable(self):
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

    if not self.training:
        structure["lm_logits"] = self.lm_head(structure["s_s"])

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
    if self.training:
        # ptm is unused by the training loss and compute_tm can fail on short chains
        structure["ptm"] = None
    else:
        try:
            structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=self.distogram_bins)
        except IndexError:
            structure["ptm"] = None
        structure.update(
            compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=self.distogram_bins)
        )

    return EsmForProteinFoldingOutput(**structure)


def patch_forward_for_training(model: EsmForProteinFolding) -> None:
    """Replace ESMFold forward so losses can backprop into trainable ESM layers."""
    model.forward = types.MethodType(_forward_preserve_esm_grad, model)


def configure_finetuning(
    model: EsmForProteinFolding,
    *,
    unfreeze_esm_layers: int = 0,
    unfreeze_trunk_blocks: int = 2,
    unfreeze_structure_module: bool = False,
) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_esm_layers > 0:
        for layer in model.esm.encoder.layer[-unfreeze_esm_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    if unfreeze_trunk_blocks > 0:
        for block in model.trunk.blocks[-unfreeze_trunk_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    if unfreeze_structure_module:
        for param in model.trunk.structure_module.parameters():
            param.requires_grad = True
        for param in model.trunk.trunk2sm_s.parameters():
            param.requires_grad = True
        for param in model.trunk.trunk2sm_z.parameters():
            param.requires_grad = True


def install_trunk_grad_stop(model: EsmForProteinFolding, n_trainable_trunk_blocks: int) -> None:
    """
    Detach activations after the last frozen trunk block so backward skips earlier blocks.
    """
    existing = getattr(model, "_trunk_grad_stop_hook", None)
    if existing is not None:
        existing.remove()
        model._trunk_grad_stop_hook = None

    n_frozen = len(model.trunk.blocks) - n_trainable_trunk_blocks
    if n_frozen <= 0:
        return

    def _detach_block_output(_module, _inputs, output):
        if isinstance(output, tuple):
            return tuple(x.detach() if torch.is_tensor(x) else x for x in output)
        if torch.is_tensor(output):
            return output.detach()
        return output

    model._trunk_grad_stop_hook = model.trunk.blocks[n_frozen - 1].register_forward_hook(
        _detach_block_output
    )


def build_model(
    model_name: str,
    device: torch.device,
    *,
    unfreeze_esm_layers: int = 0,
    unfreeze_trunk_blocks: int = 2,
    unfreeze_structure_module: bool = False,
    trunk_chunk_size: int = 4,
    esm_half: bool = True,
) -> EsmForProteinFolding:
    model = EsmForProteinFolding.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)
    if esm_half:
        # potentially remove this or have trainable layers be fp32 if training is not stable
        model.esm = model.esm.half()
    # patch_forward_for_training(model)
    configure_finetuning(
        model,
        unfreeze_esm_layers=unfreeze_esm_layers,
        unfreeze_trunk_blocks=unfreeze_trunk_blocks,
        unfreeze_structure_module=unfreeze_structure_module,
    )
    install_trunk_grad_stop(model, n_trainable_trunk_blocks=unfreeze_trunk_blocks)
    model.trunk.set_chunk_size(trunk_chunk_size)
    return model


def apply_training_mode(
    model: EsmForProteinFolding,
    *,
    unfreeze_esm_layers: int = 0,
    unfreeze_trunk_blocks: int = 2,
    unfreeze_structure_module: bool = False,
    dropout_in_frozen: bool = False,
) -> None:
    """
    If dropout_in_frozen is False, frozen ESM encoder layers and the folding trunk run in eval mode so
    dropout is disabled there while trainable ESM layers keep it enabled.
    """
    model.train()
    if dropout_in_frozen:
        return

    encoder_layers = model.esm.encoder.layer
    n_frozen_esm = len(encoder_layers) - max(0, unfreeze_esm_layers)
    for layer in encoder_layers[:n_frozen_esm]:
        layer.eval()
    for layer in encoder_layers[n_frozen_esm:]:
        layer.train()

    n_frozen_trunk = len(model.trunk.blocks) - max(0, unfreeze_trunk_blocks)
    for block in model.trunk.blocks[:n_frozen_trunk]:
        block.eval()
    for block in model.trunk.blocks[n_frozen_trunk:]:
        block.train()

    if unfreeze_structure_module:
        model.trunk.structure_module.train()
        model.trunk.trunk2sm_s.train()
        model.trunk.trunk2sm_z.train()
    else:
        model.trunk.structure_module.eval()
        model.trunk.trunk2sm_s.eval()
        model.trunk.trunk2sm_z.eval()


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _sample_num_recycles(max_recycles: int) -> int:
    if max_recycles <= 0:
        return 0
    return random.randint(0, max_recycles)


def compute_losses(
    model: EsmForProteinFolding,
    tokenizer,
    protein,
    device: torch.device,
    loss_fn: ESMFoldLoss,
    *,
    num_recycles: int | None = None,
    use_amp: bool = False,
) -> dict[str, torch.Tensor]:
    sequence = str(protein.seq)
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    amp_enabled = use_amp and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        outputs = model(**inputs, num_recycles=num_recycles)
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
    *,
    unfreeze_esm_layers: int = 0,
    unfreeze_trunk_blocks: int = 2,
    unfreeze_structure_module: bool = False,
    dropout_in_frozen: bool = False,
    train_recycles: int = 1,
    randomize_recycles: bool = True,
    use_amp: bool = False,
    grad_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    apply_training_mode(
        model,
        unfreeze_esm_layers=unfreeze_esm_layers,
        unfreeze_trunk_blocks=unfreeze_trunk_blocks,
        unfreeze_structure_module=unfreeze_structure_module,
        dropout_in_frozen=dropout_in_frozen,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    totals = defaultdict(lambda: 0.0)
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        for protein in batch:
            optimizer.zero_grad(set_to_none=True)
            if randomize_recycles and train_recycles > 1:
                num_recycles = _sample_num_recycles(train_recycles)
            else:
                num_recycles = train_recycles

            try:
                losses = compute_losses(
                    model,
                    tokenizer,
                    protein,
                    device,
                    loss_fn,
                    num_recycles=num_recycles,
                    use_amp=use_amp,
                )
            except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    print(f"OOM on seq len {len(protein.seq)}, skipping")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    print(e)
                continue

            if scaler.is_enabled():
                scaler.scale(losses["total"]).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        grad_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        grad_clip_norm,
                    )
                optimizer.step()

            for key, value in losses.items():
                totals[key] += float(value.detach())
            n += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()

    if n == 0:
        return totals
    return {key: value / n for key, value in totals.items()}


def test_model(
    model,
    tokenizer,
    loader,
    device,
    *,
    infer_recycles: int | None = None,
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

                output = model(**inputs, num_recycles=infer_recycles)
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
