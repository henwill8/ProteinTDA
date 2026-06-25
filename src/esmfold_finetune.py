"""
Fine-tune facebook/esmfold_v1 adding a topological (Wasserstein)
loss on C_beta/C_alpha distance matrices (TDA).

Model: https://huggingface.co/facebook/esmfold_v1
"""

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from tmtools import tm_align
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.modeling_esmfold import EsmForProteinFoldingOutput

from data_conversions import (
    Atom14,
    atom_positions_from_atom14,
    atom_positions_from_sidechainnet,
    pre_loss_conversion,
    SideChainAtom,
)
from esm_cache import ESMEmbeddingCache
from loss import ESMFoldLoss


def drop_esm_encoder(model: EsmForProteinFolding) -> None:
    for name in ("esm", "esm_s_mlp", "esm_s_combine", "embedding", "af2_to_esm"):
        if hasattr(model, name):
            delattr(model, name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure_finetuning(
    model: EsmForProteinFolding,
    *,
    unfreeze_trunk_blocks: int = 2,
    unfreeze_structure_module: bool = False,
) -> None:
    for param in model.parameters():
        param.requires_grad = False

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


def build_model(
    model_name: str,
    device: torch.device,
    *,
    unfreeze_trunk_blocks: int = 2,
    unfreeze_structure_module: bool = False,
    trunk_chunk_size: int = 4,
    esm_half: bool = True,
    gradient_checkpointing: bool = True,
    load_esm: bool = True,
) -> EsmForProteinFolding:
    model = EsmForProteinFolding.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)
    if load_esm:
        if esm_half:
            model.esm = model.esm.half()
    else:
        drop_esm_encoder(model)
    configure_finetuning(
        model,
        unfreeze_trunk_blocks=unfreeze_trunk_blocks,
        unfreeze_structure_module=unfreeze_structure_module,
    )
    if gradient_checkpointing and load_esm: # if esm is not loaded, gradient checkpointing is not supported
        model.gradient_checkpointing_enable()
    model.trunk.set_chunk_size(trunk_chunk_size)
    return model


def apply_training_mode(
    model: EsmForProteinFolding,
    *,
    unfreeze_trunk_blocks: int = 2,
    unfreeze_structure_module: bool = False,
    dropout_in_frozen: bool = False,
) -> None:
    """
    If dropout_in_frozen is False, frozen trunk blocks run in eval mode so dropout is disabled
    there while trainable blocks keep it enabled.
    """
    model.train()
    if dropout_in_frozen:
        return

    if getattr(model, "esm", None) is not None:
        model.esm.eval()

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


def prepare_esm_cache(
    cache_dir: Path,
    proteins: list[SCNProtein],
    model_name: str,
    tokenizer: AutoTokenizer,
    device: torch.device,
    *,
    trunk_chunk_size: int,
) -> ESMEmbeddingCache:
    cache = ESMEmbeddingCache(cache_dir)
    missing = cache.missing(proteins)
    if missing:
        print(f"Caching {len(missing)} ESM embeddings in {cache_dir}...")
        cache_model = build_model(
            model_name,
            device,
            unfreeze_trunk_blocks=0,
            unfreeze_structure_module=False,
            trunk_chunk_size=trunk_chunk_size,
            gradient_checkpointing=False,
        )
        cache_model.eval()
        cache.store(missing, cache_model, tokenizer, device)
        del cache_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"All {len(proteins)} proteins already cached at {cache_dir}.")

    if cache.missing(proteins):
        raise RuntimeError(f"Failed to cache all proteins at {cache_dir}.")
    print(f"Using ESM cache from {cache_dir} ({cache.cached_count(proteins)} entries on disk.")
    return cache


def run_esmfold(
    model: EsmForProteinFolding,
    protein,
    device: torch.device,
    tokenizer,
    *,
    num_recycles: int | None = None,
    esm_cache: ESMEmbeddingCache | None = None,
    use_amp: bool = False,
) -> EsmForProteinFoldingOutput:
    amp_enabled = use_amp and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        if esm_cache is not None:
            outputs = esm_cache.run_trunk_from_cache(
                model,
                protein,
                device,
                num_recycles=num_recycles,
            )
        else:
            sequence = str(protein.seq)
            inputs = tokenizer(
                sequence,
                return_tensors="pt",
                add_special_tokens=False,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs, num_recycles=num_recycles)

    if outputs.positions is None:
        raise RuntimeError("ESMFold did not return positions.")
    return outputs


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
    esm_cache: ESMEmbeddingCache | None = None,
) -> dict[str, torch.Tensor]:
    outputs = run_esmfold(
        model,
        protein,
        device,
        tokenizer,
        num_recycles=num_recycles,
        esm_cache=esm_cache,
        use_amp=use_amp,
    )
    out, batch = pre_loss_conversion(outputs, protein, device=device, loss_config=loss_fn.config)
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
    unfreeze_trunk_blocks: int = 2,
    unfreeze_structure_module: bool = False,
    dropout_in_frozen: bool = False,
    train_recycles: int = 1,
    randomize_recycles: bool = True,
    use_amp: bool = False,
    grad_clip_norm: float | None = 1.0,
    esm_cache: ESMEmbeddingCache | None = None,
) -> dict[str, float]:
    apply_training_mode(
        model,
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
                    esm_cache=esm_cache,
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
    esm_cache: ESMEmbeddingCache | None = None,
):
    model.eval()
    with torch.no_grad():
        plddt_list = []
        tm_score_list = []
        for batch in tqdm(loader, desc="test", leave=False):
            for protein in batch:
                output = run_esmfold(
                    model,
                    protein,
                    device,
                    tokenizer,
                    num_recycles=infer_recycles,
                    esm_cache=esm_cache,
                )

                plddt = output.plddt.tolist()
                protein_mean_plddt = np.mean(plddt)
                plddt_list.append(protein_mean_plddt)

                # We are using c-alpha atoms to extract
                atom_exists = output.atom14_atom_exists[0] if output.atom14_atom_exists is not None else None
                pred_c_alpha = atom_positions_from_atom14(output["positions"][-1][0], Atom14.CA, atom_exists)

                exp_c_alpha = atom_positions_from_sidechainnet(protein, SideChainAtom.CA).cpu().numpy()

                res = tm_align(pred_c_alpha.cpu().numpy(), exp_c_alpha, str(protein.seq), str(protein.seq))
                tm_score = res.tm_norm_chain2

                tm_score_list.append(tm_score)

        mean_plddt = np.mean(plddt_list)
        mean_tm = np.mean(tm_score_list)
        return mean_plddt, mean_tm
