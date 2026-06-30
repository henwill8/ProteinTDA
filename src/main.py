import sys
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import sidechainnet as scn
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from esmfold_finetune import (
    build_model,
    prepare_esm_cache,
    train_one_epoch,
    test_model,
    trainable_parameter_count,
)
from esm_cache import ESMEmbeddingCache
from kfold_runner import KFoldRunner
from loss import ESMFoldLoss
from config import HEAT_RFF_CONFIG, LOSS_CONFIG, RUN_CONFIG
from vpd_macros import create_vpd_kernels


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def resolve_device() -> torch.device:
    device = RUN_CONFIG.runtime.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)

# For lower memory usage, try batch size 1, chunk size 1 or 2? (would be really slow), recycles 1

def load_dataset() -> list:
    data = RUN_CONFIG.data
    print("Loading SidechainNet...")
    dataset = scn.load(
        casp_version=data.casp_version,
        casp_thinning=data.casp_thinning,
        scn_dataset=True,
        scn_dir=data.scn_dir,
        force_download=False,
        complete_structures_only=not data.allow_incomplete,
    )

    # Sidechainnet includes proteins with '.' in the mask, even when complete_structures_only=True
    if not data.allow_incomplete:
        before = len(dataset)
        bad = [
            protein
            for protein in dataset
            if len(protein.seq) != len(protein.mask)
            or str(protein.mask) != "+" * len(protein.mask)
        ]
        for protein in bad[:]:
            mask = str(protein.mask)
            non_plus = {char for char in mask if char != "+"}
            print(
                f"  excluding {protein.id}: len(seq)={len(protein.seq)}, len(mask)={len(mask)}, non_plus={non_plus or None}"
            )
        dataset = [protein for protein in dataset if protein not in bad]
        removed = before - len(dataset)
        if removed:
            print(f"Removed {removed} proteins with incomplete or misaligned masks.")

    if data.max_protein_length is not None:
        before = len(dataset)
        dataset = [protein for protein in dataset if len(protein.seq) <= data.max_protein_length]
        removed = before - len(dataset)
        if removed:
            print(f"Removed {removed} proteins longer than {data.max_protein_length} residues.")

    if data.max_proteins is not None and len(dataset) > data.max_proteins:
        dataset = dataset[-data.max_proteins:]
    # dataset = dataset[5:]
    print(f"Loaded {len(dataset)} proteins.")
    return dataset


def sample_proteins(
    proteins: list,
    max_proteins: int | None,
    rng: np.random.Generator,
) -> list:
    if max_proteins is None or max_proteins >= len(proteins):
        return proteins
    indices = rng.choice(len(proteins), size=max_proteins, replace=False)
    return [proteins[i] for i in indices]


def make_loader(
    proteins: list,
    batch_size: int,
    *,
    shuffle: bool = False,
    max_proteins: int | None = None,
    rng: np.random.Generator | None = None,
) -> DataLoader:
    if max_proteins is not None:
        if rng is None:
            raise ValueError("rng is required when max_proteins is set")
        proteins = sample_proteins(proteins, max_proteins, rng)
    return DataLoader(
        dataset=proteins,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: x,
    )


def write_log_file(
    log_file: Path,
    fold_plddt_scores: list[float],
    fold_tm_scores: list[float],
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        for fold_idx, (plddt, tm) in enumerate(zip(fold_plddt_scores, fold_tm_scores), start=1):
            handle.write(f"fold {fold_idx}: mean_plddt={plddt:.4f} mean_tm={tm:.4f}\n")
        handle.write(
            f"mean_plddt mean={np.mean(fold_plddt_scores):.4f} var={np.var(fold_plddt_scores):.4f}\n"
        )
        handle.write(
            f"mean_tm mean={np.mean(fold_tm_scores):.4f} var={np.var(fold_tm_scores):.4f}\n"
        )


def run_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    dataset: list,
    device: torch.device,
    tokenizer: AutoTokenizer,
    loss_fn: ESMFoldLoss,
    esm_cache: ESMEmbeddingCache | None,
    n_splits: int,
) -> tuple[float, float]:
    runtime = RUN_CONFIG.runtime
    training = RUN_CONFIG.training
    set_seed(training.seed + fold)

    print(f"Fold {fold + 1}/{n_splits}: loading fresh model on {device}...")
    # build new model each fold to get pre-finetuned weights
    model = build_model(
        RUN_CONFIG.model.name,
        device,
        unfreeze_trunk_blocks=training.unfreeze_trunk_blocks,
        unfreeze_structure_module=training.unfreeze_structure_module,
        trunk_chunk_size=runtime.trunk_chunk_size,
        gradient_checkpointing=training.gradient_checkpointing,
        load_esm=not runtime.use_esm_cache,
        cache_trunk_blocks=runtime.esm_cache_trunk_blocks if runtime.use_esm_cache else 0,
    )
    trainable, total = trainable_parameter_count(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    train_idx, val_idx = train_test_split(train_idx, test_size=0.25)
    train_proteins = [dataset[i] for i in train_idx]
    val_proteins = [dataset[i] for i in val_idx]
    test_proteins = [dataset[i] for i in test_idx]

    fold_rng = np.random.default_rng(training.seed + fold)
    val_loader = make_loader(
        val_proteins,
        training.batch_size,
        shuffle=False,
        max_proteins=training.val_proteins_per_epoch,
        rng=fold_rng,
    )
    test_loader = make_loader(test_proteins, training.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=training.lr)
    best_model_weights = None
    max_val_tm = 0.0
    patience = 0

    for epoch in range(training.epochs):
        train_loader = make_loader(
            train_proteins,
            training.batch_size,
            shuffle=True,
            max_proteins=training.train_proteins_per_epoch,
            rng=fold_rng,
        )
        metrics = train_one_epoch(
            model,
            tokenizer,
            train_loader,
            optimizer,
            device,
            loss_fn=loss_fn,
            unfreeze_trunk_blocks=training.unfreeze_trunk_blocks,
            unfreeze_structure_module=training.unfreeze_structure_module,
            train_recycles=training.train_recycles,
            use_amp=training.amp,
            esm_cache=esm_cache,
        )
        val_plddt_score, val_tm_score = test_model(
            model,
            tokenizer,
            val_loader,
            device,
            infer_recycles=runtime.infer_recycles,
            esm_cache=esm_cache,
        )
        if val_tm_score > max_val_tm:
            max_val_tm = val_tm_score
            patience = 0
            best_model_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        metrics["val_tm"] = val_tm_score
        metrics["val_plddt"] = val_plddt_score

        print(
            f"epoch {epoch + 1}/{training.epochs}"
            + "\n" + "  ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            + f"\nfold {fold + 1}/{n_splits}"
        )

        if patience > training.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})

    plddt_score, tm_score = test_model(
        model,
        tokenizer,
        test_loader,
        device,
        infer_recycles=runtime.infer_recycles,
        esm_cache=esm_cache,
    )
    print(f"fold {fold + 1}/{n_splits}  mean_plddt={plddt_score:.4f}  mean_tm={tm_score:.4f}")
    return plddt_score, tm_score


def run_baseline_fold(
    fold: int,
    _train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    model,
    proteins: list,
    device: torch.device,
    tokenizer: AutoTokenizer,
    esm_cache: ESMEmbeddingCache | None,
) -> tuple[float, float]:
    test_loader = make_loader(
        [proteins[i] for i in test_idx],
        RUN_CONFIG.training.batch_size,
        shuffle=False,
    )
    plddt, tm = test_model(
        model,
        tokenizer,
        test_loader,
        device,
        infer_recycles=RUN_CONFIG.runtime.infer_recycles,
        esm_cache=esm_cache,
    )
    print(f"fold {fold + 1}/{RUN_CONFIG.kfold.n_splits}  mean_plddt={plddt:.4f}  mean_tm={tm:.4f}")
    return plddt, tm


def main() -> int:
    runtime = RUN_CONFIG.runtime
    training = RUN_CONFIG.training
    device = resolve_device()
    set_seed(training.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    proteins = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(RUN_CONFIG.model.name)

    esm_cache = None
    if runtime.use_esm_cache:
        if runtime.esm_cache_trunk_blocks > 0 and (training.train_recycles != 0 or runtime.infer_recycles != 0):
            raise ValueError("esm_cache_trunk_blocks > 0 requires train_recycles and infer_recycles to be 0")
        esm_cache = prepare_esm_cache(
            Path(runtime.esm_cache_dir),
            proteins,
            RUN_CONFIG.model.name,
            tokenizer,
            device,
            trunk_chunk_size=runtime.trunk_chunk_size,
            cache_trunk_blocks=runtime.esm_cache_trunk_blocks,
        )

    runner = KFoldRunner(proteins, baseline=runtime.baseline)

    if runtime.baseline:
        print(f"Loading pretrained {RUN_CONFIG.model.name} on {device}...")
        model = build_model(
            RUN_CONFIG.model.name,
            device,
            unfreeze_trunk_blocks=0,
            unfreeze_structure_module=False,
            trunk_chunk_size=runtime.trunk_chunk_size,
            gradient_checkpointing=False,
            load_esm=not runtime.use_esm_cache,
            cache_trunk_blocks=runtime.esm_cache_trunk_blocks if runtime.use_esm_cache else 0,
        )
        model.eval()
        fold_fn = partial(
            run_baseline_fold,
            model=model,
            proteins=proteins,
            device=device,
            tokenizer=tokenizer,
            esm_cache=esm_cache,
        )
    else:
        h0rff, h1rff = create_vpd_kernels(LOSS_CONFIG, HEAT_RFF_CONFIG)
        loss_fn = ESMFoldLoss(config=LOSS_CONFIG, h0rff=h0rff, h1rff=h1rff)
        fold_fn = partial(
            run_fold,
            dataset=proteins,
            device=device,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            esm_cache=esm_cache,
            n_splits=RUN_CONFIG.kfold.n_splits,
        )

    fold_plddt_scores, fold_tm_scores = runner.run(fold_fn)

    logging = RUN_CONFIG.logging
    log_path = logging.baseline_log_file if runtime.baseline else logging.finetune_log_file
    log_file = Path(log_path)
    write_log_file(log_file, fold_plddt_scores, fold_tm_scores)
    print(f"Wrote results to {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
