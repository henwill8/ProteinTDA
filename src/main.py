import sys
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, train_test_split

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
from loss import ESMFoldLoss
from config import HEAT_RFF_CONFIG, LOSS_CONFIG
from vpd_macros import create_vpd_kernels


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ESMFold with Wasserstein TDA loss.")
    parser.add_argument("--model", default="facebook/esmfold_v1")
    parser.add_argument("--casp-version", default="debug")
    parser.add_argument("--casp-thinning", type=int, default=30)
    parser.add_argument("--allow-incomplete", type=bool, default=False)
    parser.add_argument("--scn-dir", type=Path, default=Path("data/sidechainnet"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--unfreeze-trunk-blocks", type=int, default=1)
    parser.add_argument("--unfreeze-structure-module", type=bool, default=False)
    parser.add_argument("--train-recycles", type=int, default=1)
    parser.add_argument("--trunk-chunk-size", type=int, default=1)
    # gradient checkpointing only works for the esm encoder, if use-esm-cache is enabled, checkpointing is not supported
    parser.add_argument("--gradient-checkpointing", type=bool, default=True)
    parser.add_argument("--amp", type=bool, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-file", type=Path, default=Path("logs/kfold_test_scores.log"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-esm-cache", type=bool, default=True)
    # not sure if we need the option to change which cache we use, maybe remove and turn into a constant
    parser.add_argument("--esm-cache-dir", type=Path, default=Path("cache/esm_embeddings"))
    return parser.parse_args(argv)

# For lower memory usage, try batch size 1, chunk size 1 or 2? (would be really slow), recycles 1

def load_dataset(args: argparse.Namespace) -> list:
    print("Loading SidechainNet...")
    dataset = scn.load(
        casp_version=args.casp_version,
        casp_thinning=args.casp_thinning,
        scn_dataset=True,
        scn_dir=str(args.scn_dir),
        force_download=False,
        complete_structures_only=not args.allow_incomplete,
    )
    # if len(dataset) > 1000:
    #     dataset = dataset[-1000:]
    dataset = dataset[-5:]
    return dataset


def make_loader(proteins: list, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset=proteins,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: x,
    )


def run_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: list,
    args: argparse.Namespace,
    device: torch.device,
    tokenizer: AutoTokenizer,
    loss_fn: ESMFoldLoss,
    esm_cache: ESMEmbeddingCache | None,
    n_splits: int,
) -> tuple[float, float]:
    set_seed(args.seed + fold)

    print(f"Fold {fold + 1}/{n_splits}: loading fresh model on {device}...")
    model = build_model(
        args.model,
        device,
        unfreeze_trunk_blocks=args.unfreeze_trunk_blocks,
        unfreeze_structure_module=args.unfreeze_structure_module,
        trunk_chunk_size=args.trunk_chunk_size,
        gradient_checkpointing=args.gradient_checkpointing,
        load_esm=not args.use_esm_cache,
    )
    trainable, total = trainable_parameter_count(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    train_idx, val_idx = train_test_split(train_idx, test_size=0.25)
    train_loader = make_loader([dataset[i] for i in train_idx], args.batch_size, shuffle=True)
    val_loader = make_loader([dataset[i] for i in val_idx], args.batch_size, shuffle=False)
    test_loader = make_loader([dataset[i] for i in test_idx], args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    best_model_weights = None
    max_val_tm = 0.0
    patience = 0

    for epoch in range(args.epochs):
        metrics = train_one_epoch(
            model,
            tokenizer,
            train_loader,
            optimizer,
            device,
            loss_fn=loss_fn,
            unfreeze_trunk_blocks=args.unfreeze_trunk_blocks,
            unfreeze_structure_module=args.unfreeze_structure_module,
            train_recycles=args.train_recycles,
            use_amp=args.amp,
            esm_cache=esm_cache,
        )
        val_plddt_score, val_tm_score = test_model(
            model,
            tokenizer,
            val_loader,
            device,
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
            f"epoch {epoch + 1}/{args.epochs}"
            + "\n" + "  ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            + f"\nfold {fold + 1}/{n_splits}"
        )

        if patience > args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})

    plddt_score, tm_score = test_model(
        model,
        tokenizer,
        test_loader,
        device,
        esm_cache=esm_cache,
    )
    print(f"fold {fold + 1}/{n_splits}  mean_plddt={plddt_score:.4f}  mean_tm={tm_score:.4f}")
    return plddt_score, tm_score


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    set_seed(args.seed)

    h0rff, h1rff = create_vpd_kernels(LOSS_CONFIG, HEAT_RFF_CONFIG)
    loss_fn = ESMFoldLoss(config=LOSS_CONFIG, h0rff=h0rff, h1rff=h1rff)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch.backends.cuda.matmul.allow_tf32 = True

    proteins = load_dataset(args)

    esm_cache: ESMEmbeddingCache | None = None
    if args.use_esm_cache:
        esm_cache = prepare_esm_cache(
            args.esm_cache_dir,
            proteins,
            args.model,
            tokenizer,
            device,
            trunk_chunk_size=args.trunk_chunk_size,
        )

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_plddt_scores: list[float] = []
    fold_tm_scores: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(proteins)):
        plddt, tm = run_fold(
            fold,
            train_idx,
            test_idx,
            proteins,
            args,
            device,
            tokenizer,
            loss_fn,
            esm_cache,
            kf.n_splits,
        )
        fold_plddt_scores.append(plddt)
        fold_tm_scores.append(tm)

    write_log_file(args.log_file, fold_plddt_scores, fold_tm_scores)
    return 0


if __name__ == "__main__":
    sys.exit(main())
