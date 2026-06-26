import sys
import argparse
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # I have the options that we change most frequently arbitrarily chosen to be args instead of in config
    parser = argparse.ArgumentParser(description="Fine-tune ESMFold with TDA loss functions.")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--unfreeze-trunk-blocks", type=int, default=1)
    parser.add_argument("--unfreeze-structure-module", action="store_true")
    parser.add_argument("--train-recycles", type=int, default=1)
    parser.add_argument("--trunk-chunk-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)

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
    max_proteins = data.max_proteins
    if max_proteins is not None and len(dataset) > max_proteins:
        dataset = dataset[-max_proteins:]
    # dataset = dataset[5:]
    print(f"Loaded {len(dataset)} proteins.")
    return dataset


def make_loader(proteins: list, batch_size: int, shuffle: bool) -> DataLoader:
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
    args: argparse.Namespace,
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
        unfreeze_trunk_blocks=args.unfreeze_trunk_blocks,
        unfreeze_structure_module=args.unfreeze_structure_module,
        trunk_chunk_size=args.trunk_chunk_size,
        gradient_checkpointing=training.gradient_checkpointing,
        load_esm=not runtime.use_esm_cache,
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

    for epoch in range(training.epochs):
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
            use_amp=training.amp,
            esm_cache=esm_cache,
        )
        val_plddt_score, val_tm_score = test_model(
            model,
            tokenizer,
            val_loader,
            device,
            infer_recycles=training.infer_recycles,
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
        infer_recycles=training.infer_recycles,
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
    args: argparse.Namespace,
    proteins: list,
    device: torch.device,
    tokenizer: AutoTokenizer,
    esm_cache: ESMEmbeddingCache | None,
) -> tuple[float, float]:
    training = RUN_CONFIG.training
    n_splits = RUN_CONFIG.kfold.n_splits
    test_loader = make_loader(
        [proteins[i] for i in test_idx],
        args.batch_size,
        shuffle=False,
    )
    plddt, tm = test_model(
        model,
        tokenizer,
        test_loader,
        device,
        infer_recycles=training.infer_recycles,
        esm_cache=esm_cache,
    )
    print(f"fold {fold + 1}/{n_splits}  mean_plddt={plddt:.4f}  mean_tm={tm:.4f}")
    return plddt, tm


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    runtime = RUN_CONFIG.runtime
    training = RUN_CONFIG.training
    set_seed(training.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    proteins = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(RUN_CONFIG.model.name)

    esm_cache = None
    if runtime.use_esm_cache:
        esm_cache = prepare_esm_cache(
            Path(runtime.esm_cache_dir),
            proteins,
            RUN_CONFIG.model.name,
            tokenizer,
            device,
            trunk_chunk_size=args.trunk_chunk_size,
        )

    runner = KFoldRunner(proteins, baseline=args.baseline)

    if args.baseline:
        print(f"Loading pretrained {RUN_CONFIG.model.name} on {device}...")
        model = build_model(
            RUN_CONFIG.model.name,
            device,
            unfreeze_trunk_blocks=0,
            unfreeze_structure_module=False,
            trunk_chunk_size=args.trunk_chunk_size,
            gradient_checkpointing=False,
            load_esm=not runtime.use_esm_cache,
        )
        model.eval()
        fold_fn = partial(
            run_baseline_fold,
            model=model,
            args=args,
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
            args=args,
            device=device,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            esm_cache=esm_cache,
            n_splits=RUN_CONFIG.kfold.n_splits,
        )

    fold_plddt_scores, fold_tm_scores = runner.run(fold_fn)

    logging = RUN_CONFIG.logging
    log_path = logging.baseline_log_file if args.baseline else logging.finetune_log_file
    log_file = Path(log_path)
    write_log_file(log_file, fold_plddt_scores, fold_tm_scores)
    print(f"Wrote results to {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
