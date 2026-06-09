"""Entry point for ESMFold fine-tuning with TDA Wasserstein loss."""

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
    train_one_epoch,
    test_model,
    trainable_parameter_count,
)
from loss import ESMFoldLoss
from model_config import LOSS_CONFIG


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ESMFold with Wasserstein TDA loss.")
    parser.add_argument("--model", default="facebook/esmfold_v1")
    parser.add_argument("--casp-version", default="debug")
    parser.add_argument("--casp-thinning", type=int, default=30)
    parser.add_argument("--allow-incomplete", type=bool, default=False)
    parser.add_argument("--scn-dir", type=Path, default=Path("sidechainnet_data"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--unfreeze-trunk-blocks", type=int, default=2)
    parser.add_argument("--unfreeze-structure-module", type=bool, default=False)
    parser.add_argument("--unfreeze-esm-layers", type=int, default=0)
    parser.add_argument("--train-recycles", type=int, default=1) # Should be 8, but for memory purposes keeping it low for now
    parser.add_argument("--trunk-chunk-size", type=int, default=4)
    parser.add_argument("--amp", type=bool, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-file", type=Path, default=Path("logs/kfold_test_scores.log"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)

    set_seed(seed=42)

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Loading SidechainNet...")
    dataset = scn.load(
        casp_version = args.casp_version,
        casp_thinning = args.casp_thinning,
        scn_dataset = True,
        scn_dir = str(args.scn_dir),
        force_download = False,
        complete_structures_only = not args.allow_incomplete,
    )

    if len(dataset) > 1000:
        dataset = dataset[-1000:]
    # dataset = dataset[:5]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_plddt_scores: list[float] = []
    fold_tm_scores: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        set_seed(42 + fold)

        print(f"Fold {fold + 1}/{kf.n_splits}: loading fresh model on {device}...")
        model = build_model(
            args.model,
            device,
            unfreeze_esm_layers=args.unfreeze_esm_layers,
            unfreeze_trunk_blocks=args.unfreeze_trunk_blocks,
            unfreeze_structure_module=args.unfreeze_structure_module,
            trunk_chunk_size=args.trunk_chunk_size,
        )
        trainable, total = trainable_parameter_count(model)
        print(f"Trainable parameters: {trainable:,} / {total:,}")
        
        # This will give us a 60/20/20 split
        train_idx, val_idx = train_test_split(train_idx, test_size=0.25)

        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        test_dataset = [dataset[i] for i in test_idx]

        train_loader = DataLoader(
            dataset = train_dataset,
            batch_size = args.batch_size,
            shuffle = True,
            collate_fn = lambda x: x,
        )

        val_loader = DataLoader(
            dataset = val_dataset,
            batch_size = args.batch_size,
            shuffle = False,
            collate_fn = lambda x: x,
        )

        test_loader = DataLoader(
            dataset = test_dataset,
            batch_size = args.batch_size,
            shuffle = False,
            collate_fn = lambda x: x,
        )

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
                loss_fn=ESMFoldLoss(config=LOSS_CONFIG),
                unfreeze_esm_layers=args.unfreeze_esm_layers,
                unfreeze_trunk_blocks=args.unfreeze_trunk_blocks,
                unfreeze_structure_module=args.unfreeze_structure_module,
                train_recycles=args.train_recycles,
                use_amp=args.amp,
            )
            _, val_tm_score = test_model(
                model,
                tokenizer,
                val_loader,
                device,
            )
            if val_tm_score > max_val_tm:
                max_val_tm = val_tm_score
                patience = 0
                best_model_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1

            print(
                f"epoch {epoch + 1}/{args.epochs}"
                + "\n" + "  ".join(f"{key}={value:.4f}" for key, value in metrics.items())
                + f"\nfold {fold + 1}/{kf.n_splits}"
            )

            if patience > args.patience:
                print(f"Early stoppping at epoch {epoch}")
                break

        if best_model_weights is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})

        plddt_score, tm_score = test_model(
            model,
            tokenizer,
            test_loader,
            device,
        )
        print(f"fold {fold + 1}/{kf.n_splits}  mean_plddt={plddt_score:.4f}  mean_tm={tm_score:.4f}")
        fold_plddt_scores.append(plddt_score)
        fold_tm_scores.append(tm_score)

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as log_file:
        for fold_idx, (plddt, tm) in enumerate(zip(fold_plddt_scores, fold_tm_scores), start=1):
            log_file.write(f"fold {fold_idx}: mean_plddt={plddt:.4f} mean_tm={tm:.4f}\n")
        log_file.write(
            f"mean_plddt mean={np.mean(fold_plddt_scores):.4f} var={np.var(fold_plddt_scores):.4f}\n"
        )
        log_file.write(
            f"mean_tm mean={np.mean(fold_tm_scores):.4f} var={np.var(fold_tm_scores):.4f}\n"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
