"""Entry point for ESMFold fine-tuning with TDA Wasserstein loss."""

import sys
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

from sidechainnet.dataloaders.SCNDataset import SCNDataset

import torch
from torch.utils.data import DataLoader, dataloader
from transformers import AutoTokenizer, EsmForProteinFolding

from esmfold_finetune import (
    freeze_except_last_esm_layers,
    train_one_epoch,
    trainable_parameter_count,
)
from load_dataset import load_sidechainnet

def set_seed(seed=42):
    torch.manual_seed(42)
    np.random.seed(42)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ESMFold with Wasserstein TDA loss.")
    parser.add_argument("--model", default="facebook/esmfold_v1")
    parser.add_argument("--casp-version", default="debug")
    parser.add_argument("--casp-thinning", type=int, default=30)
    parser.add_argument("--scn-dir", type=Path, default=Path("sidechainnet_data"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--unfreeze-esm-layers", type=int, default=2)
    parser.add_argument("--wasserstein-h0-weight", type=float, default=1.0)
    parser.add_argument("--wasserstein-h1-weight", type=float, default=1.0)
    parser.add_argument("--max-rips-dimension", type=int, default=2)
    parser.add_argument("--hom-dim", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128, help="Skip longer proteins (Rips cost).")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/esmfold_finetune"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Sending model to device ({device})...")
    model = EsmForProteinFolding.from_pretrained(args.model, low_cpu_mem_usage=True).to(device)
    model.esm = model.esm.half()

    print(f"Freezing all parameters except the last {args.unfreeze_esm_layers} ESM encoder layers...")
    freeze_except_last_esm_layers(model, n_layers=args.unfreeze_esm_layers)

    print("Counting trainable parameters...")
    trainable, total = trainable_parameter_count(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    torch.backends.cuda.matmul.allow_tf32 = True
    model.trunk.set_chunk_size(64)

    print("Loading SidechainNet...")
    dataset = load_sidechainnet(
        casp_version=args.casp_version,
        casp_thinning=args.casp_thinning,
        scn_dir=args.scn_dir,
        complete_structures_only=not args.allow_incomplete,
    )
    
    if len(dataset) > 1000:
        dataset=dataset[-1000:]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = [dataset[i] for i in train_idx]
        test_dataset = [dataset[i] for i in test_idx]
        
        train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=32,
                    shuffle=True
                )

        test_laoder = DataLoader(
                    dataset=test_dataset,
                    batch_size=32,
                    shuffle=True
                )

        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

        for epoch in range(args.epochs):
            metrics = train_one_epoch(
                model,
                tokenizer,
                train_loader,
                optimizer,
                device,
                wasserstein_h0_weight=args.wasserstein_h0_weight,
                wasserstein_h1_weight=args.wasserstein_h1_weight,
                max_rips_dimension=args.max_rips_dimension,
                hom_dim=args.hom_dim,
                max_length=args.max_length,
            )
            print(
                f"epoch {epoch + 1}/{args.epochs}  "
                f"loss={metrics['total']:.4f}  "
                f"structure={metrics['structure']:.4f}  "
                f"wass_h0={metrics['wasserstein_h0']:.4f}  "
                f"wass_h1={metrics['wasserstein_h1']:.4f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
