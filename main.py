"""Entry point for ESMFold fine-tuning with TDA Wasserstein loss."""

import sys

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

from esmfold_finetune import (
    freeze_except_last_esm_layers,
    parse_args,
    train_one_epoch,
    trainable_parameter_count,
)
from load_dataset import SidechainNetSplitDataset, load_sidechainnet, make_dataloader
from persistence import pd_from_graph as PDFromGraph
from persistence import wasserstein_distance as WassersteinDistance


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmForProteinFolding.from_pretrained(args.model).to(device)
    freeze_except_last_esm_layers(model, n_layers=args.unfreeze_esm_layers)
    trainable, total = trainable_parameter_count(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    print("Loading SidechainNet...")
    dataset = load_sidechainnet(
        casp_version=args.casp_version,
        casp_thinning=args.casp_thinning,
        scn_dir=args.scn_dir,
        complete_structures_only=not args.allow_incomplete,
    )
    loader = make_dataloader(
        SidechainNetSplitDataset(dataset, args.split),
        batch_size=args.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    for epoch in range(args.epochs):
        metrics = train_one_epoch(
            model,
            tokenizer,
            loader,
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
