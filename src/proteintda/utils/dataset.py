"""SidechainNet dataset loading and protein dataloaders."""

import numpy as np
import sidechainnet as scn
import torch
from torch.utils.data import DataLoader

from proteintda.config import RUN_CONFIG


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _load_sidechainnet_proteins(
    *,
    max_proteins: int | None,
    max_protein_length: int | None,
    allow_incomplete: bool,
) -> list:
    data = RUN_CONFIG.data
    print("Loading SidechainNet...")
    dataset = scn.load(
        casp_version=data.casp_version,
        casp_thinning=data.casp_thinning,
        scn_dataset=True,
        scn_dir=data.scn_dir,
        force_download=False,
        complete_structures_only=not allow_incomplete,
    )

    # Sidechainnet includes proteins with '.' in the mask which we don't want to use
    if not allow_incomplete:
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

    if max_protein_length is not None:
        before = len(dataset)
        dataset = [protein for protein in dataset if len(protein.seq) <= max_protein_length]
        removed = before - len(dataset)
        if removed:
            print(f"Removed {removed} proteins longer than {max_protein_length} residues.")

    if max_proteins is not None and len(dataset) > max_proteins:
        dataset = dataset[-max_proteins :]
    print(f"Loaded {len(dataset)} proteins.")
    return dataset


def load_all_proteins(allow_incomplete: bool = False) -> list:
    return _load_sidechainnet_proteins(
        max_proteins=None,
        max_protein_length=None,
        allow_incomplete=allow_incomplete,
    )


def load_dataset() -> list:
    data = RUN_CONFIG.data
    return _load_sidechainnet_proteins(
        max_proteins=data.max_proteins,
        max_protein_length=data.max_protein_length,
        allow_incomplete=data.allow_incomplete,
    )


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
