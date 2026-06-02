"""
Load SidechainNet and build PyTorch DataLoaders.

Each batch contains ``SCNProtein`` objects for use with ``sidechainnet_graph.scn_protein_to_graph``.
SidechainNet is downloaded automatically on first use via ``sidechainnet.load``.
"""

from pathlib import Path
from typing import Any

import sidechainnet as scn
from sidechainnet.dataloaders.SCNDataset import SCNDataset
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from torch.utils.data import DataLoader, Dataset


def load_sidechainnet(
    *,
    casp_version: int | str = 12,
    casp_thinning: int = 30,
    scn_dir: str | Path = "sidechainnet_data",
    force_download: bool = False,
    complete_structures_only: bool = False,
) -> SCNDataset:
    """
    Download (if needed) and return an ``SCNDataset``.

    Index or iterate to get ``SCNProtein`` objects::

        dataset = load_sidechainnet(casp_version="debug")
        protein = dataset["1HD1_1_A"]
    """
    return scn.load(
        casp_version=casp_version,
        casp_thinning=casp_thinning,
        scn_dataset=True,
        scn_dir=str(scn_dir),
        force_download=force_download,
        complete_structures_only=complete_structures_only,
    )


class SidechainNetSplitDataset(Dataset):
    """One SidechainNet split (e.g. ``train``); each item is an ``SCNProtein``."""

    def __init__(self, dataset: SCNDataset, split: str) -> None:
        if split not in dataset.splits:
            raise KeyError(f"Split {split!r} not in dataset. Available: {dataset.splits}")
        self._dataset = dataset
        self._split = split
        self._protein_ids = list(dataset.split_to_ids[split])

    def __len__(self) -> int:
        return len(self._protein_ids)

    def __getitem__(self, idx: int) -> SCNProtein:
        return self._dataset[self._protein_ids[idx]]

    @property
    def split(self) -> str:
        return self._split


def collate_scn_proteins(batch: list[SCNProtein]) -> dict[str, Any]:
    """Collate a batch of ``SCNProtein`` objects (variable length; no coordinate padding)."""
    return {
        "protein": batch,
        "protein_id": [protein.id for protein in batch],
        "sequence": [str(protein.seq) for protein in batch],
        "mask": [str(protein.mask) for protein in batch],
    }


def make_dataloader(
    dataset: SidechainNetSplitDataset,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_scn_proteins,
    )


def load_dataloaders(
    *,
    casp_version: int | str = 12,
    casp_thinning: int = 30,
    scn_dir: str | Path = "sidechainnet_data",
    force_download: bool = False,
    complete_structures_only: bool = False,
    splits: list[str] | None = None,
    batch_size: int = 8,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """
    Load SidechainNet and return one DataLoader per split.

    Each batch is a dict with key ``protein`` (list of ``SCNProtein``).
    """
    dataset = load_sidechainnet(
        casp_version=casp_version,
        casp_thinning=casp_thinning,
        scn_dir=scn_dir,
        force_download=force_download,
        complete_structures_only=complete_structures_only,
    )
    split_names = splits if splits is not None else dataset.splits
    loaders: dict[str, DataLoader] = {}
    for split in split_names:
        loaders[split] = make_dataloader(
            SidechainNetSplitDataset(dataset, split),
            batch_size=batch_size,
            shuffle=shuffle_train if split == "train" else False,
            num_workers=num_workers,
        )
    return loaders
