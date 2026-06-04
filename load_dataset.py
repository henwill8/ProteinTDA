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


def collate_scn_proteins(batch: list[SCNProtein]) -> dict[str, Any]:
    """Collate a batch of ``SCNProtein`` objects (variable length; no coordinate padding)."""
    return {
        "protein": batch,
        "protein_id": [protein.id for protein in batch],
        "sequence": [str(protein.seq) for protein in batch],
        "mask": [str(protein.mask) for protein in batch],
    }


