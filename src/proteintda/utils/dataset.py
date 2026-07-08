"""SidechainNet dataset loading and protein dataloaders."""

from collections import defaultdict

import numpy as np
import sidechainnet as scn
import torch
from torch.utils.data import DataLoader, Sampler

from proteintda.config import RUN_CONFIG


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _load_sidechainnet_proteins(
    *,
    casp_version: str,
    scn_dir: str,
    casp_thinning: int,
    max_proteins: int | None,
    max_protein_length: int | None,
    allow_incomplete: bool,
) -> list:
    print(
        f"Loading SidechainNet casp={casp_version}, thinning={casp_thinning}, "
        f"dir={scn_dir}..."
    )
    dataset = scn.load(
        casp_version=casp_version,
        casp_thinning=casp_thinning,
        scn_dataset=True,
        scn_dir=scn_dir,
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


def load_proteins(
    *,
    casp_version: str = "debug",
    scn_dir: str = "./data/sidechainnet",
    casp_thinning: int = 30,
    max_proteins: int | None = None,
    max_protein_length: int | None = None,
    allow_incomplete: bool = False,
) -> list:
    return _load_sidechainnet_proteins(
        casp_version=casp_version,
        scn_dir=scn_dir,
        casp_thinning=casp_thinning,
        max_proteins=max_proteins,
        max_protein_length=max_protein_length,
        allow_incomplete=allow_incomplete,
    )


def load_all_proteins(
    *,
    casp_version: str = "debug",
    scn_dir: str = "./data/sidechainnet",
    casp_thinning: int = 30,
    allow_incomplete: bool = False,
) -> list:
    return load_proteins(
        casp_version=casp_version,
        scn_dir=scn_dir,
        casp_thinning=casp_thinning,
        max_proteins=None,
        max_protein_length=None,
        allow_incomplete=allow_incomplete,
    )


def load_dataset() -> list:
    data = RUN_CONFIG.data
    return load_proteins(
        casp_version=data.casp_version,
        scn_dir=data.scn_dir,
        casp_thinning=data.casp_thinning,
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


def _protein_length(protein) -> int:
    return len(str(protein.seq))


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Batch indices so proteins in each batch have similar sequence lengths."""

    def __init__(
        self,
        proteins: list,
        batch_size: int,
        *,
        bucket_size: int,
        shuffle: bool,
        generator: torch.Generator | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if bucket_size < 1:
            raise ValueError("bucket_size must be >= 1")
        self.proteins = proteins
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self) -> int:
        n = len(self.proteins)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, protein in enumerate(self.proteins):
            bucket = (_protein_length(protein) - 1) // self.bucket_size
            buckets[bucket].append(index)

        batches: list[list[int]] = []
        for bucket_key in sorted(buckets):
            indices = buckets[bucket_key]
            if self.shuffle:
                order = torch.randperm(len(indices), generator=self.generator).tolist()
                indices = [indices[i] for i in order]
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start : start + self.batch_size])

        if self.shuffle:
            order = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[i] for i in order]

        yield from batches


def _length_bucketing_enabled(
    batch_size: int,
    *,
    length_bucketing: bool | None,
) -> bool:
    if batch_size <= 1:
        return False
    if length_bucketing is None:
        return bool(RUN_CONFIG.training.get("length_bucketing", True))
    return length_bucketing


def make_loader(
    proteins: list,
    batch_size: int,
    *,
    shuffle: bool = False,
    max_proteins: int | None = None,
    rng: np.random.Generator | None = None,
    length_bucketing: bool | None = None,
    length_bucket_size: int | None = None,
) -> DataLoader:
    if max_proteins is not None:
        if rng is None:
            raise ValueError("rng is required when max_proteins is set")
        proteins = sample_proteins(proteins, max_proteins, rng)

    if _length_bucketing_enabled(batch_size, length_bucketing=length_bucketing):
        if length_bucket_size is None:
            length_bucket_size = int(RUN_CONFIG.training.get("length_bucket_size", 100))
        generator = None
        if shuffle:
            generator = torch.Generator()
            if rng is not None:
                generator.manual_seed(int(rng.integers(0, 2**63)))
        batch_sampler = LengthBucketBatchSampler(
            proteins,
            batch_size,
            bucket_size=length_bucket_size,
            shuffle=shuffle,
            generator=generator,
        )
        return DataLoader(
            dataset=proteins,
            batch_sampler=batch_sampler,
            collate_fn=lambda x: x,
        )

    return DataLoader(
        dataset=proteins,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: x,
    )
