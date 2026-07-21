"""SidechainNet dataset loading and protein dataloaders."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import sidechainnet as scn
import torch
from torch.utils.data import DataLoader, Sampler
from tmtools import tm_align

from proteintda.config import RUN_CONFIG
from proteintda.utils.conversions import SideChainAtom, atom_positions_from_sidechainnet


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _load_baseline_tm_scores(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "scores" in payload:
        scores = payload["scores"]
        cached_size = payload.get("model_size")
        expected_size = RUN_CONFIG.runtime.model_size
        if cached_size is not None and cached_size != expected_size:
            print(
                f"Baseline TM cache model_size={cached_size!r} does not match "
                f"runtime.model_size={expected_size!r}; recomputing missing scores."
            )
            return {}
    else:
        scores = payload
    return {str(protein_id): float(tm) for protein_id, tm in scores.items()}


def _save_baseline_tm_scores(path: Path, scores: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_size": RUN_CONFIG.runtime.model_size,
        "scores": {protein_id: float(tm) for protein_id, tm in sorted(scores.items())},
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote baseline TM scores for {len(scores)} proteins to {path}")


def _resolve_device() -> torch.device:
    device = RUN_CONFIG.runtime.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _compute_baseline_tm_scores(proteins: list, *, runner) -> dict[str, float]:
    runtime = RUN_CONFIG.runtime
    batch_size = max(1, int(RUN_CONFIG.training.batch_size))
    scores: dict[str, float] = {}

    print(
        f"Computing baseline TM for {len(proteins)} proteins "
        f"(model={runtime.model_size}, batch_size={batch_size})..."
    )
    for start in range(0, len(proteins), batch_size):
        batch = proteins[start : start + batch_size]
        outputs = runner._forward_batch(
            batch,
            None,
            runtime.infer_recycles,
            include_metrics=True,
        )
        if outputs is None:
            continue
        pred_ca = outputs.get("pred_ca")
        if pred_ca is None:
            continue
        for i, protein in enumerate(batch):
            length = len(str(protein.seq))
            exp_ca = atom_positions_from_sidechainnet(protein, SideChainAtom.CA).cpu().numpy()
            alignment = tm_align(
                pred_ca[i, :length].numpy(),
                exp_ca,
                str(protein.seq),
                str(protein.seq),
            )
            scores[str(protein.id)] = float(alignment.tm_norm_chain2)
    return scores


def _select_with_tm_filter(
    dataset: list,
    *,
    max_proteins: int | None,
    max_baseline_tm: float,
    scores_path: Path,
) -> list:
    """Keep up to max_proteins with baseline TM <= threshold."""
    scores = _load_baseline_tm_scores(scores_path)
    update_scores = False
    runner = None
    batch_size = max(1, int(RUN_CONFIG.training.batch_size))

    kept: list = []
    removed = 0
    removed_examples: list[tuple[str, float]] = []
    next_idx = len(dataset) - 1

    def ensure_scores(proteins: list) -> None:
        nonlocal runner, update_scores
        missing = [protein for protein in proteins if str(protein.id) not in scores]
        if not missing:
            return
        if runner is None:
            from proteintda.minifold.runner import MiniFoldRunner

            runtime = RUN_CONFIG.runtime
            runner = MiniFoldRunner(
                Path(runtime.minifold_cache_dir),
                model_size=runtime.model_size,
                device=_resolve_device(),
            )
        scores.update(_compute_baseline_tm_scores(missing, runner=runner))
        update_scores = True

    while next_idx >= 0 and (max_proteins is None or len(kept) < max_proteins):
        if max_proteins is None:
            take = min(batch_size, next_idx + 1)
        else:
            need = max_proteins - len(kept)
            take = min(max(need, batch_size), next_idx + 1)
        chunk = dataset[next_idx - take + 1 : next_idx + 1]
        next_idx -= take

        ensure_scores(chunk)

        for protein in reversed(chunk):
            if max_proteins is not None and len(kept) >= max_proteins:
                break
            tm = scores.get(str(protein.id))
            if tm is None or tm <= max_baseline_tm:
                kept.append(protein)
            else:
                removed += 1
                if len(removed_examples) < 5:
                    removed_examples.append((str(protein.id), tm))

    if update_scores:
        _save_baseline_tm_scores(scores_path, scores)

    kept = list(reversed(kept))
    if removed:
        print(f"Removed {removed} proteins with baseline TM > {max_baseline_tm}.")
    if max_proteins is not None and len(kept) < max_proteins:
        print(f"Only {len(kept)}/{max_proteins} proteins remain after baseline TM filter.")
    return kept


def _load_sidechainnet_proteins(
    *,
    casp_version: str,
    scn_dir: str,
    casp_thinning: int,
    max_proteins: int | None,
    max_protein_length: int | None,
    allow_incomplete: bool,
    max_baseline_tm: float | None = None,
    baseline_tm_scores_path: str | None = None,
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

    if max_baseline_tm is not None:
        if baseline_tm_scores_path is None:
            raise ValueError(
                "baseline_tm_scores_path is required when max_baseline_tm is set"
            )
        dataset = _select_with_tm_filter(
            dataset,
            max_proteins=max_proteins,
            max_baseline_tm=float(max_baseline_tm),
            scores_path=Path(baseline_tm_scores_path),
        )
    elif max_proteins is not None and len(dataset) > max_proteins:
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
    max_baseline_tm: float | None = None,
    baseline_tm_scores_path: str | None = None,
) -> list:
    return _load_sidechainnet_proteins(
        casp_version=casp_version,
        scn_dir=scn_dir,
        casp_thinning=casp_thinning,
        max_proteins=max_proteins,
        max_protein_length=max_protein_length,
        allow_incomplete=allow_incomplete,
        max_baseline_tm=max_baseline_tm,
        baseline_tm_scores_path=baseline_tm_scores_path,
    )


def load_all_proteins(
    *,
    casp_version: str = "debug",
    scn_dir: str = "./data/sidechainnet",
    casp_thinning: int = 30,
    allow_incomplete: bool = False,
    max_baseline_tm: float | None = None,
    baseline_tm_scores_path: str | None = None,
) -> list:
    return load_proteins(
        casp_version=casp_version,
        scn_dir=scn_dir,
        casp_thinning=casp_thinning,
        max_proteins=None,
        max_protein_length=None,
        allow_incomplete=allow_incomplete,
        max_baseline_tm=max_baseline_tm,
        baseline_tm_scores_path=baseline_tm_scores_path,
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
        max_baseline_tm=data.max_baseline_tm,
        baseline_tm_scores_path=data.baseline_tm_scores_path,
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
