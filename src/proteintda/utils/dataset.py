"""SidechainNet dataset loading and protein dataloaders."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import sidechainnet as scn
import torch
from torch.utils.data import DataLoader, Sampler

from proteintda.config import RUN_CONFIG
from proteintda.utils.device import resolve_device as _resolve_device


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def select_backbone(name: str | None = None):
    backbone_name = str(name or RUN_CONFIG.runtime.get("backbone", "minifold")).lower()
    if backbone_name == "lightrosetta":
        from proteintda.lightrosetta import pipeline as lightrosetta_pipeline

        return backbone_name, lightrosetta_pipeline.BACKBONE
    if backbone_name == "minifold":
        from proteintda.minifold import pipeline as minifold_pipeline

        return backbone_name, minifold_pipeline.BACKBONE
    raise ValueError(
        f"Unknown runtime.backbone={backbone_name!r}; use 'minifold' or 'lightrosetta'"
    )


def _baseline_tm_fingerprint(backbone: str) -> dict:
    expected = {"backbone": backbone, "tda": False}
    if backbone == "minifold":
        expected["model_size"] = RUN_CONFIG.minifold.model_size
        expected["infer_recycles"] = RUN_CONFIG.runtime.infer_recycles or 0
    checkpoint = RUN_CONFIG.data.get("baseline_checkpoint")
    if checkpoint:
        expected["baseline_checkpoint"] = str(Path(checkpoint).resolve())
    return expected


def _load_baseline_tm_scores(path: Path, expected: dict) -> dict[str, float]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "scores" not in payload:
        raise ValueError(f"Baseline TM cache at {path} is missing a 'scores' object")
    for key, value in expected.items():
        if payload.get(key) != value:
            print(
                f"Baseline TM cache {key} mismatch: file has {payload.get(key)}, "
                f"expected {value}; recomputing."
            )
            return {}
    return {str(protein_id): float(tm) for protein_id, tm in payload["scores"].items()}


def _save_baseline_tm_scores(path: Path, scores: dict[str, float], expected: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **expected,
        "scores": {protein_id: float(tm) for protein_id, tm in sorted(scores.items())},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(path)
    print(f"Wrote baseline TM scores for {len(scores)} proteins to {path}")


def _build_baseline_tm_runner(backbone_name: str, backbone):
    runner = backbone.build_runner(_resolve_device(), train=False)
    checkpoint = RUN_CONFIG.data.get("baseline_checkpoint")
    if backbone_name == "lightrosetta":
        if not checkpoint:
            raise ValueError(
                "Baseline TM filter scores the original non-TDA model. "
                "For LightRoseTTA set data.baseline_checkpoint to weights trained "
                "without TDA (native losses only), or set max_baseline_tm=None."
            )
        state = torch.load(Path(checkpoint), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        runner.load_state_dict(state)
        print(f"Loaded non-TDA LightRoseTTA checkpoint for TM filter: {checkpoint}")
    elif checkpoint:
        raise ValueError(
            "data.baseline_checkpoint is only used for LightRoseTTA. "
            "MiniFold baseline TM always uses the pretrained checkpoint."
        )
    return runner


def _select_with_tm_filter(
    dataset: list,
    *,
    max_proteins: int | None,
    max_baseline_tm: float,
) -> list:
    """Keep proteins where the original non-TDA model has TM <= threshold."""
    backbone_name, backbone = select_backbone()
    expected = _baseline_tm_fingerprint(backbone_name)
    scores_path = Path(RUN_CONFIG.data.baseline_tm_scores_dir) / f"{backbone_name}.json"
    scores = _load_baseline_tm_scores(scores_path, expected)
    eval_kwargs = dict(backbone.eval_kwargs)
    runner = None
    updated = False
    kept: list = []
    removed = 0
    skipped = 0

    try:
        for protein in reversed(dataset):
            if max_proteins is not None and len(kept) >= max_proteins:
                break
            protein_id = str(protein.id)
            if protein_id not in scores:
                if runner is None:
                    runner = _build_baseline_tm_runner(backbone_name, backbone)
                    print(
                        f"Computing non-TDA baseline TM "
                        f"(backbone={backbone_name}, batch_size=1)..."
                    )
                totals, n = runner.run_batch(
                    [protein],
                    None,
                    backward=False,
                    include_loss=False,
                    include_metrics=True,
                    **eval_kwargs,
                )
                if n == 0 or "tm_score" not in totals:
                    skipped += 1
                    continue
                scores[protein_id] = float(totals["tm_score"]) / n
                updated = True

            if scores[protein_id] <= max_baseline_tm:
                kept.append(protein)
            else:
                removed += 1
    finally:
        if runner is not None:
            del runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if updated:
        _save_baseline_tm_scores(scores_path, scores, expected)

    kept.reverse()
    if removed:
        print(
            f"Removed {removed} proteins with non-TDA {backbone_name} TM > {max_baseline_tm}."
        )
    if skipped:
        print(f"Skipped {skipped} proteins with no baseline TM score.")
    if max_proteins is not None and len(kept) < max_proteins:
        print(f"Only {len(kept)}/{max_proteins} proteins remain after baseline TM filter.")
    return kept


def load_proteins(
    *,
    casp_version: str = "debug",
    scn_dir: str = "./data/sidechainnet",
    casp_thinning: int = 30,
    max_proteins: int | None = None,
    max_protein_length: int | None = None,
    allow_incomplete: bool = False,
    max_baseline_tm: float | None = None,
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
        dataset = _select_with_tm_filter(
            dataset,
            max_proteins=max_proteins,
            max_baseline_tm=float(max_baseline_tm),
        )
    elif max_proteins is not None and len(dataset) > max_proteins:
        dataset = dataset[-max_proteins :]

    print(f"Loaded {len(dataset)} proteins.")
    return dataset


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
