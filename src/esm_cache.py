import hashlib
from pathlib import Path

import torch
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from tqdm import tqdm
from transformers import EsmForProteinFolding
from transformers.models.esm.modeling_esmfold import (
    EsmForProteinFoldingOutput,
    categorical_lddt,
)
from transformers.models.esm.openfold_utils import (
    compute_predicted_aligned_error,
    compute_tm,
    make_atom14_masks,
)


def _sequence(protein: SCNProtein) -> str:
    return str(protein.seq)


def _cache_path(cache_dir: Path, sequence: str) -> Path:
    digest = hashlib.sha256(f"{sequence}".encode()).hexdigest()[:16]
    return cache_dir / f"{digest}.pt"


def _entry_to_cpu(entry: dict) -> dict:
    """Keep cached trunk inputs on CPU to save VRAM."""
    return {
        "sequence": entry["sequence"],
        "pairwise_state_dim": entry["pairwise_state_dim"],
        **{
            key: entry[key].detach().cpu()
            for key in ("s_s_0", "aa", "attention_mask", "position_ids")
        },
    }


def _load_entry(path: Path, *, sequence: str) -> dict:
    entry = torch.load(path, map_location="cpu", weights_only=False)
    _validate_entry(entry, sequence=sequence)
    return _entry_to_cpu(entry)


def _trunk_inputs_from_entry(entry: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    s_s_0 = entry["s_s_0"].to(device, non_blocking=device.type == "cuda")
    aa = entry["aa"].to(device, non_blocking=device.type == "cuda")
    attention_mask = entry["attention_mask"].to(device, non_blocking=device.type == "cuda")
    position_ids = entry["position_ids"].to(device, non_blocking=device.type == "cuda")
    batch_size, length = aa.shape
    pairwise_state_dim = entry["pairwise_state_dim"]
    s_z_0 = s_s_0.new_zeros(batch_size, length, length, pairwise_state_dim)
    return s_s_0, s_z_0, aa, position_ids, attention_mask


def _esmfold_output_from_trunk(
    model: EsmForProteinFolding,
    trunk_out: dict,
    *,
    aa: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> EsmForProteinFoldingOutput:
    """Runs trunk cached ESM embeddings. Based on transformers/models/esm/modeling_esmfold.py:2126."""
    batch_size, length = aa.shape
    structure = {
        k: v
        for k, v in trunk_out.items()
        if k
        in [
            "s_z",
            "s_s",
            "frames",
            "sidechain_frames",
            "unnormalized_angles",
            "angles",
            "positions",
            "states",
        ]
    }

    disto_logits = model.distogram_head(structure["s_z"])
    disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
    structure["distogram_logits"] = disto_logits

    structure["lm_logits"] = model.lm_head(structure["s_s"])
    structure["aatype"] = aa
    make_atom14_masks(structure)
    for key in ("atom14_atom_exists", "atom37_atom_exists"):
        structure[key] *= attention_mask.unsqueeze(-1)
    structure["residue_index"] = position_ids

    lddt_head = model.lddt_head(structure["states"]).reshape(
        structure["states"].shape[0], batch_size, length, -1, model.lddt_bins
    )
    structure["lddt_head"] = lddt_head
    structure["plddt"] = categorical_lddt(lddt_head[-1], bins=model.lddt_bins)

    structure["ptm_logits"] = model.ptm_head(structure["s_z"])
    structure["ptm"] = compute_tm(structure["ptm_logits"], max_bin=31, no_bins=model.distogram_bins)
    structure.update(
        compute_predicted_aligned_error(
            structure["ptm_logits"],
            max_bin=31,
            no_bins=model.distogram_bins,
        )
    )
    return EsmForProteinFoldingOutput(**structure)


def _validate_entry(entry: dict, *, sequence: str) -> None:
    expected = {
        "sequence": sequence,
    }
    for key, value in expected.items():
        if entry.get(key) != value:
            raise ValueError(
                f"ESM cache {key} mismatch: file has {entry.get(key)!r}, expected {value!r}"
            )


class ESMEmbeddingCache:
    """
    Cache ESM embeddings to disk and retrieve for use as inputs to the trunk.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._entries: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, protein: SCNProtein) -> bool:
        sequence = _sequence(protein)
        if sequence in self._entries:
            return True
        return _cache_path(self.cache_dir, sequence).is_file()

    def cached_count(self, proteins: list[SCNProtein]) -> int:
        return sum(1 for protein in proteins if protein in self)

    def missing(self, proteins: list[SCNProtein]) -> list[SCNProtein]:
        """Return dataset proteins with no file on disk."""
        return [protein for protein in proteins if protein not in self]

    @torch.no_grad()
    def compute_trunk_inputs(
        self,
        model: EsmForProteinFolding,
        tokenizer,
        protein: SCNProtein,
        device: torch.device,
    ) -> dict:
        """Run the ESM path and return trunk inputs for one protein. Based on transformers/models/esm/modeling_esmfold.py:2044."""
        sequence = _sequence(protein)
        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
        aa = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(aa, device=device)
        else:
            attention_mask = attention_mask.to(device)

        position_ids = torch.arange(aa.shape[1], device=device).expand_as(aa)
        cfg = model.config.esmfold_config

        esmaa = model.af2_idx_to_esm_idx(aa, attention_mask)

        esm_s = model.compute_language_model_representations(esmaa)
        esm_s = esm_s.to(model.esm_s_combine.dtype)

        if cfg.esm_ablate_sequence:
            esm_s = esm_s * 0

        esm_s = (model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = model.esm_s_mlp(esm_s)

        if cfg.embed_aa:
            s_s_0 = s_s_0 + model.embedding(aa)

        entry = _entry_to_cpu(
            {
                "sequence": sequence,
                "s_s_0": s_s_0,
                "aa": aa,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "pairwise_state_dim": cfg.trunk.pairwise_state_dim,
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return entry

    def store(
        self,
        proteins: list[SCNProtein],
        model: EsmForProteinFolding,
        tokenizer,
        device: torch.device,
        *,
        show_progress: bool = True,
    ) -> None:
        """Compute and write any missing entries to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        iterator = proteins
        if show_progress:
            iterator = tqdm(proteins, desc="caching esm embeddings", leave=False)

        for protein in iterator:
            sequence = _sequence(protein)
            path = _cache_path(self.cache_dir, sequence)

            if path.is_file():
                continue

            entry = self.compute_trunk_inputs(model, tokenizer, protein, device)
            torch.save(entry, path)

    def get(self, protein: SCNProtein) -> dict:
        """Return cached trunk inputs for a protein."""
        sequence = _sequence(protein)
        if sequence not in self._entries:
            path = _cache_path(self.cache_dir, sequence)
            if not path.is_file():
                raise KeyError(f"No cached ESM embedding for sequence of length {len(sequence)}")
            self._entries[sequence] = _load_entry(path, sequence=sequence)
        return self._entries[sequence]

    def run_trunk_from_cache(
        self,
        model: EsmForProteinFolding,
        protein: SCNProtein,
        device: torch.device,
        *,
        num_recycles: int | None = None,
    ) -> EsmForProteinFoldingOutput:
        """Run the trunk from cached ESM embeddings for a protein."""
        entry = self.get(protein)
        s_s_0, s_z_0, aa, position_ids, attention_mask = _trunk_inputs_from_entry(entry, device)
        trunk_out = model.trunk(
            s_s_0,
            s_z_0,
            aa,
            position_ids,
            attention_mask,
            no_recycles=num_recycles,
        )
        return _esmfold_output_from_trunk(
            model,
            trunk_out,
            aa=aa,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
