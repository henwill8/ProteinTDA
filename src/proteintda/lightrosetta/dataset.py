from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data

from proteintda.lightrosetta.path import ensure_lightrosetta_on_path

_MSA_ALPHABET = "ARNDCQEGHILKMFPSTWYV"


@dataclass
class OfficialSample:
    id: str
    seq: str
    data: Data

    @property
    def mask(self) -> str:
        return "+" * len(self.seq)


def _msa_to_seq(msa: torch.Tensor) -> str:
    tokens = msa.reshape(-1)[: msa.shape[-1]].detach().cpu().tolist()
    return "".join(_MSA_ALPHABET[t] if 0 <= int(t) < 20 else "X" for t in tokens)


def load_official_dataset(root: str | Path, *, max_proteins: int | None = None) -> list[OfficialSample]:
    root = Path(root).expanduser().resolve()
    if not (root / "raw").is_dir() or not (root / "processed").is_dir():
        raise FileNotFoundError(
            f"Official LightRoseTTA data expects {{raw,processed}} under {root}"
        )

    ensure_lightrosetta_on_path()

    from data_pipeline import Protein_Dataset

    dataset = Protein_Dataset(str(root), test_mode=False)
    n = len(dataset)
    if max_proteins is not None:
        n = min(n, int(max_proteins))

    samples: list[OfficialSample] = []
    for i in range(n):
        data = dataset.get(i)
        name = getattr(data, "protein_name", f"data_{i}")
        if isinstance(name, (list, tuple)):
            name = name[0]
        seq = _msa_to_seq(data.msa)
        if getattr(data, "ca_coords", None) is None and getattr(data, "CA_atom_index", None) is not None:
            data.ca_coords = torch.index_select(data.pos, 0, data.CA_atom_index.long())
        data.seq = seq
        samples.append(OfficialSample(id=str(name), seq=seq, data=data))

    print(f"Loaded {len(samples)} official LightRoseTTA proteins from {root}")
    return samples
