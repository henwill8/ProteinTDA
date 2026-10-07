import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data

from proteintda.lightrosetta.path import ensure_lightrosetta_on_path

_MSA_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
_PT_NAME = re.compile(r"^data_(\d+)\.pt$")


@dataclass
class OfficialSample:
    id: str
    seq: str
    path: Path

    @property
    def mask(self) -> str:
        return "+" * len(self.seq)

    def load(self) -> Data:
        ensure_lightrosetta_on_path()
        data = torch.load(self.path, map_location="cpu", weights_only=False)
        if getattr(data, "ca_coords", None) is None and getattr(
            data, "CA_atom_index", None
        ) is not None:
            data.ca_coords = torch.index_select(data.pos, 0, data.CA_atom_index.long())
        data.seq = self.seq
        return data


def _msa_to_seq(msa: torch.Tensor) -> str:
    tokens = msa.reshape(-1)[: msa.shape[-1]].detach().cpu().tolist()
    return "".join(_MSA_ALPHABET[t] if 0 <= int(t) < 20 else "X" for t in tokens)


def _list_processed_files(processed: Path) -> list[Path]:
    indexed: list[tuple[int, Path]] = []
    for path in processed.iterdir():
        match = _PT_NAME.match(path.name)
        if match and path.is_file():
            indexed.append((int(match.group(1)), path))
    indexed.sort(key=lambda item: item[0])
    return [path for _, path in indexed]


def _index_cache_path(root: Path, n_files: int, stamp: int) -> Path:
    key = f"{root.resolve().as_posix()}::{n_files}::{stamp}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return Path("cache") / "lightrosetta_index" / f"{digest}.json"


def _load_index_cache(cache_path: Path, files: list[Path]) -> list[dict] | None:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != len(files):
        return None
    for entry, path in zip(entries, files):
        if entry.get("file") != path.name:
            return None
        if "id" not in entry or "seq" not in entry:
            return None
    return entries


def _save_index_cache(cache_path: Path, entries: list[dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump({"entries": entries}, handle)
    tmp.replace(cache_path)


def _build_entries(files: list[Path]) -> list[dict]:
    ensure_lightrosetta_on_path()
    entries: list[dict] = []
    total = len(files)
    for i, path in enumerate(files):
        data = torch.load(path, map_location="cpu", weights_only=False)
        name = getattr(data, "protein_name", path.stem)
        if isinstance(name, (list, tuple)):
            name = name[0]
        seq = _msa_to_seq(data.msa)
        entries.append({"file": path.name, "id": str(name), "seq": seq})
        del data
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"Indexed LightRoseTTA metadata {i + 1}/{total}", flush=True)
    return entries


def load_official_dataset(
    root: str | Path, *, max_proteins: int | None = None
) -> list[OfficialSample]:
    root = Path(root).expanduser().resolve()
    processed = root / "processed"
    if not (root / "raw").is_dir() or not processed.is_dir():
        raise FileNotFoundError(
            f"Official LightRoseTTA data expects {{raw,processed}} under {root}"
        )

    files = _list_processed_files(processed)
    if not files:
        raise FileNotFoundError(f"No data_*.pt files under {processed}")

    stamp = int(processed.stat().st_mtime_ns)
    cache_path = _index_cache_path(root, len(files), stamp)
    entries = _load_index_cache(cache_path, files)
    if entries is None:
        entries = _build_entries(files)
        _save_index_cache(cache_path, entries)
        print(f"Wrote LightRoseTTA index cache to {cache_path}")
    else:
        print(f"Loaded LightRoseTTA index cache from {cache_path}")

    if max_proteins is not None:
        files = files[: int(max_proteins)]
        entries = entries[: int(max_proteins)]

    samples = [
        OfficialSample(id=entry["id"], seq=entry["seq"], path=path)
        for entry, path in zip(entries, files)
    ]
    print(f"Indexed {len(samples)} official LightRoseTTA proteins from {root}")
    return samples
