"""Functions based on minifold/predict.py"""

import urllib.request
from pathlib import Path

import sidechainnet as scn
import numpy as np
import torch
import torch.nn.functional as F
from esm.pretrained import load_model_and_alphabet
from sidechainnet.dataloaders.SCNProtein import SCNProtein

from minifold.data.config import model_config
from minifold.data.of_data import of_inference
from minifold.model.model import MiniFoldModel
from minifold.utils.residue_constants import (
    atom_order,
    restype_order_with_x,
    restype_order_with_x_inverse,
)

MODEL_URLS = {
    "48L": "https://huggingface.co/jwohlwend/minifold/resolve/main/minifold_48L_final.ckpt",
    "12L": "https://huggingface.co/jwohlwend/minifold/resolve/main/minifold_12L_final.ckpt",
}


def download_checkpoint(cache_dir: Path, model_size: str = "48L") -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cache_dir / f"minifold_{model_size}.ckpt"
    if not checkpoint.exists():
        print(f"Downloading MiniFold {model_size} weights to {checkpoint}...")
        urllib.request.urlretrieve(MODEL_URLS[model_size], checkpoint)
    return checkpoint


def load_minifold(
    cache_dir: Path,
    *,
    model_size: str = "48L",
    device: torch.device | None = None,
    kernels: bool = False,
    compile: bool = False,
    train: bool = False,
) -> tuple[object, MiniFoldModel, object]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = download_checkpoint(cache_dir, model_size)
    torch.hub.set_dir(cache_dir)

    if kernels:
        torch._dynamo.config.cache_size_limit = 64

    ckpt = torch.load(checkpoint, map_location="cpu")
    hparams = ckpt["hyper_parameters"]
    # TODO: need to look into the config options more, there are a few good memory saving options it seems
    config_of = model_config(
        "initial_training" if train else "finetuning",
        train=train,
        low_prec=False,
        long_sequence_inference=False,
    )
    model = MiniFoldModel(
        esm_model_name=hparams["esm_model_name"],
        num_blocks=hparams["num_blocks"],
        no_bins=hparams["no_bins"],
        config_of=config_of,
        use_structure_module=True, # Note: They only used structure module in second stage
        kernels=kernels,
    )
    _, alphabet = load_model_and_alphabet(hparams["esm_model_name"])

    state_dict = ckpt["state_dict"]
    state_dict = {k: v for k, v in state_dict.items() if "boundaries" not in k}
    state_dict = {k: v for k, v in state_dict.items() if "mid_points" not in k}
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    if compile:
        model.fold.miniformer = torch.compile(
            model.fold.miniformer,
            dynamic=True,
            fullgraph=True,
        )
    
    model = model.to(device)
    if not train:
        model.eval()

    return alphabet, model, config_of


def evaluate_minifold(
    proteins: list[SCNProtein],
    *,
    alphabet,
    model: MiniFoldModel,
    config_of,
    device: torch.device,
    num_recycling: int = 3,
) -> tuple[float, float]:
    from tmtools import tm_align
    from tqdm import tqdm

    from data_conversions import SideChainAtom, atom_positions_from_sidechainnet

    ca_idx = atom_order["CA"]
    plddt_scores: list[float] = []
    tm_scores: list[float] = []

    for protein in tqdm(proteins, desc="eval", leave=False):
        result = predict_scnprotein(
            protein,
            alphabet=alphabet,
            model=model,
            config_of=config_of,
            device=device,
            num_recycling=num_recycling,
        )
        plddt_scores.append(float(result["plddt"].mean()))
        pred_ca = result["positions"][:, ca_idx].numpy()
        exp_ca = atom_positions_from_sidechainnet(protein, SideChainAtom.CA).numpy()
        alignment = tm_align(pred_ca, exp_ca, str(protein.seq), str(protein.seq))
        tm_scores.append(alignment.tm_norm_chain2)

    return float(np.mean(plddt_scores)), float(np.mean(tm_scores))


def _prepare_input(seq: str, config_of, alphabet, train: bool = False):
    open_fold_batch = of_inference(seq, "predict" if not train else "train", config_of.data)
    of_seq = "".join(
        restype_order_with_x_inverse[x.item()] for x in open_fold_batch["aatype"]
    )[: open_fold_batch["seq_length"]]
    encoded_seq = torch.tensor(alphabet.encode(of_seq), dtype=torch.long)
    mask = open_fold_batch["seq_mask"][:, 0].bool()
    relevant = {"aatype", "seq_mask", "residx_atom37_to_atom14", "atom37_atom_exists"}
    open_fold_batch = {k: v for k, v in open_fold_batch.items() if k in relevant}
    return encoded_seq, mask, open_fold_batch


@torch.inference_mode()
def predict_scnprotein(
    protein: SCNProtein,
    *,
    alphabet,
    model: MiniFoldModel,
    config_of,
    device: torch.device,
    num_recycling: int = 3,
) -> dict[str, torch.Tensor]:
    seq = str(protein.seq)
    encoded_seq, mask, batch_of = _prepare_input(seq, config_of, alphabet)

    model_batch = {
        "seq": encoded_seq.unsqueeze(0).to(device),
        "mask": mask.unsqueeze(0).to(device),
        "batch_of": {k: v.unsqueeze(0).to(device) for k, v in batch_of.items()},
    }

    autocast_device = "cuda" if device.type == "cuda" else device.type
    with torch.autocast(autocast_device, dtype=torch.bfloat16):
        out = model(model_batch, num_recycling=num_recycling)

    length = len(seq)
    return {
        "positions": out["final_atom_positions"][0, :length].float().cpu(),
        "atom_mask": out["final_atom_mask"][0, :length].float().cpu(),
        "plddt": out["plddt"][0, :length].float().cpu(),
        "sequence": seq,
    }
