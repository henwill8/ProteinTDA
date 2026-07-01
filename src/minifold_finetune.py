import random
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from tqdm import tqdm

from minifold.model.model import MiniFoldModel
from minifold.train.loss import AlphaFoldLoss
from minifold.utils.tensor_utils import tensor_tree_map

from data_conversions import (
    Atom37,
    SideChainAtom,
    atom_positions_from_atom37,
    distance_matrix,
    scnprotein_to_minifold_batch,
)
from loss import MiniFoldTDALoss
from minifold_predict import evaluate_minifold


def register_distogram_bins(
    model: MiniFoldModel,
    *,
    max_dist: float = 25.0,
    no_bins: int | None = None,
) -> None:
    """Register distogram bin boundaries on the model. Based on from minifold.train.model.MiniFold.__init__."""
    if no_bins is None:
        no_bins = model.fold.disto_bins
    boundaries = torch.linspace(2, max_dist, no_bins - 1)
    lower = torch.tensor([1.0])
    upper = torch.tensor([max_dist + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    mid_points = (exp_boundaries[:-1] + exp_boundaries[1:]) / 2
    model.register_buffer("boundaries", boundaries)
    model.register_buffer("mid_points", mid_points)


def configure_minifold_finetuning(
    model: MiniFoldModel,
) -> None:
    pass


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _sample_num_recycles(max_recycles: int) -> int:
    if max_recycles <= 0:
        return 0
    return random.randint(0, max_recycles + 1)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def prepare_minifold_batch(
    protein: SCNProtein,
    *,
    alphabet,
    config_of,
    device: torch.device,
) -> dict:
    batch = scnprotein_to_minifold_batch(
        protein,
        alphabet=alphabet,
        config_of=config_of,
        device=device,
    )
    batch = _move_to_device(batch, device)
    return {
        "seq": batch["seq"].unsqueeze(0),
        "coords": batch["coords"].unsqueeze(0),
        "mask": batch["mask"].unsqueeze(0),
        "batch_of": {key: value.unsqueeze(0) for key, value in batch["batch_of"].items()},
    }


def _distogram_loss(
    preds: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    boundaries: torch.Tensor,
    *,
    no_bins: int,
) -> torch.Tensor:
    """Distogram loss based on minifold.train.model.MiniFold.training_step."""
    coords = coords[:, :, 1, :]
    dists = torch.cdist(coords, coords)
    labels = F.one_hot((dists.unsqueeze(-1) > boundaries).sum(dim=-1), no_bins).to(preds)
    errors = -torch.sum(labels * F.log_softmax(preds, dim=-1), dim=-1)

    square_mask = mask[:, None] * mask[:, :, None]
    square_mask = square_mask * (1 - torch.eye(dists.shape[1], device=dists.device))[None]

    denom = 1e-5 + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)
    return torch.mean(mean)


def _add_tda_fields(
    r_dict: dict,
    batch_of: dict,
    *,
    tda_atom: SideChainAtom = SideChainAtom.CB,
) -> None:
    atom = Atom37.CB if tda_atom is SideChainAtom.CB else Atom37.CA
    pred_positions = r_dict["final_atom_positions"][0]
    pred_mask = r_dict["final_atom_mask"][0]
    target_positions = batch_of["all_atom_positions"][0, -1]
    target_mask = batch_of["all_atom_mask_true"][0, -1]

    pred_pts = atom_positions_from_atom37(pred_positions, pred_mask, atom)
    target_pts = atom_positions_from_atom37(target_positions, target_mask, atom)
    r_dict["adj"] = distance_matrix(pred_pts)
    batch_of["adj"] = distance_matrix(target_pts).detach()


def compute_losses(
    model: MiniFoldModel,
    batch: dict,
    *,
    structure_loss_fn: AlphaFoldLoss,
    tda_loss_fn: MiniFoldTDALoss | None,
    num_recycling: int,
    disto_weight: float,
    structure_weight: float,
    tda_weight: float,
) -> dict[str, torch.Tensor] | None:
    try:
        r_dict = model(batch, num_recycling=num_recycling)
    except torch.cuda.OutOfMemoryError:
        print("OOM running MiniFold forward pass; skipping protein.")
        return None

    preds = r_dict["preds"]
    disto_loss = _distogram_loss(
        preds,
        batch["coords"],
        batch["mask"],
        model.boundaries,
        no_bins=preds.shape[-1],
    )
    total = disto_weight * disto_loss
    losses: dict[str, torch.Tensor] = {"distogram": disto_loss.detach(), "total": total}

    if not model.use_structure_module:
        return losses

    batch_of = tensor_tree_map(lambda t: t[..., -1], batch["batch_of"])
    loss_of, of_breakdown = structure_loss_fn(r_dict, batch_of, _return_breakdown=True)
    total = total + structure_weight * loss_of
    losses["structure"] = loss_of.detach()
    for name, value in of_breakdown.items():
        if name != "loss":
            losses[f"of_{name}"] = value

    if tda_loss_fn is not None:
        _add_tda_fields(r_dict, batch_of)
        tda_loss, tda_breakdown = tda_loss_fn(r_dict, batch_of, _return_breakdown=True)
        total = total + tda_weight * tda_loss
        for name, value in tda_breakdown.items():
            losses[name] = value

    losses["total"] = total
    return losses


def train_one_epoch(
    model: MiniFoldModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    alphabet,
    config_of,
    structure_loss_fn: AlphaFoldLoss,
    tda_loss_fn: MiniFoldTDALoss | None,
    train_recycles: int | None = None,
    randomize_recycles: bool = True,
    use_amp: bool = False,
    grad_clip_norm: float | None = 1.0,
    disto_weight: float = 0.8,
    structure_weight: float = 0.2,
    tda_weight: float = 1.0,
) -> dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    totals = defaultdict(float)
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        for protein in batch:
            optimizer.zero_grad(set_to_none=True)
            if randomize_recycles and train_recycles is not None and train_recycles > 0:
                num_recycling = _sample_num_recycles(train_recycles)
            else:
                num_recycling = train_recycles or 0

            model_batch = prepare_minifold_batch(
                protein,
                alphabet=alphabet,
                config_of=config_of,
                device=device,
            )
            autocast_device = "cuda" if device.type == "cuda" else device.type
            with torch.autocast(autocast_device, dtype=torch.bfloat16, enabled=use_amp):
                losses = compute_losses(
                    model,
                    model_batch,
                    structure_loss_fn=structure_loss_fn,
                    tda_loss_fn=tda_loss_fn,
                    num_recycling=num_recycling,
                    disto_weight=disto_weight,
                    structure_weight=structure_weight,
                    tda_weight=tda_weight,
                )
            if losses is None:
                continue

            if scaler.is_enabled():
                scaler.scale(losses["total"]).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        grad_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        grad_clip_norm,
                    )
                optimizer.step()

            for key, value in losses.items():
                totals[key] += float(value.detach())
            n += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()

    if n == 0:
        return dict(totals)
    return {key: value / n for key, value in totals.items()}


def test_model(
    model: MiniFoldModel,
    loader,
    device: torch.device,
    *,
    alphabet,
    config_of,
    num_recycling: int = 3,
) -> tuple[float, float]:
    return evaluate_minifold(
        [protein for batch in loader for protein in batch],
        alphabet=alphabet,
        model=model,
        config_of=config_of,
        device=device,
        num_recycling=num_recycling,
    )


def build_optimizer(
    model: MiniFoldModel,
    *,
    base_lr: float,
    lm_lr: float,
    struct_lr: float,
) -> torch.optim.Optimizer:
    """Based on minifold.train.model.MiniFold.configure_optimizers."""
    return torch.optim.Adam(
        [
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if p.requires_grad
                    and ("lm" not in name)
                    and ("structure_module" not in name)
                    and ("aux_heads" not in name)
                    and ("sz_project" not in name)
                ],
                "lr": base_lr,
            },
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if p.requires_grad
                    and (
                        ("structure_module" in name)
                        or ("aux_heads" in name)
                        or ("sz_project" in name)
                    )
                ],
                "lr": struct_lr,
            },
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if p.requires_grad and ("lm" in name)
                ],
                "lr": lm_lr,
            },
        ],
        lr=base_lr,
    )
