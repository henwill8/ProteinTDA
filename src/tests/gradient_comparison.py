import math
import sys
from pathlib import Path

import numpy as np
import torch
from minifold.utils.tensor_utils import tensor_tree_map

from proteintda.config import LOSS_CONFIG, RUN_CONFIG
from proteintda.minifold.loss import _as_tensor
from proteintda.minifold.pipeline import build_loss_fn
from proteintda.minifold.runner import MiniFoldRunner
from proteintda.tda.persistence import pd_from_graph
from tests.mlp_test import _log_loss_config
from tests.test_utils import (
    _resolve_device,
    _scalar,
    load_proteins,
    print_results,
    save_results,
)

N_PROTEINS = 10
PAIRS = [
    ("wasserstein", "vpd"),
    ("tda", "structure"),
    ("wasserstein", "structure"),
    ("vpd", "structure"),
]


def _weighted_sum(names, terms):
    return sum(LOSS_CONFIG.tda.terms[name].weight * terms[name] for name in names)


def _named_losses(tda, r_dict, batch_of, structure_loss_fn):
    pred_adj, target_adj = tda._create_adjs(r_dict, batch_of, 0)
    pred_diags = pd_from_graph(pred_adj)
    with torch.no_grad():
        target_diags = pd_from_graph(target_adj)
    terms = {}
    for name, loss in tda._term_losses(pred_diags, target_diags).items():
        loss = _as_tensor(loss, pred_adj)
        if torch.isfinite(loss).all() and loss.requires_grad:
            terms[name] = loss
    w_names = [name for name in terms if name.startswith("wasserstein_")]
    v_names = [name for name in terms if name.startswith("vpd_")]
    if w_names:
        terms["wasserstein"] = _weighted_sum(w_names, terms)
    if v_names:
        terms["vpd"] = _weighted_sum(v_names, terms)
    structure = structure_loss_fn(r_dict, batch_of)
    if torch.isfinite(structure).all() and structure.requires_grad:
        terms["structure"] = structure
    return terms


def _grad_vec(loss, params, retain_graph):
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    parts = []
    for param, grad in zip(params, grads):
        if grad is None:
            parts.append(param.new_zeros(param.numel()))
        else:
            parts.append(grad.reshape(-1))
    vec = torch.cat(parts).float()
    if not torch.isfinite(vec).all():
        return None
    return vec


def _direction_stats(a, b):
    a_norm = torch.linalg.vector_norm(a)
    b_norm = torch.linalg.vector_norm(b)
    if a_norm < LOSS_CONFIG.eps or b_norm < LOSS_CONFIG.eps:
        return None
    dot = torch.dot(a, b)
    cosine = torch.clamp(dot / (a_norm * b_norm), -1.0, 1.0)
    a_proj = (dot / torch.dot(b, b)) * b
    orth = torch.linalg.vector_norm(a - a_proj)
    return {
        "cosine": _scalar(cosine),
        "align_ratio": _scalar(torch.linalg.vector_norm(a_proj) / orth.clamp_min(LOSS_CONFIG.eps)),
    }


def _loss_grads(loss_fn, r_dict, batch_of, params):
    losses = _named_losses(loss_fn._tda, r_dict, batch_of, loss_fn.structure_loss)
    names = [name for name in ("wasserstein", "vpd", "structure") if name in losses]
    grads = {}
    for i, name in enumerate(names):
        vec = _grad_vec(losses[name], params, retain_graph=i < len(names) - 1)
        if vec is not None:
            grads[name] = vec
    if "wasserstein" in grads and "vpd" in grads:
        grads["tda"] = grads["wasserstein"] + grads["vpd"]
    elif "wasserstein" in grads:
        grads["tda"] = grads["wasserstein"]
    elif "vpd" in grads:
        grads["tda"] = grads["vpd"]
    return grads


def _summarize(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


def run_comparison(loss_fn, runner, proteins, num_recycling):
    params = [p for p in runner.model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable MiniFold parameters.")
    buckets = {
        f"{left} vs {right} {metric}": []
        for left, right in PAIRS
        for metric in ("cosine", "align_ratio")
    }
    for i, protein in enumerate(proteins):
        print(f"  protein {i + 1}/{len(proteins)}  n={len(protein.seq)}", flush=True)
        model_batch = runner.prepare_batch(protein, train=True, crop=False)
        try:
            r_dict = runner.model(model_batch, num_recycling=num_recycling)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("    skip: OOM", flush=True)
            continue
        if "sm" not in r_dict:
            print("    skip: no structure module outputs", flush=True)
            continue
        batch_of = tensor_tree_map(lambda t: t[..., -1], model_batch["batch_of"])
        grads = _loss_grads(loss_fn, r_dict, batch_of, params)
        for left, right in PAIRS:
            if left not in grads or right not in grads:
                continue
            stats = _direction_stats(grads[left], grads[right])
            if stats is None:
                continue
            buckets[f"{left} vs {right} cosine"].append(stats["cosine"])
            buckets[f"{left} vs {right} align_ratio"].append(stats["align_ratio"])
    return {name: _summarize(values) for name, values in buckets.items()}


def main(write):
    device = _resolve_device()
    loss_fn = build_loss_fn()
    if not loss_fn.tda_enabled:
        raise ValueError(
            "No TDA loss terms enabled. Enable wasserstein and/or vpd in LOSS_CONFIG."
        )
    if not LOSS_CONFIG.structure.enabled:
        raise ValueError("Structure loss is disabled in LOSS_CONFIG.")
    _log_loss_config()
    runtime = RUN_CONFIG.runtime
    training = RUN_CONFIG.training
    runner = MiniFoldRunner(
        Path(runtime.minifold_cache_dir),
        model_size=runtime.model_size,
        device=device,
        train=True,
        unfreeze_fold_blocks=training.unfreeze_fold_blocks,
        unfreeze_structure_module=training.unfreeze_structure_module,
    )
    runner.model.eval()
    trainable, total = runner.trainable_parameter_count
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    proteins = load_proteins(N_PROTEINS)
    print(f"Comparing MiniFold gradients on {len(proteins)} proteins")
    results = {
        "MiniFold": run_comparison(
            loss_fn, runner, proteins, runtime.infer_recycles
        )
    }
    if write:
        save_results(results)
    else:
        print_results(results)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--write":
            sys.exit(main(write=True))
    sys.exit(main(write=False))
