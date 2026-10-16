import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import ml_collections as mlc
import numpy as np
import torch
from minifold.utils.tensor_utils import tensor_tree_map
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold, train_test_split

from proteintda.config import CONFIG_OF, LOSS_CONFIG, RUN_CONFIG
from proteintda.minifold.loss import MiniFoldLoss
from proteintda.minifold.pipeline import build_lr_scheduler, evaluate_loader
from proteintda.minifold.runner import MiniFoldRunner
from proteintda.tda.persistence import pd_from_graph
from proteintda.utils.dataset import load_dataset, make_loader, set_seed
from tests.radius_test import build_kernels
from tests.test_utils import _resolve_device

OUT_DIR = Path(__file__).resolve().parents[2] / "cache" / "topology_trial"
WEIGHTS_PATH = OUT_DIR / "weights.json"
H0_KERNEL, H1_KERNEL = (0, 5, 300), (1, 2, 300)
WASSERSTEIN_RATIO, VPD_RATIO = 0.15, 0.10
CALIBRATION_PROTEINS = 8
SPLIT_SEED = 42
MAX_TRAIN_LENGTH = 350
SWEEP_SCALES, SWEEP_EPOCHS = (0.3, 1.0, 3.0), 8
MAIN_ARMS, MAIN_SEEDS, MAIN_EPOCHS = ("base", "topo", "wass", "vpd"), (42, 43, 44, 45, 46), 25

class TorchNormalizedVPD:
    def __init__(self, rff):
        self.rff = rff
        self.th = torch.tensor(np.asarray(rff.thetas).reshape(-1, rff.kernel.dim), dtype=torch.float64)

    def vpd_loss(self, pred_pd, target_pd):
        with torch.autocast(pred_pd.device.type, enabled=False):
            pred = self.rff.get_vpd(pred_pd.double()) / max(len(pred_pd), 1)
            target = self.rff.get_vpd(target_pd.double()) / max(len(target_pd), 1)
            if self.th.device != pred.device:
                self.th = self.th.to(pred.device)
            return 2 * (1 - torch.cos(self.th @ (pred - target))).mean()

def load_vpd_kernels():
    h0, _ = build_kernels(H0_KERNEL[0], resolutions=(H0_KERNEL[1],), t_sweep=(H0_KERNEL[2],))
    h1, _ = build_kernels(H1_KERNEL[0], resolutions=(H1_KERNEL[1],), t_sweep=(H1_KERNEL[2],))
    return TorchNormalizedVPD(h0[H0_KERNEL]), TorchNormalizedVPD(h1[H1_KERNEL])

def loss_config(arm, weights=None):
    cfg = mlc.ConfigDict(LOSS_CONFIG.to_dict())
    terms = cfg.tda.terms
    cfg.tda.weight = 1.0
    cfg.tda.enabled = arm != "base"
    for name in terms:
        terms[name].enabled = False
    if arm in ("wass", "topo"):
        for name in ("wasserstein_h0", "wasserstein_h1", "wasserstein_h2"):
            terms[name].enabled = True
    if arm in ("vpd", "topo"):
        for name in ("vpd_h0", "vpd_h1"):
            terms[name].enabled = True
    if weights is not None:
        for name, value in weights.items():
            terms[name].weight = value
    return cfg

def make_loss(arm, kernels, weights=None):
    h0, h1 = kernels
    return MiniFoldLoss(CONFIG_OF, loss_config=loss_config(arm, weights), h0rff=h0, h1rff=h1)

def make_runner():
    runtime, training = RUN_CONFIG.runtime, RUN_CONFIG.training
    return MiniFoldRunner(
        Path(runtime.minifold_cache_dir),
        model_size=runtime.model_size,
        device=_resolve_device(),
        train=True,
        unfreeze_fold_blocks=training.unfreeze_fold_blocks,
        unfreeze_structure_module=training.unfreeze_structure_module,
    )

def split_proteins(proteins):
    kf = KFold(n_splits=RUN_CONFIG.kfold.n_splits, shuffle=True, random_state=SPLIT_SEED)
    train_idx, test_idx = next(iter(kf.split(proteins)))
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=SPLIT_SEED)
    pick = lambda idx: [proteins[i] for i in idx]
    return pick(train_idx), pick(val_idx), pick(test_idx)

def _grad_vec(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([
        (g if g is not None else p.new_zeros(p.shape)).reshape(-1).float() for p, g in zip(params, grads)
    ])

def _cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))

def calibrate():
    training = RUN_CONFIG.training
    set_seed(training.seed)
    kernels = load_vpd_kernels()
    train_proteins, _, _ = split_proteins(load_dataset())
    runner = make_runner()
    loss = make_loss("topo", kernels)
    cfg = loss.loss_config
    wass_names = [n for n in cfg.tda.terms if n.startswith("wasserstein_")]
    params = [p for p in runner.model.parameters() if p.requires_grad]
    rng = np.random.default_rng(training.seed)
    sample = [train_proteins[i] for i in rng.choice(len(train_proteins), CALIBRATION_PROTEINS, replace=False)]

    rows = []
    for protein in sample:
        torch.cuda.synchronize()
        t0 = time.time()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=training.amp):
                batch = runner.prepare_batch(protein, train=True, crop=False)
                r_dict = runner.model(batch, num_recycling=1)
                batch_of = tensor_tree_map(lambda t: t[..., -1], batch["batch_of"])
                disto = loss._distogram_loss(
                    r_dict["preds"], batch["coords"], batch["mask"], runner.model.boundaries,
                    no_bins=r_dict["preds"].shape[-1])
                structure = loss.structure_loss(r_dict, batch_of)
                base = cfg.distogram.weight * disto + cfg.structure.weight * structure
                torch.cuda.synchronize()
                t_forward = time.time() - t0
                t0 = time.time()
                pred_adj, target_adj = loss._tda._create_adjs(r_dict, batch_of, 0)
                pred_diags = pd_from_graph(pred_adj)
                with torch.no_grad():
                    target_diags = pd_from_graph(target_adj)
                terms = loss._tda._term_losses(pred_diags, target_diags)
                t_tda = time.time() - t0
            wass = sum(LOSS_CONFIG.tda.terms[n].weight * terms[n] for n in wass_names)
            t0 = time.time()
            g = {"base": _grad_vec(base, params)}
            torch.cuda.synchronize()
            t_backward = time.time() - t0
            g["wass"] = _grad_vec(wass, params)
            g["vpd_h0"] = _grad_vec(terms["vpd_h0"], params)
            g["vpd_h1"] = _grad_vec(terms["vpd_h1"], params)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"n={len(str(protein.seq))} skipped: OOM", flush=True)
            continue
        row = {
            "length": len(str(protein.seq)),
            "t_forward": t_forward, "t_tda": t_tda, "t_backward": t_backward,
            "loss": {"base": float(base), "distogram": float(disto), "structure": float(structure),
                     **{n: float(v) for n, v in terms.items()}},
            "norm": {k: float(v.norm()) for k, v in g.items()},
            "cosine": {f"{a}|{b}": _cosine(g[a], g[b]) for a, b in (
                ("wass", "base"), ("vpd_h0", "base"), ("vpd_h1", "base"),
                ("vpd_h0", "wass"), ("vpd_h1", "wass"), ("vpd_h0", "vpd_h1"))},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del g, r_dict, terms
        torch.cuda.empty_cache()

    ratio = lambda k: float(np.median([r["norm"]["base"] / max(r["norm"][k], 1e-30) for r in rows]))
    wass_scale = WASSERSTEIN_RATIO * ratio("wass")
    weights = {n: wass_scale * LOSS_CONFIG.tda.terms[n].weight for n in wass_names}
    weights["vpd_h0"] = VPD_RATIO * ratio("vpd_h0")
    weights["vpd_h1"] = VPD_RATIO * ratio("vpd_h1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps({"weights": weights, "rows": rows}, indent=1))
    print("weights:", json.dumps(weights), flush=True)
    return 0

def per_protein_metrics(runner, proteins):
    rows = []
    for protein in proteins:
        totals, n = runner.run_batch(
            [protein], None, num_recycling=RUN_CONFIG.runtime.infer_recycles,
            backward=False, include_loss=False, include_metrics=True)
        rows.append({
            "id": str(protein.id), "length": len(str(protein.seq)),
            "tm": totals.get("tm_score") if n else None, "plddt": totals.get("plddt") if n else None,
        })
    return rows

def train_epoch(runner, loader, optimizer, loss_fn):
    training = RUN_CONFIG.training
    totals, n, skipped = defaultdict(float), 0, 0
    scaler = torch.amp.GradScaler("cuda", enabled=training.amp and runner.device.type == "cuda")
    for batch in loader:
        oom = False
        try:
            batch_totals, batch_n = runner.run_batch(
                batch, loss_fn, optimizer=optimizer, scaler=scaler,
                num_recycling=training.train_recycles or 0, randomize_recycles=training.randomize_recycles,
                use_amp=training.amp, grad_clip_norm=training.grad_clip_norm,
                backward=True, include_loss=True, include_metrics=False)
        except torch.OutOfMemoryError:
            oom = True
        if oom:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            skipped += 1
            continue
        if batch_n == 0:
            skipped += 1
        for key, value in batch_totals.items():
            totals[key] += value
        n += batch_n
    metrics = {key: value / n for key, value in totals.items()} if n else {}
    metrics["skipped"] = skipped
    return metrics

def _mean(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else float("nan")

def train(arm, seed, epochs, out_path, weight_scale=1.0):
    runtime, training = RUN_CONFIG.runtime, RUN_CONFIG.training
    torch.backends.cuda.matmul.allow_tf32 = True
    weights = {k: v * weight_scale for k, v in json.loads(WEIGHTS_PATH.read_text())["weights"].items()}
    kernels = load_vpd_kernels()
    train_proteins, val_proteins, test_proteins = split_proteins(load_dataset())
    train_proteins = [p for p in train_proteins if len(str(p.seq)) <= MAX_TRAIN_LENGTH]
    set_seed(seed)
    runner = make_runner()
    train_loss = make_loss(arm, kernels, weights)
    metric_loss = make_loss("topo", kernels, weights)
    result = {"arm": arm, "seed": seed, "epochs": epochs, "weight_scale": weight_scale, "weights": weights,
              "n_train": len(train_proteins), "n_val": len(val_proteins), "n_test": len(test_proteins),
              "history": []}
    result["test_before"] = per_protein_metrics(runner, test_proteins)
    print(f"test before: tm={_mean(result['test_before'], 'tm'):.4f}", flush=True)

    params = [p for p in runner.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=training.lr, weight_decay=training.weight_decay)
    scheduler = build_lr_scheduler(optimizer)
    rng = np.random.default_rng(seed)
    val_loader = make_loader(val_proteins, training.batch_size, shuffle=False)
    test_loader = make_loader(test_proteins, training.batch_size, shuffle=False)
    snapshot = lambda: {n: p.detach().cpu().clone() for n, p in runner.model.named_parameters() if p.requires_grad}
    best, best_val_tm = snapshot(), -1.0

    for epoch in range(epochs):
        t0 = time.time()
        train_loader = make_loader(train_proteins, training.batch_size, shuffle=True, rng=rng)
        train_metrics = train_epoch(runner, train_loader, optimizer, train_loss)
        val_metrics = evaluate_loader(runner, val_loader, metric_loss, num_recycling=runtime.infer_recycles)
        if val_metrics["tm_score"] > best_val_tm:
            best_val_tm, best = val_metrics["tm_score"], snapshot()
        if scheduler is not None:
            scheduler.step()
        entry = {"epoch": epoch + 1, "seconds": time.time() - t0, "train": train_metrics, "val": val_metrics}
        result["history"].append(entry)
        print(f"epoch {epoch + 1}/{epochs} {entry['seconds']:.0f}s skipped={train_metrics['skipped']} train_total={train_metrics.get('total', float('nan')):.4f} "
              f"val_tm={val_metrics['tm_score']:.4f} val: " +
              " ".join(f"{k}={v:.4g}" for k, v in sorted(val_metrics.items()) if k not in ("tm_score", "plddt")), flush=True)
        Path(out_path).write_text(json.dumps(result, indent=1))

    runner.model.load_state_dict(best, strict=False)
    result["best_val_tm"] = best_val_tm
    result["test_after"] = per_protein_metrics(runner, test_proteins)
    result["test_losses"] = evaluate_loader(runner, test_loader, metric_loss, num_recycling=runtime.infer_recycles)
    result["complete"] = True
    Path(out_path).write_text(json.dumps(result, indent=1))
    print(f"test after: tm={_mean(result['test_after'], 'tm'):.4f} (before {_mean(result['test_before'], 'tm'):.4f})", flush=True)
    return 0

def _load_complete(path):
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    return result if result.get("complete") else None

def _run_train(arm, seed, epochs, out, weight_scale):
    if _load_complete(out) is not None:
        return 0.0
    t0 = time.time()
    cmd = [sys.executable, "-u", __file__, "train", "--arm", arm, "--seed", str(seed), "--epochs", str(epochs),
           "--weight-scale", str(weight_scale), "--out", str(out)]
    with open(out.with_suffix(".log"), "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    seconds = time.time() - t0
    print(f"{out.name}: exit={code} in {seconds / 60:.1f} min", flush=True)
    return seconds

def summarize():
    runs = {}
    for seed in MAIN_SEEDS:
        for arm in MAIN_ARMS:
            result = _load_complete(OUT_DIR / f"{arm}_seed{seed}.json")
            if result is not None:
                runs[(arm, seed)] = result
    tm = lambda rows: {r["id"]: r["tm"] for r in rows if r["tm"] is not None}
    lines, summary = [], {"arms": {}, "paired": {}}
    for arm in MAIN_ARMS:
        seeds = [s for s in MAIN_SEEDS if (arm, s) in runs]
        if not seeds:
            continue
        after = [float(np.mean(list(tm(runs[(arm, s)]["test_after"]).values()))) for s in seeds]
        before = [float(np.mean(list(tm(runs[(arm, s)]["test_before"]).values()))) for s in seeds]
        summary["arms"][arm] = {"seeds": seeds, "test_tm_after": after, "test_tm_before": before,
                                "best_val_tm": [runs[(arm, s)]["best_val_tm"] for s in seeds],
                                "test_losses": [runs[(arm, s)]["test_losses"] for s in seeds]}
        lines.append(f"{arm:<5} seeds={seeds} test TM after={np.mean(after):.4f} (per seed {' '.join(f'{a:.4f}' for a in after)}) "
                     f"before={np.mean(before):.4f}")
    for arm in MAIN_ARMS[1:]:
        seeds = [s for s in MAIN_SEEDS if (arm, s) in runs and ("base", s) in runs]
        if not seeds:
            continue
        per_seed = []
        for s in seeds:
            a, b = tm(runs[(arm, s)]["test_after"]), tm(runs[("base", s)]["test_after"])
            per_seed.append({k: a[k] - b[k] for k in a if k in b})
        ids = sorted(set.intersection(*(set(d) for d in per_seed)))
        delta = np.array([np.mean([d[k] for d in per_seed]) for k in ids])
        rng = np.random.default_rng(0)
        boot = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(5000)]
        p = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else float("nan")
        summary["paired"][arm] = {"seeds": seeds, "n_proteins": len(ids), "mean_delta_tm": float(delta.mean()),
                                  "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                                  "wilcoxon_p": p, "n_better": int((delta > 0).sum()),
                                  "per_seed_mean_delta": [float(np.mean(list(d.values()))) for d in per_seed],
                                  "per_protein": dict(zip(ids, delta.tolist()))}
        lines.append(f"{arm:<5} vs base over seeds {seeds}: mean dTM={delta.mean():+.4f} "
                     f"CI95=[{np.quantile(boot, 0.025):+.4f}, {np.quantile(boot, 0.975):+.4f}] wilcoxon p={p:.3f} "
                     f"better on {(delta > 0).sum()}/{len(ids)} proteins")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

def overnight(hours):
    deadline = time.time() + hours * 3600
    best_scale, best_tm = 1.0, -1.0
    for scale in SWEEP_SCALES:
        out = OUT_DIR / f"sweep_scale{scale}.json"
        _run_train("topo", MAIN_SEEDS[0], SWEEP_EPOCHS, out, scale)
        result = _load_complete(out)
        if result is not None:
            print(f"sweep scale={scale}: best val TM={result['best_val_tm']:.4f}", flush=True)
            if result["best_val_tm"] > best_tm:
                best_scale, best_tm = scale, result["best_val_tm"]
    (OUT_DIR / "chosen_scale.json").write_text(json.dumps({"scale": best_scale, "best_val_tm": best_tm}))
    print(f"chosen weight scale: {best_scale}", flush=True)

    longest = 0.0
    for seed in MAIN_SEEDS:
        for arm in MAIN_ARMS:
            needed = longest * 1.15 if longest else 3600
            if time.time() + needed > deadline:
                print(f"stopping before {arm} seed {seed}: not enough time left", flush=True)
                summarize()
                (OUT_DIR / "DONE").write_text("deadline\n")
                return 0
            longest = max(longest, _run_train(arm, seed, MAIN_EPOCHS, OUT_DIR / f"{arm}_seed{seed}.json", best_scale))
            summarize()
    (OUT_DIR / "DONE").write_text("complete\n")
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("calibrate", "train", "overnight", "summarize"))
    parser.add_argument("--arm", choices=("base", "wass", "vpd", "topo"), default="base")
    parser.add_argument("--seed", type=int, default=RUN_CONFIG.training.seed)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", default=None)
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--hours", type=float, default=9.0)
    args = parser.parse_args()
    if args.mode == "calibrate":
        return calibrate()
    if args.mode == "overnight":
        return overnight(args.hours)
    if args.mode == "summarize":
        summarize()
        return 0
    out = args.out or str(OUT_DIR / f"{args.arm}_seed{args.seed}.json")
    return train(args.arm, args.seed, args.epochs, Path(out), args.weight_scale)

if __name__ == "__main__":
    sys.exit(main())
