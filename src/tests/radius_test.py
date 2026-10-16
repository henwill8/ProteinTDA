import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from vpd import _cpp

from minifold.utils.tensor_utils import tensor_tree_map

from proteintda.config import HEAT_RFF_CONFIG, LOSS_CONFIG, RUN_CONFIG, SamplingMethod
from proteintda.minifold.loss import _distance_matrix
from proteintda.minifold.runner import MiniFoldRunner
from proteintda.tda.persistence import pd_from_graph, wasserstein_distance
from proteintda.utils.conversions import Atom37, atom_positions_from_atom37
from tests.test_utils import _resolve_device, load_proteins, protein_problem

BURN_IN, THINNING = 3000, 30
S_RATIO = 1000
T_SWEEP = (1, 3, 10, 30, 100, 300, 1000, 3000)
RESOLUTIONS = {0: (5, 10, 15, 30), 1: (2, 4, 6, 10)}
MAX_DRIFT, MAX_AUTOCORR, MAX_ATTEMPTS = 0.5, 0.5, 6
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "heat_rff_sweep"

N_PROTEINS, N_REPS = 50, 5
NOISE_SIGMAS = (0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 2.0, 4.0)
INTERP_STEPS = (0.1, 0.25, 0.5, 0.75, 1.0)
ERROR_BAND = (0.63, 0.94)
MIN_AUC = 0.9

def build_mala_kernel(n, axis_dim, resolution, R, t, s, seed, burn_in, thinning, device):
    path = CACHE_DIR / (
        f"n-{n}_axisdim-{axis_dim}_res-{resolution}_s-{s}_t-{t}_R-{R}_seed-{seed}"
        f"_burn-{burn_in}_thin-{thinning}.pt"
    )
    if path.is_file():
        cached = torch.load(path, weights_only=False)
        kernel = _cpp.Heat_Kernel(n, axis_dim, resolution, R, s, t, cached["thetas"], cached["weights"])
        return _cpp.VPD(kernel)

    kernel = _cpp.Heat_Kernel(n, axis_dim, resolution, R, s, t)
    sampler = _cpp.MALASamplingKernel(sigma=0.1, burn_in=burn_in, thinning=thinning, tune_sigma=True)
    sampler.init(kernel, True, seed=seed,
        graph_representation_type=_cpp.Graph_Representation.LATTICE, device=device)
    sampler.build()
    rff = _cpp.VPD(kernel)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"thetas": rff.thetas, "weights": rff.weights}, path)
    return rff

def lattice_lambda(th, n, ppa):
    def c(d):
        return 1 - np.cos(d)
    R, dim = th.shape
    if n == 1:
        total = c(th[:, 0]) + 2 * c(th[:, 1:] - th[:, :-1]).sum(1)
        return total / (1 + 2 * (dim - 1))
    i = np.arange(dim)
    iy = ((1 + np.sqrt(1 + 8 * i)) / 2).astype(int)
    ix = i - iy * (iy - 1) // 2
    total, count = np.zeros(R), 0
    for mask, offset in (
        (ix > 0, -1),                 # left
        (iy + 1 < ppa, iy),           # up
        (ix + 1 < iy, 1),             # right
        (ix + 1 < iy, 1 - iy),        # down
    ):
        a = i[mask]
        b = a + (offset[mask] if isinstance(offset, np.ndarray) else offset)
        total += c(th[:, a] - th[:, b]).sum(1)
        count += len(a)
    diag = i[ix + 1 >= iy]            # nodes one step from the diagonal
    total += c(th[:, diag]).sum(1)
    return total / (count + len(diag))

def check_kernel(rff):
    kernel = rff.kernel
    th = np.asarray(rff.thetas).reshape(-1, kernel.dim)
    ppa = kernel.axis_dim * int(kernel.resolution) - 1
    lam = lattice_lambda(th, kernel.n, ppa)

    q = len(lam) // 4
    first, last = lam[:q], lam[-q:]
    spread = np.sqrt((first.var() + last.var()) / 2)
    autocorr = np.corrcoef(lam[:-1], lam[1:])[0, 1]
    return {
        "lam_mean": lam.mean(),
        "lam_std": lam.std(),
        "drift": (first.mean() - last.mean()) / spread,
        "autocorr": autocorr,
        "ess": len(lam) * (1 - autocorr) / (1 + autocorr),
        "R_lt_dim": len(lam) < kernel.dim,
    }

def build_kernels(hdim, resolutions=None, t_sweep=T_SWEEP):
    cfg = HEAT_RFF_CONFIG["h0rff" if hdim == 0 else "h1rff"]
    kernels, checks = {}, {}
    for res in resolutions or RESOLUTIONS[hdim]:
        for t in t_sweep:
            burn_in, thinning = BURN_IN, THINNING
            for _ in range(MAX_ATTEMPTS):
                rff = build_mala_kernel(cfg.n, cfg.axis_dim, res, cfg.R, t, (S_RATIO - 1) * t,
                                        cfg.seed, burn_in, thinning, cfg.device)
                check = check_kernel(rff)
                used = {"burn_in": burn_in, "thinning": thinning}
                bad_burn = abs(check["drift"]) > MAX_DRIFT
                bad_thin = check["autocorr"] > MAX_AUTOCORR
                if not (bad_burn or bad_thin):
                    break
                burn_in *= 3 if bad_burn else 1
                thinning *= 2 if bad_thin else 1
            key = (hdim, res, t)
            kernels[key] = rff
            checks[key] = {**check, **used, "ok": not (bad_burn or bad_thin)}
            print(f"h{hdim} res={res} t={t}: " + " ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in checks[key].items()), flush=True)
    return kernels, checks

class NormalizedVPD:
    def __init__(self, rff):
        self.rff = rff
        self.kernel = rff.kernel
        self.th = np.asarray(rff.thetas).reshape(-1, self.kernel.dim)

    def vpd_loss(self, pred_pd, target_pd):
        pred = self.rff.get_vpd(pred_pd).detach().cpu().numpy() / max(len(pred_pd), 1)
        target = self.rff.get_vpd(target_pd).detach().cpu().numpy() / max(len(target_pd), 1)
        return torch.tensor(2 * (1 - np.cos(self.th @ (pred - target))).mean())

def normalized_kernels(kernels):
    return {(*key, "norm"): NormalizedVPD(rff) for key, rff in kernels.items()}

def intrinsic_radius(rff, n_starts=10, seed=0):
    kernel = rff.kernel
    rng = np.random.default_rng(seed)
    deltas = np.geomspace(0.01 / kernel.resolution, kernel.axis_dim / 2, 40)
    radii = []
    for _ in range(n_starts):
        d = rng.uniform(1.0, kernel.axis_dim / 2 - 0.5)
        b = 0.0 if kernel.n == 1 else rng.uniform(0.0, d - 0.5)
        base = torch.tensor([[b, d]], dtype=torch.float64)
        loss = np.array([
            rff.vpd_loss(base, torch.tensor([[b, d + x]], dtype=torch.float64)).item() for x in deltas
        ])
        above = np.nonzero(loss >= 1.0)[0]
        if len(above) == 0:
            radii.append(np.inf)
        elif above[0] == 0:
            radii.append(deltas[0])
        else:
            j = above[0]
            radii.append(np.interp(1.0, loss[j - 1:j + 1], deltas[j - 1:j + 1]))
    return float(np.median(radii))

def perturb(pts, sigma, gen):
    return pts + sigma * torch.randn(pts.shape, generator=gen, dtype=pts.dtype)

def rmsd(a,b):
    return torch.sqrt(((a - b) ** 2).sum(-1).mean()).item()

def lddt(pred_adj, target_adj, cutoff=15.0):
    n = target_adj.shape[0]
    mask = (target_adj < cutoff) & ~torch.eye(n, dtype=torch.bool, device=target_adj.device)
    if not mask.any():
        return float("nan")
    err = (pred_adj - target_adj).abs()[mask]
    return torch.stack([(err < thr).float().mean() for thr in (0.5, 1.0, 2.0, 4.0)]).mean().item()

def run_ladder(proteins, kernels):
    gen = torch.Generator().manual_seed(0)
    hom_dim = LOSS_CONFIG.tda.pd.hom_dim
    records = []
    for pi, protein in enumerate(proteins):
        target_pts, target_adj = protein_problem(protein, torch.device("cpu"))
        target_pds = pd_from_graph(target_adj)
        for sigma in NOISE_SIGMAS:
            for _ in range(N_REPS):
                pred_pts = perturb(target_pts, sigma, gen)
                pred_adj = _distance_matrix(pred_pts)
                pred_pds = pd_from_graph(pred_adj)
                wass = wasserstein_distance(pred_pds, target_pds, hom_dim)
                records.append({
                    "protein": pi,
                    "level": sigma,
                    "rmsd": rmsd(pred_pts, target_pts),
                    "lddt": lddt(pred_adj, target_adj),
                    "wass": [float(w) for w in wass],
                    "vpd": {key: rff.vpd_loss(pred_pds[key[0]], target_pds[key[0]]).item()
                            for key, rff in kernels.items()},
                })
        print(f"protein {pi + 1}/{len(proteins)} n={len(target_pts)}", flush=True)
    return records

def kabsch_align(pred, target):
    pred_centre, target_centre = pred.mean(0), target.mean(0)
    P, Q = pred - pred_centre, target - target_centre
    U, _, Vt = torch.linalg.svd(P.T @ Q)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=pred.dtype))
    return P @ (Vt.T @ D @ U.T).T + target_centre

def predict_cb(runner, protein):
    batch = runner.prepare_batch(protein, train=True, crop=False)
    batch_of = tensor_tree_map(lambda t: t[..., -1], batch["batch_of"])
    with torch.no_grad():
        out = runner.model(batch, num_recycling=RUN_CONFIG.runtime.infer_recycles)
    pred = atom_positions_from_atom37(
        out["final_atom_positions"][0], out["final_atom_mask"][0], Atom37.CB).double().cpu()
    target = atom_positions_from_atom37(
        batch_of["all_atom_positions"][0], batch_of["all_atom_mask_true"][0], Atom37.CB).double().cpu()
    return pred, target

def run_interpolation_ladder(structures, kernels):
    hom_dim = LOSS_CONFIG.tda.pd.hom_dim
    records = []
    for pi, (pred_pts, target_pts) in enumerate(structures):
        aligned = kabsch_align(pred_pts, target_pts)
        target_adj = _distance_matrix(target_pts)
        target_pds = pd_from_graph(target_adj)
        for step in INTERP_STEPS:
            pts = target_pts + step * (aligned - target_pts)
            adj = _distance_matrix(pts)
            pds = pd_from_graph(adj)
            wass = wasserstein_distance(pds, target_pds, hom_dim)
            records.append({
                "protein": pi,
                "level": step,
                "rmsd": rmsd(pts, target_pts),
                "lddt": lddt(adj, target_adj),
                "wass": [float(w) for w in wass],
                "vpd": {key: rff.vpd_loss(pds[key[0]], target_pds[key[0]]).item()
                        for key, rff in kernels.items()},
            })
        print(f"protein {pi + 1}/{len(structures)} n={len(target_pts)}", flush=True)
    return records

def _adjacent_auc(loss, level, levels):
    aucs = []
    for lo, hi in zip(levels[:-1], levels[1:]):
        a, b = loss[level == lo][:, None], loss[level == hi][None, :]
        aucs.append(((b > a) + 0.5 * (b == a)).mean())
    return aucs

def score(records, key, levels=NOISE_SIGMAS, band=ERROR_BAND):
    hdim = key[0]
    prot = np.array([r["protein"] for r in records])
    level = np.array([r["level"] for r in records])
    rmsd_ = np.array([r["rmsd"] for r in records])
    lddt_ = np.array([r["lddt"] for r in records])
    wass = np.array([r["wass"][hdim] for r in records])
    loss = np.array([r["vpd"][key] for r in records])

    rho_rmsd, rho_wass, aucs = [], [], []
    for p in np.unique(prot):
        m = prot == p
        aucs.append(_adjacent_auc(loss[m], level[m], levels))
        if np.ptp(loss[m]) > 0:
            rho_rmsd.append(spearmanr(loss[m], rmsd_[m]).correlation)
            if np.ptp(wass[m]) > 0:
                rho_wass.append(spearmanr(loss[m], wass[m]).correlation)
    auc = np.mean(aucs, axis=0)

    if band is None:
        in_band = np.ones(len(levels) - 1, dtype=bool)
        near = np.ones(len(loss), dtype=bool)
    else:
        level_lddt = np.array([np.median(lddt_[level == s]) for s in levels])
        in_band = (level_lddt[:-1] > band[0]) & (level_lddt[1:] < band[1])
        near = (lddt_ > band[0]) & (lddt_ < band[1])
    return {
        "rho_rmsd": float(np.median(rho_rmsd)) if rho_rmsd else float("nan"),
        "rho_wass": float(np.median(rho_wass)) if rho_wass else float("nan"),
        "auc": [float(a) for a in auc],
        "coverage": float((auc[in_band] > MIN_AUC).mean()) if in_band.any() else float("nan"),
        "mean_loss": [float(loss[level == s].mean()) for s in levels],
        "scale": float(np.median(loss[near])) if near.any() else float("nan"),
    }

def main():
    return 0

if __name__ == "__main__":
    sys.exit(main())
