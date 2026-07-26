import numpy as np
import sys

from proteintda.config import HEAT_RFF_CONFIG, LOSS_CONFIG 

from proteintda.tda.vpd_kernels import create_heat_random_fourier_features
from tests.binning_metrics import compute_metrics
from tests.test_utils import load_proteins, print_results, protein_pds, _resolve_device, save_results 

HOM_DIM = LOSS_CONFIG["pd"]["hom_dim"]

AXISDIM_SWEEP = (8, 10, 12, 14, 16)
RESOLUTION_SWEEP = (1, 2, 3, 4, 5)

def _finite(dgm) -> np.ndarray:
    arr = dgm.detach().cpu().numpy() if hasattr(dgm, "detach") else np.asarray(dgm)
    arr = np.asarray(arr, dtype=float).reshape(-1, 2)
    return arr[np.isfinite(arr[:, 1])]


def _by_dim(pds, dim) -> tuple[np.ndarray, np.ndarray]:
    per_protein = [_finite(pd[dim]) if len(pd) > dim else np.zeros((0, 2)) for pd in pds]
    counts = np.array([len(a) for a in per_protein])
    nonempty = [a for a in per_protein if len(a)]
    pooled = np.vstack(nonempty) if nonempty else np.zeros((0, 2))
    return pooled, counts

def _dim_summary(pds) -> tuple[dict, dict]:
    pooled_pds_by_dim = {}
    results={}
    for dim in range(HOM_DIM):
        pooled, counts = _by_dim(pds, dim)
        pooled_pds_by_dim[dim] = pooled
        if not len(pooled):
            print(f"{dim:4d} {'none':>12s}")
            continue
        q1, q3 = np.percentile(pooled[:,1], [25,75])
        lives = pooled[:,1] - pooled[:,0]
        results[f"Dim: {dim} Mean Features Per Protein"] = np.mean(counts)
        results[f"Dim: {dim} Birth Range"] = pooled[:,0].max() - pooled[:,0].min()
        results[f"Dim: {dim} Death Range"] = pooled[:,1].max() - pooled[:,1].min()
        results[f"Dim: {dim} Death IQR"] = q3 - q1
        results[f"Dim: {dim} Median Life"] = np.median(lives)
        results[f"Dim: {dim} Latest Death"] = pooled[:,1].max()
    return pooled_pds_by_dim, results

def _grid_metrics(informative, pds) -> dict:
    results = {}
    for axis_dim in AXISDIM_SWEEP:
        for resolution in RESOLUTION_SWEEP:
            heat_config = HEAT_RFF_CONFIG
            for dim in range(HOM_DIM):
                pooled = informative[dim]
                heat_config[f"h{dim}rff"]["axis_dim"] = axis_dim
                heat_config[f"h{dim}rff"]["resolution"] = resolution 
                rff = create_heat_random_fourier_features(**heat_config[f"h{dim}rff"])
                results["Dim: {dim}, Axis Dim: {axis_dim}, Resolution: {resolution} n"] = rff.kernel.dim
                binning_metrics = [compute_metrics(pd[dim], rff) for pd in pds] 
                total_nonzero = [metric["total_nonzero"] for metric in binning_metrics]
                total_mult= [metric["mean_nonzero"] for metric in binning_metrics]
                mean_nonzero = np.mean(total_nonzero)
                used = float(mean_nonzero) / rff.kernel.dim
                results["Dim: {dim}, Axis Dim: {axis_dim}, Resolution: {resolution} mean_nonzero"] = mean_nonzero
                results["Dim: {dim}, Axis Dim: {axis_dim}, Resolution: {resolution} used"] = used 
                results["Dim: {dim}, Axis Dim: {axis_dim}, Resolution: {resolution} mean_mult"] = np.mean(total_mult)
                inside = pooled[pooled[:, 1] <= axis_dim]
                lost_from_axis_dim = 100.0 * (1.0 - len(inside) / len(informative))
                results["Dim: {dim}, Axis Dim: {axis_dim}, Resolution: {resolution} lost_from_axis_dim"] = lost_from_axis_dim 
    return results
            

def protein_homology_metrics(n_proteins : int | None, results, device) -> None:
    if n_proteins is None:
        n_proteins = 50

    proteins = load_proteins(n_proteins)
    print(f"Computing metrics for {len(proteins)} proteins")

    pds = protein_pds(proteins)

    informative, results["Dim Summary"] = _dim_summary(pds)

    results["Grid Metrics"] = _grid_metrics(informative, pds)

def main(write = False):
    device = _resolve_device()
    results = {}
    protein_homology_metrics(n_proteins=50, results=results, device=device)
    if write:
        save_results(results)
    else:
        print_results(results)
    return 0

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        if sys.argv[1] == "--write":
            sys.exit(main(write=True))
    sys.exit(main(write=False))
