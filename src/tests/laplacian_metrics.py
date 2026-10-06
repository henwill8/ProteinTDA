import sys

import numpy as np

from proteintda.config import GraphRepresentation, HEAT_RFF_CONFIG
from proteintda.tda.vpd_kernels import create_heat_random_fourier_features 
from tests.test_utils import print_results, save_results

KERNEL_COUNT = 3

def _i_laplacian_metrics(kernel, results) -> None:
    lam = np.asarray(kernel.weights, dtype=float).ravel()
    results["lam_max"] = float(max(lam))
    results["lam_min"] = float(min(lam))
    results["lam_mean"] = float(np.mean(lam))
    results["lam_std"] = float(np.std(lam))

def laplacian_metrics(kernels, results) -> None:
    for i in range(len(kernels)):
        results[f"h{i}"] = {}
        _i_laplacian_metrics(kernels[i], results[f"h{i}"])

def sweep_kernel_types(results) -> None:
    for graph_representation in GraphRepresentation:
            kernels = []
            for i in range(KERNEL_COUNT):
                HEAT_RFF_CONFIG[f"h{i}rff"]["graph_representation_type"] = graph_representation
                kernels.append(create_heat_random_fourier_features(**HEAT_RFF_CONFIG[f"h{i}rff"]))
            results[f"{graph_representation.name}"] = {}
            laplacian_metrics(kernels, results[f"{graph_representation.name}"])


def main(write):
    results={}
    sweep_kernel_types(results)
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
