#include <torch/extension.h>
#include <vector>

// def graph_laplacian(adjacency: np.ndarray, normalized: bool = True) -> csr_matrix:
// """Compute graph Laplacian."""
// if not isinstance(adjacency, csr_matrix):
//     adjacency = csr_matrix(adjacency)

// n = adjacency.shape[0]
// degrees = np.array(adjacency.sum(axis=1)).flatten()

// if normalized:
//     deg_sqrt_inv = np.zeros(n)
//     deg_sqrt_inv[degrees > 0] = 1.0 / np.sqrt(degrees[degrees > 0])
//     D_sqrt_inv = csr_matrix((deg_sqrt_inv, (np.arange(n), np.arange(n))))
//     L = eye(n) - D_sqrt_inv @ adjacency @ D_sqrt_inv
// else:
//     D = csr_matrix((degrees, (np.arange(n), np.arange(n))))
//     L = D - adjacency

// return L

torch::Tensor graph_laplacian(torch::Tensor adjacency, bool normalized) {

    return adjacency;
}