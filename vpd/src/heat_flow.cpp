#include "heat_flow.hpp"

// def graph_laplacian(adjacency: np.ndarray, normalized: bool = True) -> csr_matrix:
//     """Compute graph Laplacian."""
//     if not isinstance(adjacency, csr_matrix):
//         adjacency = csr_matrix(adjacency)
    
//     n = adjacency.shape[0]
//     degrees = np.array(adjacency.sum(axis=1)).flatten()
    
//     if normalized:
//         deg_sqrt_inv = np.zeros(n)
//         deg_sqrt_inv[degrees > 0] = 1.0 / np.sqrt(degrees[degrees > 0])
//         D_sqrt_inv = csr_matrix((deg_sqrt_inv, (np.arange(n), np.arange(n))))
//         L = eye(n) - D_sqrt_inv @ adjacency @ D_sqrt_inv
//     else:
//         D = csr_matrix((degrees, (np.arange(n), np.arange(n))))
//         L = D - adjacency
    
//     return L

torch::Tensor graph_laplacian(torch::Tensor adjacency, bool normalized) {
    TORCH_CHECK(adjacency.dim() == 2, "adjacency must be a 2D tensor");
    TORCH_CHECK(
        adjacency.size(0) == adjacency.size(1),
        "adjacency must be square"
    );

    const auto n = adjacency.size(0);
    // look into turning into sparse matrix?
    const auto degrees = adjacency.sum(1);// dim=1

    if (normalized) {
        // Do stuff
        const auto deg_sqrt_inv = torch::where(
            degrees > 0,
            torch::rsqrt(degrees),
            torch::zeros_like(degrees)
        );
        // this should be equivalent to D_sqrt_inv @ adjacency @ D_sqrt_inv but I am not 100% sure
        const auto normalized_adj = adjacency * deg_sqrt_inv.unsqueeze(1) * deg_sqrt_inv.unsqueeze(0);
        return torch::eye(n, adjacency.options()) - normalized_adj;
    }

    return torch::diag_embed(degrees) - adjacency;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "graph_laplacian",
        &graph_laplacian,
        "Compute graph Laplacian",
        py::arg("adjacency"),
        py::arg("normalized") = true
    );
}
