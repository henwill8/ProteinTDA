#include "heat_flow.hpp"

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

torch::Tensor heat_kernel(torch::Tensor L, double tau) {
    TORCH_CHECK(L.dim() == 2, "L must be a 2D tensor");
    TORCH_CHECK(L.size(0) == L.size(1), "L must be square");

    const auto n = L.size(0);
    const auto I = torch::eye(n, L.options());

    if (n < 1000)
        return torch::linalg_matrix_exp(-tau * L);

    constexpr int k = 20;
    const auto step = I - (tau / static_cast<double>(k)) * L;
    auto H = step.clone();
    for (int i = 0; i < k - 1; ++i) {
        H = H.matmul(step);
    }
    return H;
}

torch::Tensor heat_edge_weights(
    torch::Tensor adjacency,
    double tau,
    bool normalized
) {
    const auto L = graph_laplacian(adjacency, normalized);
    const auto H = heat_kernel(L, tau);
    return H * (adjacency > 0);
}

torch::Tensor heat_vertex_function(
    torch::Tensor adjacency,
    double tau,
    const std::optional<torch::Tensor>& source,
    const std::string& method,
    const std::optional<std::string>& normalize
) {
    const auto L = graph_laplacian(adjacency, true);
    const auto H = heat_kernel(L, tau);

    torch::Tensor f;
    if (method == "content") {
        f = H.diagonal();
    } else if (method == "diffusion") {
        const auto n = adjacency.size(0);
        torch::Tensor initial_source;
        if (source.has_value()) {
            initial_source = source.value() / source.value().sum();
        } else {
            initial_source = torch::ones(n) / n;
        }
        f = H * initial_source;
    } else {
        throw std::invalid_argument("Unknown method: " + method);
    }

    if (normalize.has_value()) {
        if (normalize.value() == "rank") {
            f = torch::argsort(torch::argsort(f)) / f.size(0);
        } else if (normalize.value() == "minmax") {
            f_min, f_max = f.min(), f.max();
            if (f_max > f_min) {    
                f = (f - f_min) / (f_max - f_min);
            } else {
                f = torch::zeros_like(f);
            }
        } else if (normalize.value() == "zscore") {
            f_mean, f_std = f.mean(), f.std();
            if (f_std > 0) {
                f = (f - f_mean) / f_std;
            } else {
                f = torch::zeros_like(f);
            }
        } else {
            throw std::invalid_argument("Unknown normalization method: " + normalize.value());
        }
    }
    return f;
}

torch::Tensor lower_star_filtration_value(
    torch::Tensor clique_vertices,
    torch::Tensor vertex_function
) {
    TORCH_CHECK(clique_vertices.dim() == 1, "clique_vertices must be 1D");
    TORCH_CHECK(vertex_function.dim() == 1, "vertex_function must be 1D");
    return std::get<0>(vertex_function.index_select(0, clique_vertices).max(0));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "graph_laplacian",
        &graph_laplacian,
        "Compute graph Laplacian",
        py::arg("adjacency"),
        py::arg("normalized") = true
    );
    m.def(
        "heat_kernel",
        &heat_kernel,
        "Compute heat kernel H(tau) = exp(-tau * L)",
        py::arg("L"),
        py::arg("tau")
    );
    m.def(
        "heat_edge_weights",
        &heat_edge_weights,
        "Compute heat-based edge weights w_tau(u,v) = H(tau)_{uv}",
        py::arg("adjacency"),
        py::arg("tau") = 1.0,
        py::arg("normalized") = true
    );
    m.def(
        "heat_vertex_function",
        &heat_vertex_function,
        "Compute heat-derived vertex function for lower-star filtration",
        py::arg("adjacency"),
        py::arg("tau") = 1.0,
        py::arg("source") = py::none(),
        py::arg("method") = "content",
        py::arg("normalize") = "rank"
    );
    m.def(
        "lower_star_filtration_value",
        &lower_star_filtration_value,
        "Compute lower-star filtration value: f(sigma) = max_{v in sigma} f(v)",
        py::arg("clique_vertices"),
        py::arg("vertex_function")
    );
}
