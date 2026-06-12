#include "utils.hpp"

#include <cmath>

// Quotient distance
torch::Tensor qdist(const torch::Tensor& p1, const torch::Tensor& p2) {
    TORCH_CHECK(p1.dim() == 1, "p1 must be 1D");
    TORCH_CHECK(p2.dim() == 1, "p2 must be 1D");
    TORCH_CHECK(p1.size(0) == 2, "p1 must have 2 elements");
    TORCH_CHECK(p2.size(0) == 2, "p2 must have 2 elements");

    // Euclidean distance
    const auto dx = p2[0] - p1[0];
    const auto dy = p2[1] - p1[1];
    const auto d_euclidean = sqrt(dx * dx + dy * dy);

    // Distance to line y = x
    const auto d_line = fabs(p1[1] - p1[0]) / sqrt(2.0f) + fabs(p2[1] - p2[0]) / sqrt(2.0f);

    return fmin(d_euclidean, d_line);
}

// Not sure if I interpreted this correctly. Also I am using tensors anyways just cause theyre
// easier to work with and I am not sure how the std::vectors are passed in in python.
torch::Tensor laplacian_eigenvalues(const torch::Tensor& theta, const torch::Tensor& edges) {
    TORCH_CHECK(theta.dim() == 1, "theta must be 1D");
    TORCH_CHECK(edges.dim() == 2, "edges must be 2D");
    TORCH_CHECK(
        edges.size(0) == theta.size(0) && edges.size(1) == theta.size(0),
        "edges must be square with size matching theta"
    );

    double result = 0.0;
    const auto n = theta.size(0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            result += edges.index({i, j}).item<double>()
                * (1.0 - cos(theta.index({i}).item<double>() - theta.index({j}).item<double>()));
        }
    }
    return torch::tensor(result, theta.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "qdist",
        &qdist,
        "Compute quotient distance",
        py::arg("x1"),
        py::arg("y1"),
        py::arg("x2"),
        py::arg("y2")
    );
    m.def(
        "laplacian_eigenvalues",
        &laplacian_eigenvalues,
        "Compute Laplacian eigenvalues",
        py::arg("theta"),
        py::arg("edges")
    );
}