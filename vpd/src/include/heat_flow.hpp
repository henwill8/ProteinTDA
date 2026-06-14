#pragma once

#include <optional>
#include <string>

#include <torch/extension.h>

torch::Tensor graph_laplacian(torch::Tensor adjacency, bool normalized = true);
torch::Tensor heat_kernel(torch::Tensor L, double tau);
torch::Tensor heat_edge_weights(
    torch::Tensor adjacency,
    double tau = 1.0,
    bool normalized = true
);
torch::Tensor heat_vertex_function(
    torch::Tensor adjacency,
    double tau = 1.0,
    const std::optional<torch::Tensor>& source = std::nullopt,
    const std::string& method = "content",
    const std::optional<std::string>& normalize = "rank"
);
torch::Tensor lower_star_filtration_value(
    torch::Tensor clique_vertices,
    torch::Tensor vertex_function
);
