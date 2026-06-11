#pragma once

#include <torch/extension.h>

torch::Tensor graph_laplacian(torch::Tensor adjacency, bool normalized = true);
