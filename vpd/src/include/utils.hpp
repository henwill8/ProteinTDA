#pragma once

#include <torch/extension.h>

float qdist(const float& x1, const float& y1, const float& x2, const float& y2);
torch::Tensor laplacian_eigenvalues(const torch::Tensor& theta, const torch::Tensor& edges);
torch::Tensor random_thetas(int R, int dim, std::optional<uint32_t> seed);
