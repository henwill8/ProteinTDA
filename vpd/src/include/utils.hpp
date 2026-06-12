#pragma once

#include <torch/extension.h>

float qdist(const float& x1, const float& y1, const float& x2, const float& y2);
torch::Tensor laplacian_eigenvalues(const torch::Tensor& theta, const torch::Tensor& edges);
