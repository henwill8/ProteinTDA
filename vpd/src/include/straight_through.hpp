#pragma once

#include <torch/torch.h>
#include <torch/extension.h>

torch::Tensor straight_through_round(torch::Tensor x);
torch::Tensor straight_through_bincount(torch::Tensor indices, int64_t dim);
