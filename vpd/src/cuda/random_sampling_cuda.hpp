#pragma once

#include "heat_kernel.hpp"
#include "sampling_method.hpp"
#include "sampling_method_cuda.hpp"

#include <vector>
#include <utility>

std::pair<std::vector<double>, std::vector<double>> cuda_sample_random(bool normalize, int edge_weights_total, int seed, Heat_Kernel_device& kernel, SamplingMethod& base_method);
