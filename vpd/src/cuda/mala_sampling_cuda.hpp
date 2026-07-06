#pragma once

#include "heat_kernel.hpp"
#include "sampling_method.hpp"
#include "sampling_method_cuda.hpp"

std::vector<double> cuda_sample(double sigma, int burn_in, int thinning, bool tune, bool normalize, int total_edge_weights, int seed, Heat_Kernel_device& kernel, SamplingMethod& base_method); 
