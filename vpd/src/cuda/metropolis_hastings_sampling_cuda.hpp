#pragma once

#include "heat_kernel.hpp"
#include "sampling_method.hpp"
#include "sampling_method_cuda.hpp"

void cuda_sample(double sigma, int burn_in, int thinning, bool normalize, int total_edge_weights, int seed, Heat_Kernel& kernel); 
