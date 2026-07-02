#pragma once

#include "heat_kernel.hpp"

void cuda_sample(int R, double sigma, int mala_burn_in, int mala_sigma, bool tune, int seed, Heat_Kernel& kernel); 
