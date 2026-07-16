#pragma once

#include "sampling_method.hpp"

#ifdef VPD_WITH_CUDA
#include "random_sampling_cuda.hpp"
#endif

#include <utility>

class RandomSampling : public SamplingMethod {
    private:
        void cpu_sample();
        void reset_progress() override;
        void sample() override;
};
