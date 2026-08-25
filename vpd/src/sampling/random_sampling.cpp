#include "random_sampling.hpp"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

void RandomSampling::reset_progress() {
    SamplingMethod::reset_progress();
    const int64_t ops_per_pass = ops_per_theta_sampling_ + ops_per_laplacian_;
    const int64_t passes = static_cast<int64_t>(kernel->R);
    set_total_ops(ops_per_pass * passes);
}

void RandomSampling::cpu_sample() {
    const int total = kernel->R * kernel->dim;
    std::vector<double> total_thetas(total);
    std::vector<double> weights(kernel->R);

    std::mt19937 gen(static_cast<uint32_t>(this->seed));
    std::vector<double> curr_thetas(kernel->dim);

    for (int r = 0; r < kernel->R; ++r) {
        sample_thetas(curr_thetas, gen);
        double lambda = laplacian_symbol(curr_thetas.data());
        std::copy(curr_thetas.begin(), curr_thetas.end(), total_thetas.begin() + r * kernel->dim);
        weights[r] = lambda;
    }
    kernel->thetas = std::move(total_thetas);
    kernel->weights = std::move(weights);
}

void RandomSampling::sample() {
#ifdef VPD_WITH_CUDA
    switch (this->device) {
        case Device::CPU:
            cpu_sample();
            break;
        case Device::CUDA: {
            Heat_Kernel_device cuda_kernel = Heat_Kernel_device{
                kernel->n,
                kernel->axis_dim,
                kernel->ppa,
                kernel->resolution,
                kernel->R,
                kernel->s,
                kernel->t,
                kernel->dim
            };
            const int edge_weight_total = this->normalized_lambdas
                ? static_cast<int>(this->edge_weight_total)
                : 0;
            auto sampled = cuda_sample_random(
                this->normalized_lambdas,
                edge_weight_total,
                this->seed,
                cuda_kernel,
                *this);
            kernel->thetas = sampled.first;
            kernel->weights = sampled.second;
            break;
        }
    }
#else
    cpu_sample();
#endif
}
