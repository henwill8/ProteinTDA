#include "metropolis_hastings_sampling_cuda.hpp"
#include "sampling_common.cuh"

#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <driver_types.h>
#include <math_constants.h>
#include <math.h>
#include <random>
#include <numbers>
#include <vector>

__global__ void delta_laplacian_symbol(double* delta,
    const double* theta,
    int k,
    double* proposed_val,
    double sigma,
    curandState *states,
    Heat_Kernel_device kernel
) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;      
    if (i >= kernel.dim) return;

    curandState local_state = states[i];

    double prop = theta[k] + sigma * (2 * curand_uniform(&local_state) - 1);
    states[i] = local_state;

    prop = std::fmod(prop, TWO_PI);
    if (prop < 0.0) {
        prop += TWO_PI;
    }
    *proposed_val = prop;

    if (i == k) return;
    double weight = qdist(i, k, kernel);
    if (weight == 0) return;
    double delta_i = 2 * weight * (cos(theta[k] - theta[i]) - cos(*proposed_val - theta[i]));

    if (i == 0) {
        weight = dist_to_diagonal_grid(k, kernel);
        delta_i += 2 * weight * (cos(theta[k]) - cos(*proposed_val));
    }

    atomicAdd(delta, delta_i);
}

std::vector<double> cuda_sample(double sigma, int burn_in, int thinning, bool normalize, int edge_weights_total, int seed, Heat_Kernel_device& kernel, SamplingMethod& base_method) {
    const double TWO_PI = 2.0 * std::numbers::pi;
    const int dim = kernel.dim;
    const size_t array_size = dim * sizeof(double);

    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<double> uniform_dist(0.0,1.0);

    double *curr_theta;
    CUDA_CHECK(cudaMalloc(&curr_theta, array_size));

    double *lambda_device, *curr_lambda, *curr_delta_lambda;
    CUDA_CHECK(cudaMalloc(&lambda_device,sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&curr_lambda,sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&curr_delta_lambda,sizeof(double)));

    double *final_thetas;
    CUDA_CHECK(cudaMallocHost(&final_thetas, array_size * kernel.R));

    curandState *rand_states;
    CUDA_CHECK(cudaMalloc(&rand_states, dim * sizeof(curandState)));

    const int THREADSPERBLOCK = 256;
    const int BLOCKSPERGRID = (dim + THREADSPERBLOCK - 1) / THREADSPERBLOCK;

    setup_random_states<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(rand_states, seed, dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    sample_theta<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, rand_states, dim);
    base_method.add_op(base_method.ops_per_theta_sampling_);
    
    CUDA_CHECK(cudaMemset(lambda_device, 0, sizeof(double)));
    laplacian_symbol<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, lambda_device, kernel);
    CUDA_CHECK(cudaMemcpy(curr_lambda, lambda_device, sizeof(double), cudaMemcpyDeviceToHost));
    if (normalize) {
        *curr_lambda /= edge_weights_total;
    }
    double *prop;
    CUDA_CHECK(cudaMalloc(&prop, sizeof(double)));
    
    auto mcmc_pass = [&]() {
        for (int k = 0; k < dim; ++k) {
            CUDA_CHECK(cudaMemset(lambda_device, 0, sizeof(double)));
            delta_laplacian_symbol<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(lambda_device, curr_theta, k, prop, sigma, rand_states, kernel);
            base_method.add_op(kernel.dim + 1);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(curr_delta_lambda, lambda_device, sizeof(double), cudaMemcpyDeviceToHost));

            double next_lambda = *curr_lambda + *curr_delta_lambda;

            double log_alpha = -kernel.t * *curr_delta_lambda + std::log1p(-std::exp(-kernel.s * next_lambda)) - std::log1p(-std::exp(-kernel.s * *curr_lambda));

            if (std::log(uniform_dist(gen)) > log_alpha) {
                CUDA_CHECK(cudaMemcpy(curr_theta + k, prop, sizeof(double), cudaMemcpyHostToDevice));
                *curr_lambda = next_lambda;
            }
        }
    };

    for (int s = 0; s < burn_in; ++s) mcmc_pass();

    for (int r = 0; r < kernel.R; ++r) {
        for (int s = 0; s < thinning; ++s) mcmc_pass();
        CUDA_CHECK(cudaMemcpy(&final_thetas[r * dim], curr_theta, array_size, cudaMemcpyDeviceToHost));
    }

    std::vector<double> final_thetas_vector(final_thetas, final_thetas + kernel.R * dim);

    CUDA_CHECK(cudaFree(curr_theta));
    CUDA_CHECK(cudaFree(lambda_device));
    CUDA_CHECK(cudaFree(prop));
    CUDA_CHECK(cudaFree(rand_states));
    CUDA_CHECK(cudaFreeHost(curr_lambda));
    CUDA_CHECK(cudaFreeHost(curr_delta_lambda));
    CUDA_CHECK(cudaFreeHost(final_thetas));

    return final_thetas_vector;
}
