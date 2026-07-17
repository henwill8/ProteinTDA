#include "rejection_sampling_cuda.hpp"
#include "sampling_common.cuh"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <random>
#include <vector>

std::vector<double> cuda_sample(bool normalize, int edge_weights_total, int seed, Heat_Kernel_device& kernel, SamplingMethod& base_method) {
    const int dim = kernel.dim;
    const size_t array_size = dim * sizeof(double);

    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<double> uniform_dist(0.0,1.0);

    double *curr_theta;
    CUDA_CHECK(cudaMalloc(&curr_theta, array_size));

    double* final_thetas;
    CUDA_CHECK(cudaMallocHost(&final_thetas, array_size * kernel.R));

    curandState *rand_states;
    CUDA_CHECK(cudaMalloc(&rand_states, dim * sizeof(curandState)));

    const int THREADSPERBLOCK = 256;
    const int BLOCKSPERGRID = (dim + THREADSPERBLOCK - 1) / THREADSPERBLOCK; 

    setup_random_states<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(rand_states, seed, dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    double *lambda_device, *lambda_host;
    CUDA_CHECK(cudaMalloc(&lambda_device, sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&lambda_host, sizeof(double)));

    for (int r = 0; r < kernel.R; ++r) {
        for (;;) {
            sample_theta<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, rand_states, dim);
            CUDA_CHECK(cudaMemset(lambda_device, 0, sizeof(double)));
            laplacian_symbol<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, lambda_device, kernel);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(lambda_host, lambda_device, sizeof(double), cudaMemcpyDeviceToHost));
            double weight = std::exp(-kernel.t * *lambda_host) * (1 - std::exp(-kernel.s * *lambda_host));
            if (uniform_dist(gen) <= weight) {
                CUDA_CHECK(cudaMemcpy(&final_thetas[r * kernel.dim], curr_theta, array_size, cudaMemcpyDeviceToHost));
                break;
            }
        }
    }

    std::vector<double> final_thetas_vector(final_thetas, final_thetas + kernel.R * dim);

    CUDA_CHECK(cudaFree(curr_theta));
    CUDA_CHECK(cudaFree(lambda_device));
    CUDA_CHECK(cudaFree(rand_states));
    CUDA_CHECK(cudaFreeHost(lambda_host));
    CUDA_CHECK(cudaFreeHost(final_thetas));

    return final_thetas_vector;
}
