#include "random_sampling_cuda.hpp"
#include "sampling_common.cuh"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <driver_types.h>
#include <iostream>
#include <math_constants.h>
std::pair<std::vector<double>, std::vector<double>> cuda_sample_random(bool normalize, int edge_weights_total, int seed, Heat_Kernel_device& kernel, SamplingMethod& base_method) {
    const int dim = kernel.dim;
    const size_t array_size = dim * sizeof(double);
    
    double *device_thetas;
    double *host_thetas;
    CUDA_CHECK(cudaMalloc(&device_thetas, array_size * kernel.R));
    CUDA_CHECK(cudaMallocHost(&host_thetas, array_size * kernel.R));

    double *device_lambdas;
    double *host_lambdas;
    CUDA_CHECK(cudaMalloc(&device_lambdas, kernel.R * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&host_lambdas, kernel.R * sizeof(double)));

    curandState *rand_states;
    CUDA_CHECK(cudaMalloc(&rand_states, dim * sizeof(curandState)));

    const int THREADSPERBLOCK = 256;
    const int BLOCKSPERGRID = (dim + THREADSPERBLOCK - 1) / THREADSPERBLOCK; 

    setup_random_states<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(rand_states, seed, dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int r = 0; r < kernel.R; ++r) { 
        double *curr_theta = device_thetas + (r * kernel.dim);
        double *lambda = device_lambdas + r;
        sample_theta<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, rand_states, dim);
        laplacian_symbol<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, lambda, kernel); 
        base_method.add_op(base_method.ops_per_theta_sampling_);
        base_method.add_op(kernel.dim * kernel.dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(host_thetas, device_thetas, array_size * kernel.R, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_lambdas, device_lambdas, kernel.R * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> final_thetas_vector(host_thetas, host_thetas + kernel.R * dim);
    std::vector<double> final_lambdas_vector(host_lambdas, host_lambdas + kernel.R);

    if (normalize) {
        for (int r = 0; r < kernel.R; ++r) {
            final_lambdas_vector[r] /= edge_weights_total;
        }
    }

    CUDA_CHECK(cudaFree(device_thetas));
    CUDA_CHECK(cudaFree(device_lambdas));
    CUDA_CHECK(cudaFree(rand_states));
    CUDA_CHECK(cudaFreeHost(host_thetas));
    CUDA_CHECK(cudaFreeHost(host_lambdas));

    return std::make_pair(final_thetas_vector, final_lambdas_vector);
}
