#include "mala_sampling_cuda.hpp"
#include "sampling_common.cuh"

#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <driver_types.h>
#include <iostream>
#include <math_constants.h>
#include <numbers>

__global__ void mala_laplacians(const double* theta, double* lambda, double* grad, bool normalize, int edge_weights_total, const Heat_Kernel_device kernel) {
    constexpr int BLOCK = 256;
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    double d_i = 0.0;
    double lambda_i = 0;
    
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < kernel.dim) { 

        double theta_i = theta[i];

        for (int j = 0; j < kernel.dim; ++j) {
            if (i == j) continue;
            double weight = qdist(i,j, kernel);
            if (weight == 0) continue;
            double diff = theta_i - theta[j];
            d_i += 2 * weight * sin(diff);
            lambda_i += weight * (1 - cos(diff));
        }
        double weight = dist_to_diagonal_grid(i, kernel);
        d_i += 2 * weight * sin(theta_i); 
        lambda_i += 2 * weight * (1 - cos(theta_i));
        if (normalize) {
            d_i /= edge_weights_total;
        }
        grad[i] = d_i;
    }

    double block_sum = BlockReduce(temp).Sum(lambda_i);
    if (threadIdx.x == 0) {
        atomicAdd(lambda, block_sum);
    }
}

__global__ void multiply_vector(double* vector, double c, int dim) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= dim) return;

    vector[i] *= c;
}

__global__ void drift_theta (double *curr_theta, 
        double *prop_theta,
        double *curr_grad,
        curandState* states,
        double sigma,
        int dim
) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= dim) return;

    curandState local_state = states[i];
    double drift = -sigma * curr_grad[i];
    double gaussian = curand_normal_double(&local_state);
    double brownian = sqrt(2 * sigma) * gaussian;
    states[i] = local_state;
    prop_theta[i] = wrap_2pi(curr_theta[i] + drift + brownian);
}

__global__ void compute_move_probabilities(double *curr_theta,
        double *prop_theta,
        double *curr_grad,
        double *prop_grad,
        double *q_fwd,
        double *q_bwd,
        double sigma,
        int dim
) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= dim) return;

    double d = wrap_pi(curr_theta[i] - prop_theta[i]);
    double fwd = (d - sigma * curr_grad[i]) * (d - sigma * curr_grad[i]);
    double bwd = (-d - sigma * prop_grad[i]) * (-d - sigma * prop_grad[i]);
    atomicAdd(q_fwd, fwd);
    atomicAdd(q_bwd, bwd);
}

std::vector<double> cuda_sample(double sigma, int burn_in, int thinning, bool tune, bool normalize, int edge_weights_total, int seed, Heat_Kernel_device& kernel, SamplingMethod& base_method) {
    const double OPTIMAL = 0.574;
    const int dim = kernel.dim;
    const size_t dim_size_t = static_cast<size_t>(kernel.dim);
    const size_t array_size = dim * sizeof(double);

    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    double *curr_theta, *prop_theta, *curr_grad, *prop_grad;
    CUDA_CHECK(cudaMalloc(&curr_theta, array_size));
    CUDA_CHECK(cudaMalloc(&prop_theta, array_size));
    CUDA_CHECK(cudaMalloc(&curr_grad, array_size));
    CUDA_CHECK(cudaMalloc(&prop_grad, array_size));
    
    double* final_thetas;
    CUDA_CHECK(cudaMallocHost(&final_thetas, array_size * kernel.R));

    curandState* rand_states;
    CUDA_CHECK(cudaMalloc(&rand_states, dim_size_t * sizeof(curandState)));

    const int THREADSPERBLOCK = 256;
    const int BLOCKSPERGRID = (dim + THREADSPERBLOCK - 1) / THREADSPERBLOCK; 

    setup_random_states<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(rand_states, seed, dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    sample_theta<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, rand_states, dim);
    base_method.add_op(base_method.ops_per_theta_sampling_);

    double *lambda_device, *curr_lambda_host, *prop_lambda_host;
    CUDA_CHECK(cudaMalloc(&lambda_device, sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&curr_lambda_host, sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&prop_lambda_host, sizeof(double)));
    
    double *q_device, *q_host;
    CUDA_CHECK(cudaMalloc(&q_device, 2 * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&q_host, 2 * sizeof(double)));


    auto compute_grad = [&](double *theta, double *grad, double* lambda_device, double* lambda_host){
        CUDA_CHECK(cudaMemset(lambda_device, 0, sizeof(double)));
        mala_laplacians<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(theta, lambda_device, grad, normalize, edge_weights_total, kernel);
        base_method.add_op(2 * static_cast<int64_t>(dim) * (1  + dim));
        CUDA_CHECK(cudaMemcpy(lambda_host, lambda_device, sizeof(double), cudaMemcpyDeviceToHost));
        if (normalize) {
            *lambda_host /= edge_weights_total; 
        }
        double dUdL = (kernel.t - kernel.s / (std::expm1(kernel.s * *lambda_host)));
        multiply_vector<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(grad, dUdL, dim);
        base_method.add_op(dim);
    };

    compute_grad(curr_theta, curr_grad, lambda_device, curr_lambda_host); 
    
    auto mala_pass = [&](bool tune) {
        drift_theta<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, prop_theta, curr_grad, rand_states, sigma, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        base_method.add_op(dim);

        compute_grad(prop_theta, prop_grad, lambda_device, prop_lambda_host);

        CUDA_CHECK(cudaMemset(q_device, 0, 2 * sizeof(double)));
        CUDA_CHECK(cudaMemset(q_host, 0, 2 * sizeof(double)));
        CUDA_CHECK(cudaDeviceSynchronize());

        compute_move_probabilities<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, prop_theta, curr_grad, prop_grad, q_device, q_device + 1, sigma, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(q_host, q_device, 2 * sizeof(double), cudaMemcpyDeviceToHost));
        base_method.add_op(dim);

        double alpha_log = (q_host[0] - q_host[1]) / (4 * sigma) - kernel.t * (*prop_lambda_host - *curr_lambda_host) + std::log1p(-std::exp(-kernel.s * *prop_lambda_host)) - std::log1p(-std::exp(-kernel.s * *curr_lambda_host));
        double alpha = std::min(1.0, std::exp(alpha_log));

        if (std::log(uniform_dist(gen)) < alpha_log) {
            std::swap(curr_theta, prop_theta); 
            std::swap(curr_grad, prop_grad); 
            std::swap(curr_lambda_host, prop_lambda_host);
        }

        if (tune) {
            sigma *= std::exp(0.05 * (alpha - OPTIMAL));
            sigma = std::clamp(sigma, 1e-6, 0.5);
        }
    };

    for (int s = 0; s < burn_in; ++s) mala_pass(tune);

    for (int r = 0; r < kernel.R; ++r) {
        for (int s = 0; s < thinning; ++s) mala_pass(false);
        CUDA_CHECK(cudaMemcpy(&final_thetas[r * dim_size_t], curr_theta, array_size, cudaMemcpyDeviceToHost));
    }

    std::vector<double> final_thetas_vector(final_thetas, final_thetas + kernel.R * dim_size_t);

    CUDA_CHECK(cudaFree(curr_theta));
    CUDA_CHECK(cudaFree(prop_theta));
    CUDA_CHECK(cudaFree(curr_grad));
    CUDA_CHECK(cudaFree(prop_grad));
    CUDA_CHECK(cudaFreeHost(final_thetas));
    CUDA_CHECK(cudaFree(rand_states));
    CUDA_CHECK(cudaFree(lambda_device));
    CUDA_CHECK(cudaFreeHost(curr_lambda_host));
    CUDA_CHECK(cudaFreeHost(prop_lambda_host));
    CUDA_CHECK(cudaFreeHost(q_host));
    CUDA_CHECK(cudaFree(q_device));

    return final_thetas_vector;
}
