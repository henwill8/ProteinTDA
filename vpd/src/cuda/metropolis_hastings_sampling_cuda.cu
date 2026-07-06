#include "metropolis_hastings_sampling_cuda.hpp"
#include "sampling_method_cuda.hpp"

#include <__clang_device_builtin_vars.h>
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

__device__ double wrap_2pi(double x) {
    const double TWO_PI = 2.0 * CUDART_PI;
    x = fmod(x, TWO_PI);
    if (x < 0) x += TWO_PI;
    return x;
}

__device__ void node_at(int index, double &x, double &y, const Heat_Kernel kernel) {
    if (kernel.n == 1) {
        x = (index + 1) / kernel.resolution;
        y = 0.0;
        return;
    }
    
    const int iy = static_cast<int>((sqrt(8.0 * index + 1.0) - 1.0) / 2.0); // solution to iy(iy + 1) / 2 <= index
    const int ix = index - iy * (iy + 1) / 2; // checks how many nodes were in the previous rows n(n + 1) / 2

    x = ix / kernel.resolution;
    y = iy / kernel.resolution;
}

__device__ double dist_coords_to_diagonal_grid(double px, double py, const Heat_Kernel kernel) {
    double t = 0.5 * (px + py);

    double min_t = 0.0;
    double max_t = kernel.ppa;

    double d_grid = round((t - min_t) * kernel.resolution) / kernel.resolution + min_t; 
    d_grid =  fmaxf(min_t, fminf(d_grid, max_t));


    double dx = px - d_grid;
    double dy = py - d_grid;
    return sqrt(dx * dx + dy * dy);
}

__device__ double dist_to_diagonal_grid(int index, const Heat_Kernel kernel) {
    double px, py;
    node_at(index, px, py, kernel.;
    return dist_coords_to_diagonal_grid(px, py, kernel.;
}


__device__ double qdist(int i, int j, const Heat_Kernel kernel) {
    double p1_x, p1_y, p2_x, p2_y;
    node_at(i, p1_x, p1_y, kernel.;
    node_at(j, p2_x, p2_y, kernel.;

    const double dx = p2_x - p1_x;
    const double dy = p2_y - p1_y;

    const double d_euclidean = sqrt(dx * dx + dy * dy);
    const double d_line = dist_coords_to_diagonal_grid(p1_x, p1_y, kernel. + dist_coords_to_diagonal_grid(p2_x, p2_y, kernel.;

    return fmin(d_euclidean, d_line);
}
__global__ void setup_random_states(curandState* state, int seed, int dim) {
    int i = threadIdx.x + blockDim.x * blockIdx.x; 

    if (i >= dim) return;

    curand_init(seed, i, 0, &state[i]);
}

__global__ void sample_theta (double *theta, curandState *states, int dim) {
    int i = threadIdx.x + blockDim.x * blockIdx.x; 
    if (i >= dim) return;

    curandState local_state = states[i];
    theta[i] = curand_uniform(&local_state);
    states[i] = local_state;
}

__global__ void laplacian_symbol(const double* theta, double* lambda, const Heat_Kernel kernel) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= kernel.dim) return;

    double theta_i = theta[i];
    double lambda_i = 0;

    for (int j = i + 1; j < kernel.dim; ++j) {
        double weight = qdist(i, j, kernel);
        if (weight == 0) continue;
        double diff = theta_i - theta[j];
        lambda_i += 2 * weight * (1 - cos(diff));
    }
    double weight = dist_to_diagonal_grid(i, kernel);
    lambda_i += 2 * weight * (1 - cos(theta_i));

    atomicAdd(lambda, lambda_i);
}

__global__ void delta_laplacian_symbol(double* delta,
    const double* theta,
    int k,
    double proposed_val,
    Heat_Kernel kernel
) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;      
    if (i >= kernel.dim) return;

    if (i == k) return;
    double weight = qdist(i, k, kernel);
    if (weight == 0) return;
    double delta_i = 2 * weight * (cos(theta[k] - theta[i]) - cos(proposed_val - theta[i]));

    if (i == 0) {
        weight = dist_to_diagonal_grid(k, kernel);
        delta_i += 2 * weight * (cos(theta_k) - cos(proposed_val));
    }

    atomicAdd(delta, delta_i);
}

void cuda_sample(double sigma, int burn_in, int thinning, bool normalized_lambdas, int total_edge_weights, int seed, Heat_Kernel& kernel) {
    const int dim = kernel->dim;
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

    curandState* rand_states;
    CUDA_CHECK(cudaMalloc(&rand_states, dim * sizeof(curandState)));

    const int THREADSPERBLOCK = 256;
    const int BLOCKSPERGRID = (dim + THREADSPERBLOCK - 1) / THREADSPERBLOCK;

    setup_random_states<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(rand_states, seed, dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    sample_theta<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, rand_states, dim);
    add_op(ops_per_theta_sampling_);
    
    CUDA_CHECK(cudaMemset(lambda_device, 0, sizeof(double)));
    laplacian_symbol(curr_theta, lambda_device, *kernel);
    CUDA_CHECK(cudaMemcpy(curr_lambda, lambda_device, sizeof(double), cudaMemcpyDeviceToHost));
    if (normalized) {
        *curr_lambda /= total_edge_weights;
    }
    
    auto mcmc_pass = [&]() {
        for (int k = 0; k < dim; ++k) {
            double prop = curr_thetas[k] + this->mcmc_sigma * (2 * uniform_dist(gen) - 1);
            prop = std::fmod(prop, TWO_PI);
            if (prop < 0.0) {
                prop += TWO_PI;
            }

            CUDA_CHECK(cudaMemset(lambda_device, 0, sizeof(double)));
            delta_laplacian_symbol(lambda_device, curr_theta, k, prop, *kernel);
            add_op(kernel->dim + 1);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(curr_delta_lambda, lambda_device, sizeof(double), cudaMemcpyDeviceToHost))

            double next_lambda = *curr_lambda + *curr_delta_lambda;

            double log_alpha = -kernel->t * *curr_delta_lambda + std::log1p(std::exp(-kernel->s * next_lambda)) - std::log1p(std::exp(-kernel->s * *curr_lambda));

            if (std::log(uniform_dist(gen)) > log_diff) {
                CUDA_CHECK(cudaMemcpy(curr_thetas[k], prop, sizeof(double), cudaMemcpyHostToDevice));
                *curr_lambda = next_lambda;
            }
        }
    };

    for (int s = 0; s < burn_in; ++s) mcmc_pass();

    for (int r = 0; r < kernel->R; ++r) {
        for (int s = 0; s < thinning; ++s) mcmc_pass();
        CUDA_CHECK(cudaMemcpy(final_thetas[r * dim], curr_theta, array_size, cudaMemcpyDeviceToHost));
    }

    std::vector<double> final_thetas_vector(final_thetas, final_thetas + kernel.R * dim);
    kernel->thetas = final_thetas_vector;

    CUDA_CHECK(cudaFree(curr_theta));
    CUDA_CHECK(cudaFree(lambda_device));
    CUDA_CHECK(cudaFree(rand_states));
    CUDA_CHECK(cudaFreeHost(curr_lambda));
    CUDA_CHECK(cudaFreeHost(curr_delta_lambda));
    CUDA_CHECK(cudaFreeHost(final_thetas));

}
