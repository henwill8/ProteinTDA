#include "rejection_sampling_cuda.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <random>
#include <vector>

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

void cuda_sample(double sigma, bool normalize, int total_edge_weights, int seed, Heat_Kernel& kernel) {
    const int dim = kernel->dim;
    const size_t array_size = dim * sizeof(double);

    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<double> uniform_dist(0.0,1.0);

    double *curr_theta;
    CUDA_CHECK(cudaMalloc(&curr_theta, array_size));

    double* final_thetas;
    CUDA_CHECK(cudaMallocHost(&final_thetas, array_size * kernel.R));

    curandState* rand_states;
    CUDA_CHECK(cudaMalloc(&rand_states, dim * sizeof(curandState)));

    const int THREADSPERBLOCK = 256;
    const int BLOCKSPERGRID = (dim + THREADSPERBLOCK - 1) / THREADSPERBLOCK; 

    setup_random_states<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(rand_states, seed, dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    sample_theta<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, rand_states, dim);

    double *lambda_device, *lambda_host;
    CUDA_CHECK(cudaMalloc(&lambda_device, sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&lambda_host, sizeof(double)));

    for (int r = 0; r < kernel->R; ++r) {
        for (;;) {
            sample_theta<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, rand_states, dim);
            CUDA_CHECK(cudaMemset(lambda_device, 0, sizeof(double)));
            laplacian_symbol<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, lambda_device, *kernel);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(lambda_host, lambda_device, sizeof(double), cudaMemcpyDeviceToHost));
            double weight = std::exp(-kernel->t * *lambda_host) * (1 - std::exp(-kernel->s * *lambda_host));
            if (uniform_dist(gen) <= weight) {
                CUDA_CHECK(cudaMemcpy(final_thetas[r * kernel->dim], curr_theta, array_size, cudaMemcpyDeviceToHost));
                break;
            }
        }
    }

    std::vector<double> final_thetas_vector(final_thetas, final_thetas + kernel.R * dim);
    kernel->thetas = final_thetas_vector;

    CUDA_CHECK(cudaFree(curr_theta));
    CUDA_CHECK(cudaFree(lambda_device));
    CUDA_CHECK(cudaFree(rand_states));
    CUDA_CHECK(cudaFreeHost(lambda_host));
    CUDA_CHECK(cudaFreeHost(final_thetas));
}
