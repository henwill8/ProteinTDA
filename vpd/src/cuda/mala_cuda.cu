#include "mala_cuda.hpp"

#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_types.h>
#include <numbers>

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
    node_at(index, px, py, kernel);
    return dist_coords_to_diagonal_grid(px, py, kernel);
}


__device__ double qdist(int i, int j, const Heat_Kernel kernel) {
    double p1_x, p1_y, p2_x, p2_y;
    node_at(i, p1_x, p1_y, kernel);
    node_at(j, p2_x, p2_y, kernel);

    const double dx = p2_x - p1_x;
    const double dy = p2_y - p1_y;

    const double d_euclidean = sqrt(dx * dx + dy * dy);
    const double d_line = dist_coords_to_diagonal_grid(p1_x, p1_y, kernel) + dist_coords_to_diagonal_grid(p2_x, p2_y, kernel);

    return fmin(d_euclidean, d_line);
}

__global__ void setup_random_states(curandState* state, int seed, int dim) {
    int i = threadIdx.x + blockDim.x + blockIdx.x; 

    if (i >= dim) return;

    curand_init(seed, i, 0, &state[i]);
}

__global__ void laplacian_symbol(const double* theta, double* lambda, const Heat_Kernel kernel) {
    int i = threadIdx.x + blockDim.x + blockIdx.x;
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

__global__ void grad_laplacian_symbol(const double* theta, double* grad, const Heat_Kernel kernel) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= kernel.dim) return;
    
    double di = 0.0;
    double theta_i = theta[i];

    for (int j = 0; j < kernel.dim; ++j) {
        if (i == j) continue;
        double weight = qdist(i,j, kernel);
        if (weight == 0) continue;
        di += 2 * weight * sin(theta_i - theta[j]);
    }
    double weight = dist_to_diagonal_grid(i, kernel);
    di += 2 * weight * sin(theta[i]); 
    grad[i] = di;
}

__global__ void multiply_vector(double* vector, double c, int dim) {
    int i = threadIdx.x + blockDim.x + blockIdx.x;
    if (i >= dim) return;

    vector[i] *= c;
}

__global__ void sample_theta (double *theta, curandState *states, int dim) {
    int i = threadIdx.x + blockDim.x + blockIdx.x; 
    if (i >= dim) return;

    curandState local_state = states[i];
    theta[i] = curand_uniform(&local_state);
    states[i] = local_state;
}

__global__ void drift_theta (double *curr_theta, 
        double *prop_theta,
        double *curr_grad,
        curandState* states,
        double sigma,
        int dim
) {

    int i = threadIdx.x + blockDim.x + blockIdx.x;
    if (i >= dim) return;

    curandState local_state = states[i];
    double drift = sigma * curr_grad[i];
    double gaussian = curand_normal_double(&local_state);
    double brownian = sqrt(2 * sigma) * gaussian;
    states[i] = local_state;
    prop_theta[i] = wrap_2pi(curr_theta[i] + drift + brownian);
}

void cuda_sample(int R, double sigma, int mala_burn_in, int mala_sigma, bool tune, int seed, Heat_Kernel& kernel) {
    const double TWO_PI = 2.0 * std::numbers::pi;
    const double OPTIMAL = 0.574;
    const int dim = kernel.dim;
    const size_t array_size = dim * sizeof(double);

    double *curr_theta, *prop_theta, *curr_grad, *prop_grad;
    cudaMalloc(&curr_theta, array_size);
    cudaMalloc(&prop_theta, array_size);
    cudaMalloc(&curr_grad, array_size);
    cudaMalloc(&prop_grad, array_size);
    
    double* final_thetas;
    cudaMalloc(&final_thetas, array_size * R);

    curandState* rand_states;
    cudaMalloc(&rand_states, dim * sizeof(curandState));

    const int THREADSPERBLOCK = 256;
    const int BLOCKSPERGRID = (dim + THREADSPERBLOCK - 1) / THREADSPERBLOCK; 

    auto wrap_pi = [&](double x) {
        x = std::fmod(x, TWO_PI); if (x <= -std::numbers::pi) x += TWO_PI; else if (x > std::numbers::pi) x-= TWO_PI; return x;
    };

    setup_random_states<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(rand_states, seed, kernel.dim);
    sample_theta<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, rand_states, kernel.dim);

    double *curr_lambda_cuda;
    cudaMalloc(&curr_lambda_cuda, sizeof(double));
    double *curr_lambda_cpu = (double*)malloc(sizeof(double));

    auto compute_grad = [&](double *theta, double *grad, double *U){
        cudaMemset(curr_lambda_cuda, 0.0, sizeof(double));
        laplacian_symbol<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(theta, curr_lambda_cuda, kernel);
        grad_laplacian_symbol<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(theta, grad, kernel);
        cudaMemcpy(curr_lambda_cpu, curr_lambda_cuda, sizeof(double) ,cudaMemcpyDeviceToHost);
        double dUdL = (kernel.t - kernel.s / (std::expm1(kernel.s * *curr_lambda_cpu)));
        multiply_vector(grad, dUdL, kernel.dim);
        *U = kernel.t * *curr_lambda_cpu - std::log1p(-std::exp(-kernel.s * *curr_lambda_cpu));
        return U;
    };

    double curr_U;
    compute_grad(curr_theta, curr_grad, &curr_U); 
    
    auto mala_pass = [&](bool tune) {
        
    }
}
