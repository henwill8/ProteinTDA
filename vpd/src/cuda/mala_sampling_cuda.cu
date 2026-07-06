#include "mala_sampling_cuda.hpp"

#include <__clang_device_builtin_vars.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <driver_types.h>
#include <math_constants.h>
#include <numbers>

__device__ double wrap_2pi(double x) {
    const double TWO_PI = 2.0 * CUDART_PI;
    x = fmod(x, TWO_PI);
    if (x < 0) x += TWO_PI;
    return x;
}

__device__ double wrap_pi(double x) {
    const double TWO_PI = 2.0 * CUDART_PI;
    x = fmod(x, TWO_PI);
    if (x <= -CUDART_PI) x += TWO_PI;
    else if (x > CUDART_PI) x -= TWO_PI;
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

__global__ void grad_laplacian_symbol(const double* theta, double* grad, bool normalized, int total_edge_weights, const Heat_Kernel kernel) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= kernel.dim) return;
    
    double di = 0.0;
    double theta_i = theta[i];

    for (int j = 0; j < kernel.dim; ++j) {
        if (i == j) continue;
        double weight = qdist(i,j, kernel.;
        if (weight == 0) continue;
        di += 2 * weight * sin(theta_i - theta[j]);
    }
    double weight = dist_to_diagonal_grid(i, kernel.;
    di += 2 * weight * sin(theta[i]); 
    if (normalized) {
        di /= total_edge_weights;
    }
    grad[i] = di;
}

__global__ void multiply_vector(double* vector, double c, int dim) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= dim) return;

    vector[i] *= c;
}

__global__ void sample_theta (double *theta, curandState *states, int dim) {
    int i = threadIdx.x + blockDim.x * blockIdx.x; 
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

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= dim) return;

    curandState local_state = states[i];
    double drift = sigma * curr_grad[i];
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
    double bwd = (-d - sigma * curr_grad[i]) * (-d - sigma * curr_grad[i]);
    atomicAdd(q_fwd, fwd);
    atomicAdd(q_bwd, bwd);
}

void cuda_sample(double sigma, int burn_in, int thinning, bool tune, bool normalized, int total_edge_weights, int seed, Heat_Kernel& kernel) {
    const double OPTIMAL = 0.574;
    const int dim = kernel->dim;
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
    CUDA_CHECK(cudaMalloc(&rand_states, dim * sizeof(curandState)));

    const int THREADSPERBLOCK = 256;
    const int BLOCKSPERGRID = (dim + THREADSPERBLOCK - 1) / THREADSPERBLOCK; 

    setup_random_states<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(rand_states, seed, dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    sample_theta<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, rand_states, dim);
    add_op(ops_per_theta_sampling_);

    double *lambda_device, *curr_lambda_host, *prop_lambda_host;
    CUDA_CHECK(cudaMalloc(&lambda_device, sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&curr_lambda_host, sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&prop_lambda_host, sizeof(double)));
    
    double* q_device, q_host;
    CUDA_CHECK(cudaMalloc(&q_device, 2 * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&q_host, 2 * sizeof(double)));

    auto compute_grad = [&](double *theta, double *grad, double* lambda_device, double* lambda_host, double *U){
        CUDA_CHECK(cudaMemset(curr_lambda_device, 0, sizeof(double)));
        laplacian_symbol<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(theta, lambda_device, *kernel);
        grad_laplacian_symbol<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(theta, grad, normalized, total_edge_weights, *kernel);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(lambda_host, lambda_device, sizeof(double), cudaMemcpyDeviceToHost));
        if (normalized) {
            *lambda_device /= total_edge_weights; 
        }
        double dUdL = (kernel->t - kernel->s / (std::expm1(kernel->s * *lambda_host)));
        multiply_vector(grad, dUdL, dim);
        *U = kernel->t * *lambda_host - std::log1p(-std::exp(-kernel->s * *lambda_host));
        add_op(dim);
    };

    double curr_U;
    compute_grad(curr_theta, curr_grad, lambda_device, curr_lambda_host, &curr_U); 
    
    auto mala_pass = [&](bool tune) {
        drift_theta<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, prop_theta, rand_states, sigma, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        add_op(dim);

        double prop_U;
        compute_grad(prop_theta, prop_grad, lambda_device, prop_lambda_host, &prop_U);

        CUDA_CHECK(cudaMemset(q_device, 0, 2 * sizeof(double)));
        CUDA_CHECK(cudaMemset(q_host, 0, 2 * sizeof(double)));
        CUDA_CHECK(cudaDeviceSynchronize());

        compute_move_probabilities<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(curr_theta, prop_theta, curr_grad, prop_grad, q_device[0], q_device[1], sigma, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cuda_Memcpy(q_host, q_device, 2 * sizeof(double), cudaMemcpyDeviceToHost));
        add_op(dim);

        double alpha_log = (q_host[0] - q_host[1]) / (4 * sigma) - kernel->t * (*prop_lambda_host - *curr_lambda_host) + std::log1p(-std::exp(-kernel->s * *prop_lambda_host)) - std::log1p(-std::exp(-kernel->s * *curr_lambda_host));
        double alpha = std::min(1.0, std::exp(alpha_log));

        if (std::log(uniform_dist(gen)) < alpha_log) {
            CUDA_CHECK(cudaMemcpy(curr_theta, prop_theta, array_size, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(curr_grad, prop_grad, array_size, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(curr_lambda_host, prop_lambda_host, sizeof(double), cudaMemcpyHostToHost));
            curr_U = prop_U;
        }

        if (tune) {
            sigma *= std::exp(0.05 * (alpha - OPTIMAL));
            sigma = std::clamp(sigma, 1e-6, 0.5);
        }
    };

    for (int s = 0; s < burn_in; ++s) mala_pass(tune);

    for (int r = 0; r < kernel->R; ++r) {
        for (int s = 0; s < burn_in; ++s) mala_lass(false);
        CUDA_CHECK(cudaMemcpy(final_thetas[r * dim], curr_theta, array_size, cudaMemcpyDeviceToHost));
    }

    std::vector<double> final_thetas_vector(final_thetas, final_thetas + kernel.R * dim);
    kernel->thetas = final_thetas_vector;

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
    CUDA_CHECK(cudaFreeHost(q_device));
}
