#include "sampling_common.cuh"

__global__ void setup_random_states(curandState* state, int seed, int dim) {
    int i = threadIdx.x + blockDim.x * blockIdx.x; 

    if (i >= dim) return;

    curand_init(seed, i, 0, &state[i]);
}

__global__ void sample_theta (double *theta, curandState *states, int dim) {
    int i = threadIdx.x + blockDim.x * blockIdx.x; 
    if (i >= dim) return;

    curandState local_state = states[i];
    theta[i] = curand_uniform(&local_state) * 2 *CUDART_PI;
    states[i] = local_state;
}


__global__ void laplacian_symbol(const double* theta, double* lambda, const Heat_Kernel_device kernel) {
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
