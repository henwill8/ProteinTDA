#pragma once

#include "sampling_method.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>

__device__ const double TWO_PI = 2.0*CUDART_PI;

__device__ inline double wrap_2pi(double x) {
    const double TWO_PI = 2.0 * CUDART_PI;
    x = fmod(x, TWO_PI);
    if (x < 0) x += TWO_PI;
    return x;
}

__device__ inline double wrap_pi(double x) {
    x = fmod(x, TWO_PI);
    if (x <= -CUDART_PI) x += TWO_PI;
    else if (x > CUDART_PI) x -= TWO_PI;
    return x;
}

__device__ inline void node_at(int index, double &x, double &y, const Heat_Kernel_device kernel) {
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

__device__ inline double dist_coords_to_diagonal_grid(double px, double py, const Heat_Kernel_device kernel) {
    double t = 0.5 * (px + py);

    double min_t = 0.0;
    double max_t = kernel.ppa * kernel.resolution;

    double d_grid = round((t - min_t) * kernel.resolution) / kernel.resolution + min_t; 
    d_grid =  fmax(min_t, fmin(d_grid, max_t));


    double dx = px - d_grid;
    double dy = py - d_grid;
    return sqrt(dx * dx + dy * dy);
}

__device__ inline double dist_to_diagonal_grid(int index, const Heat_Kernel_device kernel) {
    double px, py;
    node_at(index, px, py, kernel);
    return dist_coords_to_diagonal_grid(px, py, kernel);
}


__device__ inline double qdist(int i, int j, const Heat_Kernel_device kernel) {
    double p1_x, p1_y, p2_x, p2_y;
    node_at(i, p1_x, p1_y, kernel);
    node_at(j, p2_x, p2_y, kernel);

    const double dx = p2_x - p1_x;
    const double dy = p2_y - p1_y;

    const double d_euclidean = sqrt(dx * dx + dy * dy);
    const double d_line = dist_coords_to_diagonal_grid(p1_x, p1_y, kernel) + dist_coords_to_diagonal_grid(p2_x, p2_y, kernel);

    return fmin(d_euclidean, d_line);
}


__global__ void setup_random_states(curandState* state, int seed, int dim);

__global__ void sample_theta(double* theta, curandState* states, int dim);

__global__ void laplacian_symbol(const double* theta, double* lambda, const Heat_Kernel_device kernel);
