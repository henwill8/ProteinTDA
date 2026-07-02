#include "sampling_method.hpp"

#include <cmath>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif


void SamplingMethod::init(
    std::shared_ptr<Heat_Kernel> kernel,
    bool normalized_lambdas,
    int seed,
    Device device)
{
    this->kernel = std::move(kernel);
    this->normalized_lambdas = normalized_lambdas;
    this->seed = seed;
    this->device = device;
    if (normalized_lambdas) compute_total_edge_weights();
}

void SamplingMethod::compute_total_edge_weights() {
    double total = 0.0;
    for (int i = 0; i < kernel->dim; ++i) {
        for (int j = i; j < kernel->dim; ++j) {
            total += 2 * qdist(node_at(i), node_at(j));
        }
        total += 2 * dist_to_diagonal_grid(node_at(i));
    }
    this->edge_weight_total = total;
}

double SamplingMethod::dist_to_diagonal_grid(const std::array<double, 2>& p) const {
    // Project p onto the diagonal (t, t)
    double t = 0.5 * (p[0] + p[1]);

    double min_t = 0.0;
    double max_t = kernel->points_per_axis() * kernel->resolution;

    // Find closest grid value to (t, t)
    double d_grid = std::round((t - min_t) * kernel->resolution) / kernel->resolution + min_t;
    // Clamp to grid range
    d_grid = std::clamp(d_grid, min_t, max_t);

    double dx = p[0] - d_grid;
    double dy = p[1] - d_grid;
    return std::sqrt(dx * dx + dy * dy);
}

// Quotient distance
double SamplingMethod::qdist(const std::array<double, 2>& p1, const std::array<double, 2>& p2) const {
    const auto dx = p2[0] - p1[0];
    const auto dy = p2[1] - p1[1];
    const auto d_euclidean = std::sqrt(dx * dx + dy * dy);
    const auto d_line = dist_to_diagonal_grid(p1) + dist_to_diagonal_grid(p2);
    return std::min(d_euclidean, d_line);
}

std::array<double, 2> SamplingMethod::node_at(int index) const {
    if (kernel->n == 1) {
        return {(index + 1) / kernel->resolution};
    }

    const int iy = static_cast<int>((std::sqrt(8.0 * index + 1.0) - 1.0) / 2.0); // solution to iy(iy + 1) / 2 <= index
    const int ix = index - iy * (iy + 1) / 2; // checks how many nodes were in the previous rows n(n + 1) / 2
    return {ix / kernel->resolution, iy / kernel->resolution};
}

void SamplingMethod::sample_thetas(std::vector<double>& thetas, std::mt19937& gen) {
    const double TWO_PI = 2.0 * std::numbers::pi;
    std::uniform_real_distribution<double> theta_dist(0.0, TWO_PI);
    thetas.resize(kernel->dim);
    for (int j = 0; j < kernel->dim; ++j) {
        thetas[j] = theta_dist(gen);
        add_op();
    }
}

double SamplingMethod::laplacian_symbol(const double* theta) {
    double result = 0.0;
    const int n = kernel->dim;

#pragma omp parallel reduction(+ : result)
    {
#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            for (int64_t j = i + 1; j < n; ++j) {
                // TODO: we need to make sure the average distance in n=2 is correct (and double check n=1 again probably)
                double edge_weight = qdist(node_at(i), node_at(j));
                if (edge_weight != 0.0) {
                    double diff = theta[i] - theta[j];
                    result += 2 * edge_weight * (1.0 - std::cos(diff));
                }
                // If adding op everytime to global counter is too slow, switch to batched local counter
            }
            add_op(kernel->dim);
            double edge_weight = dist_to_diagonal_grid(node_at(i));
            result += 2 * edge_weight * (1.0 - std::cos(theta[i]));
            add_op();
        }
    }
    if (this->normalized_lambdas) result /= this->edge_weight_total;
    return result;
}

double SamplingMethod::delta_laplacian_symbol(const double* theta, int k, double proposed_val) {
    const auto k_node = node_at(k);
    double current_val = theta[k];
    double delta = 0;

    for (int i = 0; i < kernel->dim; ++i) {
        add_op();
        if (i == k) continue;
        double weight = qdist(k_node, node_at(i));
        if (weight == 0) continue;
        delta += 2 * weight * (std::cos(current_val - theta[i]) - std::cos(proposed_val - theta[i]));
    }

    double weight = dist_to_diagonal_grid(k_node);
    delta += 2 * weight * (std::cos(current_val) - std::cos(proposed_val));
    add_op();

    if (this->normalized_lambdas) delta /= this->edge_weight_total;
    return delta;
}

void SamplingMethod::grad_laplacian_symbol(const double* theta, double* grad) {
#pragma omp parallel 
    {
#pragma omp for schedule(static)
        for (int i = 0; i < kernel->dim; ++i) {
            double d_i = 0.0;
            for (int j = 0; j < kernel->dim; ++j) {
                if (i == j) continue;
                double weight = qdist(node_at(i), node_at(j));
                if (weight == 0) continue;
                d_i += 2 * weight * std::sin(theta[i] - theta[j]);
            }
            add_op(kernel->dim);
            add_op();
            double weight = dist_to_diagonal_grid(node_at(i));
            d_i += 2 * weight * std::sin(theta[i]);
            grad[i] = d_i;
        }
    } 
}

void SamplingMethod::reset_progress() {
    total_weights_ = kernel->R;
    ops_per_laplacian_ = static_cast<int64_t>(kernel->dim) * (kernel->dim + 1) / 2;
    ops_per_theta_sampling_ = kernel->dim;
    completed_ops_.store(0, std::memory_order_relaxed);
    weights_completed_.store(0, std::memory_order_relaxed);
    total_ops_.store(0, std::memory_order_relaxed);
}

std::shared_ptr<Heat_Kernel> SamplingMethod::build() {
    if (!kernel) {
        throw std::runtime_error("SamplingMethod::init must be called before build()");
    }
    reset_progress();
    sample();
    return kernel;
}

void SamplingMethod::set_total_ops(int64_t value) {
    total_ops_.store(value, std::memory_order_relaxed);
}

void SamplingMethod::add_op() {
    completed_ops_.fetch_add(1, std::memory_order_relaxed);
}

void SamplingMethod::add_op(int amount) {
    completed_ops_.fetch_add(amount, std::memory_order_relaxed);
}

int64_t SamplingMethod::completed_ops() const {
    return completed_ops_.load(std::memory_order_relaxed);
}

int64_t SamplingMethod::total_ops() const {
    return total_ops_.load(std::memory_order_relaxed);
}

int SamplingMethod::weights_completed() const {
    return weights_completed_.load(std::memory_order_relaxed);
}

std::string SamplingMethod::progress_postfix() const {
    std::ostringstream oss;
    oss << "w=" << weights_completed() << "/" << total_weights_;
    return oss.str();
}
