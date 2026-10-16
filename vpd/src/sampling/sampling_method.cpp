#include "complete_graph.hpp"
#include "graph_representation.hpp"
#include "lattice_graph.hpp"
#include "sampling_method.hpp"

#include <cmath>
#include <iostream>
#include <memory>
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
    Graph_Representation_Type graph_representation_type,
    Device device)
{
    this->kernel = std::move(kernel);
    this->seed = seed;
    this->normalized_lambdas = normalized_lambdas;
    switch (graph_representation_type) {
        case Graph_Representation_Type::COMPLETE:
            this->graph = std::make_unique<CompleteGraph>(*(this->kernel), normalized_lambdas);
            break;
        case Graph_Representation_Type::LATTICE:
            this->graph = std::make_unique<LatticeGraph>(*(this->kernel), normalized_lambdas);
            break;
    }
    this->edge_weight_total = this->graph->compute_total_edge_weights();
    this->device = device;
}

void SamplingMethod::sample_thetas(std::vector<double>& thetas, std::mt19937& gen) {
    const double TWO_PI = 2.0 * std::numbers::pi;
    std::uniform_real_distribution<double> theta_dist(0.0, TWO_PI);
    thetas.resize(kernel->dim);
    for (int j = 0; j < kernel->dim; ++j) {
        thetas[j] = theta_dist(gen);
    }
    add_op(kernel->dim);
}

void SamplingMethod::reset_progress() {
    total_weights_ = kernel->R;
    ops_per_laplacian_ = graph->ops_per_lambda();
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

void SamplingMethod::add_op(int64_t amount) {
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
