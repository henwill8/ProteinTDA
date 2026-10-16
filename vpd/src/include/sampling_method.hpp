#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "graph_representation.hpp"
#include "heat_kernel.hpp"

enum class Device {
    CPU,
    CUDA
};

enum class Graph_Representation_Type {
    COMPLETE,
    LATTICE
};

struct Heat_Kernel_device {
    int n;
    int axis_dim;
    int ppa;
    double resolution;
    int R; 
    double s;
    double t;
    int dim;
};

class SamplingMethod {
public:
    SamplingMethod() = default;
    virtual ~SamplingMethod() = default;

    void init(
        std::shared_ptr<Heat_Kernel> kernel,
        bool normalized_lambdas = true,
        int seed = 42,
        Graph_Representation_Type graph_representation_type = Graph_Representation_Type::LATTICE,
        Device device = Device::CPU);

    std::shared_ptr<Heat_Kernel> build();

    int64_t completed_ops() const;
    int64_t total_ops() const;
    int weights_completed() const;
    int total_weights() const { return total_weights_; }

    virtual std::string progress_postfix() const;

protected:
    std::shared_ptr<Heat_Kernel> kernel;
    std::unique_ptr<GraphRepresentation> graph;
    int seed;
    Device device;
    bool normalized_lambdas = true;
    double edge_weight_total = 0.0;

    int total_weights_{0};
    std::atomic<int64_t> completed_ops_{0};
    std::atomic<int64_t> total_ops_{0};
    std::atomic<int> weights_completed_{0};

    void compute_total_edge_weights() {graph->compute_total_edge_weights(); }
    double laplacian_symbol(const double* theta) { return graph->laplacian_symbol(theta); }
    double delta_laplacian_symbol(const double* theta, int k, double proposed_val) { return graph->delta_laplacian_symbol(theta, k, proposed_val); }
    void grad_laplacian_symbol(const double* theta, double* grad) {graph->grad_laplacian_symbol(theta, grad); }

    virtual void reset_progress();
    void set_total_ops(int64_t value);

    void sample_thetas(std::vector<double>& thetas, std::mt19937& gen);

    virtual void sample() = 0;

public:
    void add_op();
    void add_op(int amount);
    void add_op(int64_t amount);
    int64_t ops_per_laplacian_{0};
    int64_t ops_per_theta_sampling_{0};
};
