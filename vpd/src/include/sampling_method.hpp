#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "heat_kernel.hpp"

class SamplingMethod {
public:
    SamplingMethod() = default;
    virtual ~SamplingMethod() = default;

    void init(
        std::shared_ptr<Heat_Kernel> kernel,
        bool normalized_lambdas = true,
        int seed = 42);

    std::shared_ptr<Heat_Kernel> build();

    int64_t completed_ops() const;
    int64_t total_ops() const;
    int weights_completed() const;
    int total_weights() const { return total_weights_; }

    virtual std::string progress_postfix() const;

protected:
    std::shared_ptr<Heat_Kernel> kernel;
    bool normalized_lambdas;
    double edge_weight_total;
    int seed;

    int total_weights_{0};
    int64_t ops_per_laplacian_{0};
    int64_t ops_per_theta_sampling_{0};
    std::atomic<int64_t> completed_ops_{0};
    std::atomic<int64_t> total_ops_{0};
    std::atomic<int> weights_completed_{0};

    void compute_total_edge_weights();
    std::array<double, 2> node_at(int index) const;
    double dist_to_diagonal_grid(const std::array<double, 2>& p) const;
    double qdist(const std::array<double, 2>& p1, const std::array<double, 2>& p2) const;
    double laplacian_symbol(const double* theta);
    double delta_laplacian_symbol(const double* theta, int k, double proposed_val);
    void grad_laplacian_symbol(const double* theta, double* grad);

    virtual void reset_progress();
    void add_op();
    void set_total_ops(int64_t value);

    void sample_thetas(std::vector<double>& thetas, std::mt19937& gen);

    virtual void sample() = 0;
};
