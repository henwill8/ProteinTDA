#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "heat_kernel.hpp"

class Heat_KernelBuilder {
public:
    static constexpr int DEFAULT_PROGRESS_BATCH = 100;

private:
    int n;
    int axis_dim;
    double resolution;
    int R;
    double tau;
    int seed;
    int progress_batch_;
    std::shared_ptr<Heat_Kernel> kernel_;

    int total_weights_{0};
    int64_t ops_per_laplacian_{0};
    int64_t ops_per_theta_sampling_{0};
    int64_t ops_per_attempt_{0};
    std::atomic<int64_t> completed_ops_{0};
    std::atomic<int> weights_completed_{0};
    std::atomic<int> attempts_completed_{0};

    friend class Heat_Kernel;

    void reset_progress(int dim);
    void add_theta_sampling_ops();
    void add_laplacian_ops(int count);
    void rollback_attempt();
    void accept_attempt();
    int64_t estimated_total_ops() const;

public:
    Heat_KernelBuilder(
        int n,
        int axis_dim,
        double resolution,
        int R,
        double tau,
        const std::optional<std::vector<int>>& mask = std::nullopt,
        std::optional<uint32_t> seed = std::nullopt,
        int progress_batch = DEFAULT_PROGRESS_BATCH);

    void build();
    std::shared_ptr<Heat_Kernel> kernel() const;

    int64_t completed_ops() const;
    int64_t total_ops() const;
    int weights_completed() const;
    int attempts_completed() const;
    double acceptance_rate() const;
    int total_weights() const { return total_weights_; }
};
