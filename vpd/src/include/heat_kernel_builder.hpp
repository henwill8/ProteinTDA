#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "heat_kernel.hpp"

class Heat_KernelBuilder {
public:
    enum class Phase {
        Init,
        Thetas,
        Lambdas,
        Done,
    };

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

    int total_thetas_{0};
    int total_lambdas_{0};
    int64_t ops_per_lambda_{0};
    int64_t total_ops_{0};
    std::atomic<int64_t> completed_ops_{0};
    std::atomic<int> thetas_completed_{0};
    std::atomic<int> lambdas_completed_{0};
    std::atomic<Phase> phase_{Phase::Init};

    friend class Heat_Kernel;

    void reset_progress(int dim);
    void set_phase(Phase phase);
    void add_theta_ops(int count);
    void add_laplacian_ops(int count);
    void add_lambda_completed(int count);

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
    int64_t total_ops() const { return total_ops_; }
    int thetas_completed() const;
    int lambdas_completed() const;
    int total_thetas() const { return total_thetas_; }
    int total_lambdas() const { return total_lambdas_; }
    double fraction() const;
    bool done() const;
    std::string phase() const;
};
