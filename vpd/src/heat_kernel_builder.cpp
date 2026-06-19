#include "heat_kernel_builder.hpp"

#include <stdexcept>

Heat_KernelBuilder::Heat_KernelBuilder(
    int n,
    int axis_dim,
    double resolution,
    int R,
    double tau,
    const std::optional<std::vector<int>>& mask,
    std::optional<uint32_t> seed,
    int progress_batch)
    : n(n),
      axis_dim(axis_dim),
      resolution(resolution),
      R(R),
      tau(tau),
      seed(static_cast<int>(seed.value_or(42))),
      progress_batch_(progress_batch < 1 ? DEFAULT_PROGRESS_BATCH : progress_batch) {}

void Heat_KernelBuilder::reset_progress(int dim) {
    total_thetas_ = this->R * dim;
    total_lambdas_ = this->R;
    ops_per_lambda_ = static_cast<int64_t>(dim) * (dim - 1) / 2;
    total_ops_ = static_cast<int64_t>(total_thetas_) + static_cast<int64_t>(total_lambdas_) * ops_per_lambda_;
    completed_ops_.store(0, std::memory_order_relaxed);
    thetas_completed_.store(0, std::memory_order_relaxed);
    lambdas_completed_.store(0, std::memory_order_relaxed);
    phase_.store(Phase::Init, std::memory_order_relaxed);
}

void Heat_KernelBuilder::set_phase(Phase phase) {
    phase_.store(phase, std::memory_order_relaxed);
}

void Heat_KernelBuilder::add_theta_ops(int count) {
    thetas_completed_.fetch_add(count, std::memory_order_relaxed);
    completed_ops_.fetch_add(count, std::memory_order_relaxed);
}

void Heat_KernelBuilder::add_laplacian_ops(int count) {
    completed_ops_.fetch_add(count, std::memory_order_relaxed);
}

void Heat_KernelBuilder::add_lambda_completed(int count) {
    lambdas_completed_.fetch_add(count, std::memory_order_relaxed);
}

int64_t Heat_KernelBuilder::completed_ops() const {
    return completed_ops_.load(std::memory_order_relaxed);
}

int Heat_KernelBuilder::thetas_completed() const {
    return thetas_completed_.load(std::memory_order_relaxed);
}

int Heat_KernelBuilder::lambdas_completed() const {
    return lambdas_completed_.load(std::memory_order_relaxed);
}

double Heat_KernelBuilder::fraction() const {
    if (total_ops_ <= 0) {
        return 1.0;
    }
    return static_cast<double>(completed_ops()) / static_cast<double>(total_ops_);
}

bool Heat_KernelBuilder::done() const {
    return completed_ops() >= total_ops_;
}

std::string Heat_KernelBuilder::phase() const {
    switch (phase_.load(std::memory_order_relaxed)) {
        case Phase::Thetas:
            return "thetas";
        case Phase::Lambdas:
            return "lambdas";
        case Phase::Done:
            return "done";
        case Phase::Init:
        default:
            return "init";
    }
}

void Heat_KernelBuilder::build() {
    auto built = std::shared_ptr<Heat_Kernel>(new Heat_Kernel());
    built->init_base(this->n, this->axis_dim, this->resolution, this->R, this->tau, this->seed);
    this->reset_progress(built->dim);

    this->set_phase(Phase::Thetas);
    built->thetas = built->generate_random_thetas(this);

    this->set_phase(Phase::Lambdas);
    built->lambdas = built->compute_lambdas(this);
    built->apply_tau();

    this->set_phase(Phase::Done);
    this->thetas_completed_.store(this->total_thetas_, std::memory_order_relaxed);
    this->lambdas_completed_.store(this->total_lambdas_, std::memory_order_relaxed);
    this->completed_ops_.store(this->total_ops_, std::memory_order_relaxed);
    this->kernel_ = std::move(built);
}

std::shared_ptr<Heat_Kernel> Heat_KernelBuilder::kernel() const {
    if (!this->kernel_) {
        throw std::runtime_error("Heat_KernelBuilder::kernel called before build()");
    }
    return this->kernel_;
}
