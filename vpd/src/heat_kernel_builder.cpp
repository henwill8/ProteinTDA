#include "heat_kernel_builder.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

Heat_KernelBuilder::Heat_KernelBuilder(
    int n,
    int axis_dim,
    double resolution,
    int R,
    double tau,
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
    total_weights_ = this->R;
    ops_per_laplacian_ = static_cast<int64_t>(dim) * (dim - 1) / 2;
    ops_per_theta_sampling_ = dim;
    ops_per_attempt_ = ops_per_theta_sampling_ + ops_per_laplacian_;
    completed_ops_.store(0, std::memory_order_relaxed);
    committed_ops_.store(0, std::memory_order_relaxed);
    weights_completed_.store(0, std::memory_order_relaxed);
    attempts_completed_.store(0, std::memory_order_relaxed);
}

void Heat_KernelBuilder::add_theta_sampling_ops() {
    completed_ops_.fetch_add(ops_per_theta_sampling_, std::memory_order_relaxed);
}

void Heat_KernelBuilder::add_laplacian_ops(int count) {
    completed_ops_.fetch_add(count, std::memory_order_relaxed);
}

void Heat_KernelBuilder::reject_attempt() {
    committed_ops_.store(completed_ops_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    attempts_completed_.fetch_add(1, std::memory_order_relaxed);
}

void Heat_KernelBuilder::accept_attempt() {
    weights_completed_.fetch_add(1, std::memory_order_relaxed);
    attempts_completed_.fetch_add(1, std::memory_order_relaxed);
    committed_ops_.store(completed_ops_.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

int64_t Heat_KernelBuilder::completed_ops() const {
    return completed_ops_.load(std::memory_order_relaxed);
}

int64_t Heat_KernelBuilder::estimated_total_ops() const {
    const int64_t committed = committed_ops_.load(std::memory_order_relaxed);
    const int remaining = total_weights_ - weights_completed();
    if (remaining <= 0) {
        return completed_ops();
    }

    const int attempts = attempts_completed();
    if (attempts <= 0) {
        return committed + static_cast<int64_t>(total_weights_) * ops_per_attempt_;
    }

    const double rate = acceptance_rate();
    if (rate <= 0.0) {
        return std::numeric_limits<int64_t>::max();
    }

    const int64_t estimated_remaining = static_cast<int64_t>(std::ceil(
        static_cast<double>(remaining) * static_cast<double>(ops_per_attempt_) / rate));
    return committed + estimated_remaining;
}

int64_t Heat_KernelBuilder::total_ops() const {
    return estimated_total_ops();
}

int Heat_KernelBuilder::weights_completed() const {
    return weights_completed_.load(std::memory_order_relaxed);
}

int Heat_KernelBuilder::attempts_completed() const {
    return attempts_completed_.load(std::memory_order_relaxed);
}

double Heat_KernelBuilder::acceptance_rate() const {
    const int attempts = attempts_completed();
    if (attempts <= 0) {
        return 0.0;
    }
    return static_cast<double>(weights_completed()) / static_cast<double>(attempts);
}

void Heat_KernelBuilder::build() {
    auto built = std::shared_ptr<Heat_Kernel>(new Heat_Kernel());
    built->init_base(this->n, this->axis_dim, this->resolution, this->R, this->tau, this->seed);
    this->reset_progress(built->dim);
    built->generate_weights(this);
    this->kernel_ = std::move(built);
}

std::shared_ptr<Heat_Kernel> Heat_KernelBuilder::kernel() const {
    if (!this->kernel_) {
        throw std::runtime_error("Heat_KernelBuilder::kernel called before build()");
    }
    return this->kernel_;
}
