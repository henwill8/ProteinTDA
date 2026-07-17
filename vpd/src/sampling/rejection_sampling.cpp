#include "rejection_sampling.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

void RejectionSampling::reset_progress() {
    SamplingMethod::reset_progress();
    ops_per_attempt_ = ops_per_theta_sampling_ + ops_per_laplacian_;
    committed_ops_.store(0, std::memory_order_relaxed);
    attempts_completed_.store(0, std::memory_order_relaxed);
    update_total_ops();
}

void RejectionSampling::reject_attempt() {
    committed_ops_.store(completed_ops(), std::memory_order_relaxed);
    attempts_completed_.fetch_add(1, std::memory_order_relaxed);
    update_total_ops();
}

void RejectionSampling::accept_attempt() {
    weights_completed_.fetch_add(1, std::memory_order_relaxed);
    attempts_completed_.fetch_add(1, std::memory_order_relaxed);
    committed_ops_.store(completed_ops(), std::memory_order_relaxed);
    update_total_ops();
}

void RejectionSampling::update_total_ops() {
    const int64_t committed = committed_ops_.load(std::memory_order_relaxed);
    const int remaining = total_weights_ - weights_completed();
    if (remaining <= 0) {
        set_total_ops(completed_ops());
        return;
    }

    const int attempts = attempts_completed();
    if (attempts <= 0) {
        set_total_ops(committed + static_cast<int64_t>(total_weights_) * ops_per_attempt_);
        return;
    }

    const double rate = acceptance_rate();
    if (rate <= 0.0) {
        set_total_ops(std::numeric_limits<int64_t>::max());
        return;
    }

    const int64_t estimated_remaining = static_cast<int64_t>(std::ceil(
        static_cast<double>(remaining) * static_cast<double>(ops_per_attempt_) / rate));
    set_total_ops(committed + estimated_remaining);
}

void RejectionSampling::cpu_sample() {
    const int total = kernel->R * kernel->dim;
    std::vector<double> total_thetas(total);
    std::vector<double> weights(kernel->R);

    std::mt19937 gen(static_cast<uint32_t>(this->seed));
    std::uniform_real_distribution<double> acceptance_dist(0.0, 1.0);

    std::vector<double> curr_thetas(kernel->dim);

    for (int r = 0; r < kernel->R; ++r) {
        for (;;) {
            sample_thetas(curr_thetas, gen);

            double lambda = laplacian_symbol(curr_thetas.data());
            double weight = std::exp(-kernel->t * lambda) * (1 - std::exp(-kernel->s * lambda));
            if (acceptance_dist(gen) <= weight) {
                weights[r] = weight;
                accept_attempt();
                break;
            }
            reject_attempt();
        }

        std::copy(curr_thetas.begin(), curr_thetas.end(), total_thetas.begin() + r * kernel->dim);
    }

    kernel->thetas = std::move(total_thetas);
    kernel->weights = std::move(weights);
}

void RejectionSampling::sample() {
#ifdef VPD_WITH_CUDA
    switch(this->device) {
        case Device::CPU:
            cpu_sample();
            break;
        case Device::CUDA:
            Heat_Kernel_device cuda_kernel = Heat_Kernel_device{
                kernel->n,
                kernel->axis_dim,
                kernel->ppa,
                kernel->resolution,
                kernel->R,
                kernel->s,
                kernel->t,
                kernel->dim
            };
            if (this->normalized_lambdas) {
                int edge_weight_total = this->edge_weight_total; 
            } else { 
                int edge_weight_total = 0;
            }
            kernel->thetas = cuda_sample(this->normalized_lambdas, edge_weight_total, this->seed, cuda_kernel, *this);
            break;
    }
#else
    cpu_sample();
#endif
}

int RejectionSampling::attempts_completed() const {
    return attempts_completed_.load(std::memory_order_relaxed);
}

double RejectionSampling::acceptance_rate() const {
    const int attempts = attempts_completed();
    if (attempts <= 0) {
        return 0.0;
    }
    return static_cast<double>(weights_completed()) / static_cast<double>(attempts);
}

std::string RejectionSampling::progress_postfix() const {
    std::ostringstream oss;
    oss << SamplingMethod::progress_postfix();
    oss << ", a=";
    if (attempts_completed() <= 0) {
        oss << "n/a";
    } else {
        oss << std::fixed << std::setprecision(1) << (acceptance_rate() * 100.0) << "%";
    }
    return oss.str();
}
