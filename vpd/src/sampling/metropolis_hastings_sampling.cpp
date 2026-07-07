#include "metropolis_hastings_sampling.hpp"

#include <cmath>
#include <numbers>
#include <random>

void MetropolisHastingsSampling::reset_progress() {
    SamplingMethod::reset_progress();
    const int64_t initial_ops = ops_per_theta_sampling_ + ops_per_laplacian_;
    const int64_t ops_per_mcmc_pass = static_cast<int64_t>(kernel->dim) * (kernel->dim + 1); // goes through all thetas, and each calls delta_laplacian_symbol
    const int64_t mcmc_passes = static_cast<int64_t>(mcmc_burn_in) + static_cast<int64_t>(kernel->R) * mcmc_thinning;
    set_total_ops(initial_ops + mcmc_passes * ops_per_mcmc_pass);
}

void MetropolisHastingsSampling::cpu_sample() {
    const double TWO_PI = 2.0 * std::numbers::pi;
    const int total = kernel->R * kernel->dim;

    std::mt19937 gen(static_cast<uint32_t>(this->seed));
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    std::vector<double> curr_thetas(kernel->dim);
    sample_thetas(curr_thetas, gen);

    double curr_lambda = laplacian_symbol(curr_thetas.data());

    std::vector<double> total_thetas(total);
    std::vector<double> weights(kernel->R, 1.0);

    auto mcmc_pass = [&]() {
        for (int k = 0; k < kernel->dim; ++k) {
            double prop = curr_thetas[k] + this->mcmc_sigma * (2 * uniform_dist(gen) - 1);
            prop = std::fmod(prop, TWO_PI);
            if (prop < 0.0) {
                prop += TWO_PI;
            }

            const double dL = delta_laplacian_symbol(curr_thetas.data(), k, prop);
            double next_lambda = curr_lambda + dL;

            double log_diff = -kernel->t * dL
                              + std::log1p(std::exp(-kernel->s * next_lambda))
                              - std::log1p(std::exp(-kernel->s * curr_lambda));

            if (std::log(uniform_dist(gen) > log_diff)) {
                curr_thetas[k] = prop;
                curr_lambda = next_lambda;
            }
        }
    };

    for (int step = 0; step < this->mcmc_burn_in; ++step) mcmc_pass();

    for (int r = 0; r < kernel->R; ++r) {
        for (int step = 0; step < this->mcmc_thinning; ++step) mcmc_pass();
        std::copy(curr_thetas.begin(), curr_thetas.end(), total_thetas.begin() + r * kernel->dim);
        double lambda = laplacian_symbol(curr_thetas.data());
        double weight = std::exp(-kernel->t * lambda) * (1 - std::exp(-kernel->s * lambda));
        weights[r] = weight;
        weights_completed_.fetch_add(1, std::memory_order_relaxed);
    }

    kernel->thetas = std::move(total_thetas);
    kernel->weights = std::move(weights);
}

void MetropolisHastingsSampling::sample() {
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
            kernel->thetas = cuda_sample(this->mcmc_sigma, this->mcmc_burn_in, this->mcmc_thinning, this->normalized_lambdas, edge_weight_total, this->seed, cuda_kernel, *this);
            break;
    }
}

MetropolisHastingsSampling::MetropolisHastingsSampling(
    double mcmc_sigma,
    int mcmc_burn_in,
    int mcmc_thinning)
    : mcmc_sigma(mcmc_sigma),
      mcmc_burn_in(mcmc_burn_in),
      mcmc_thinning(mcmc_thinning) {}
