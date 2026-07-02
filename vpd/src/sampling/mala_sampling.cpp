#include "mala_sampling.hpp"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>

void MALASampling::reset_progress() {
   SamplingMethod::reset_progress();
}

void MALASampling::sample() {
    const double TWO_PI = 2.0 * std::numbers::pi;
    const int total = kernel->R * kernel->dim;
    
    std::mt19937 gen(static_cast<uint32_t>(this->seed));
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::normal_distribution<double> gaussian(0.0,1.0);

    auto wrap_2pi = [&](double x) {
        x = std::fmod(x, TWO_PI); if (x < 0) x += TWO_PI; return x;
    };

    auto wrap_pi = [&](double x) {
        x = std::fmod(x, std::numbers::pi); if (x < 0) x += std::numbers::pi; return x;
    };

    std::vector<double> curr_thetas(kernel->dim);
    sample_thetas(curr_thetas, gen);

    auto compute_grad = [&](double *grad_U) {
        double curr_lambda = laplacian_symbol(curr_thetas.data());
        grad_laplacian_symbol(curr_thetas.data(), grad_U);
        double dUdL = 2 * (kernel->t + kernel->s/(std::expm1(kernel->s * curr_lambda)));
        for (int j = 0; j < kernel->dim; ++j) grad_U[j] *= dUdL;
        const double U = kernel->t * curr_lambda - std::log1p(-std::exp(-kernel->s * curr_lambda));
        return std::make_pair(U, curr_lambda);
    };
    
    std::vector<double> curr_grad(kernel->dim);
    auto [curr_U, curr_lambda] = compute_grad(curr_grad.data());

    std::vector<double> total_thetas(total);
    std::vector<double> weights(kernel->R, 1.0);

    const double OPTIMAL = 0.574;

    auto mala_pass = [&](bool tune) {
        std::vector<double> prop(kernel->dim);
        for (int i = 0; i < kernel->dim; ++i) {
            double drift = this->mala_sigma * curr_grad[i];
            double brownian = std::sqrt(2 * this->mala_sigma) * gaussian(gen);
            prop[i] = wrap_2pi(curr_thetas[i] + drift + brownian);
        }
        std::vector<double> prop_grad(kernel->dim);
        auto [prop_U, prop_lambda] = compute_grad(prop_grad.data());
        double q_fwd = 0.0;
        double q_bwd = 0.0;
        for (int i = 0; i < kernel->dim; ++i) {
            double d = wrap_pi(curr_thetas[i] - prop[i]);
            q_fwd += (-d - this->mala_sigma * curr_grad[i]) * (-d  -this->mala_sigma * curr_grad[i]);
            q_bwd += (d - this->mala_sigma * prop_grad[i]) * (d  -this->mala_sigma * prop_grad[i]);
        }
        double alpha_log =  (q_fwd - q_bwd) / 4*(this->mala_sigma) - kernel->t * (prop_lambda - curr_lambda) + std::log1p(-std::exp(-kernel->s * prop_lambda)) - std::log1p(-std::exp(-kernel->s * curr_lambda));
        double alpha = std::min(1.0, std::exp(alpha_log));
        if (std::log(uniform_dist(gen)) < alpha_log) {
            curr_thetas.swap(prop);
            curr_grad.swap(prop_grad);
            curr_U = prop_U;
            curr_lambda = prop_lambda;
        }

        if (tune) {
            this->mala_sigma *= std::exp(0.05 * (alpha - OPTIMAL));
            this->mala_sigma = std::clamp(this->mala_sigma, 1e-6, 0.5);
        }
    };

    for (int s = 0; s < this->mala_burn_in; ++s) mala_pass(this->tune_sigma);

    for (int r = 0; r < kernel->R; ++r) {
        for (int s = 0; s < this->mala_thinning; ++s) mala_pass(false);
        std::copy(curr_thetas.begin(), curr_thetas.end(), total_thetas.begin() + r * kernel->dim);
    }
    kernel->thetas = total_thetas;
}

MALASampling::MALASampling(
    double mala_sigma,
    int mala_burn_in,
    int mala_thinning,
    bool tune_sigma)
    : mala_sigma(mala_sigma),
      mala_burn_in(mala_burn_in),
      mala_thinning(mala_thinning),
      tune_sigma(tune_sigma) {}
