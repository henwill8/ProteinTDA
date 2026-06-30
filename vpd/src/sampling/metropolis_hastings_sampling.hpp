#pragma once

#include "sampling_method.hpp"

/**
 * @brief Metropolis-Hastings sampling for heat kernel theta generation.
 */
class MetropolisHastingsSampling : public SamplingMethod {
private:
    double mcmc_sigma;
    int mcmc_burn_in;
    int mcmc_thinning;

    void reset_progress() override;
    void sample() override;

public:
    /**
     * @brief Creates a new Heat_Kernel for persistent diagrams using Metropolis-Hastings sampling.
     *
     * @param[in] kernel The heat kernel to sample thetas and compute weights for.
     * @param[in] mcmc_sigma The step size used for Metropolis-Hastings sampling.
     * @param[in] mcmc_burn_in The amount of unused iterations for Metropolis-Hastings sampling.
     * @param[in] mcmc_thinning The amount of iterations used for Metropolis-Hastings thinning.
     * @param[in] seed (optional) A seed for reproducible randomness. Defaults to 42.
     */
    MetropolisHastingsSampling(
        std::shared_ptr<Heat_Kernel> kernel,
        double mcmc_sigma,
        int mcmc_burn_in,
        int mcmc_thinning,
        std::optional<uint32_t> seed = std::nullopt);
};
