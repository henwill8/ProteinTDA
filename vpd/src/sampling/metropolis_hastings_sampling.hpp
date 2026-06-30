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
     * @brief Creates a new MetropolisHastingsSampling object to sample thetas and compute weights for a heat kernel.
     *
     * @param[in] mcmc_sigma The step size used for Metropolis-Hastings sampling.
     * @param[in] mcmc_burn_in The amount of unused iterations for Metropolis-Hastings sampling.
     * @param[in] mcmc_thinning The amount of iterations used for Metropolis-Hastings thinning.
     */
    MetropolisHastingsSampling(double mcmc_sigma, int mcmc_burn_in, int mcmc_thinning);
};
