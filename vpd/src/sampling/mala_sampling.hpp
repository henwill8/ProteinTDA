#pragma once

#include "sampling_method.hpp"

/**
 * @brief Metropolis Adjusted Langevin Algorithm for heat kernel theta generation.
 */
class MALASampling : public SamplingMethod {
private:
    double mala_sigma;
    int mala_burn_in;
    int mala_iter;
    bool tune_sigma;

    void reset_progress() override;
    void sample() override;

public:
    /**
     * @brief Creates a new MALASampling object to sample thetas and compute weights for a heat kernel.
     *
     * @param[in] mala_sigma The step size used for MALA sampling.
     * @param[in] mala_burn_in The amount of unused iterations for MALA sampling.
     * @param[in] mala_thinning The amount of iterations used for MALA thinning.
     * @param[in] tune_sigma Whether to tune the step size during sampling.
     */
    MALASampling(double mala_sigma, int mala_burn_in, int mala_thinning, bool tune_sigma);
};
