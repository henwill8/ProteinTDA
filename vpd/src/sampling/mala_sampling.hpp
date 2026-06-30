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
     * @brief Creates a new Heat_Kernel for persistent diagrams using MALA sampling.
     *
     * @param[in] kernel The heat kernel to sample thetas and compute weights for.
     * @param[in] mala_sigma The step size used for MALA sampling.
     * @param[in] mala_burn_in The amount of unused iterations for MALA sampling.
     * @param[in] mala_thinning The amount of iterations used for Metropolis-Hastings thinning.
     * @param[in] tune_sigma Determines if the step size should be tuned towards an optimal acceptance rate of 0.574
     * @param[in] seed (optional) A seed for reproducible randomness. Defaults to 42.
     */
    MALASampling(
        std::shared_ptr<Heat_Kernel> kernel,
        double mala_sigma,
        int mala_burn_in,
        int mala_thinning,
        bool tune_sigma,
        std::optional<uint32_t> seed = std::nullopt);
};
