#include "mala_sampling.hpp"

#include <cmath>
#include <numbers>
#include <random>

void MALASampling::reset_progres() {
   SamplingMetod::reset_progress();
}

void MALASampling::sample() {
}

MALASampling::MALASampling(
    std::shared_ptr<Heat_Kernel> kernel,
    double mala_sigma,
    int mala_burn_in,
    int mala_thinning,
    bool tune_sigma,
    std::optional<uint32_t> seed)
    : SamplingMethod (
      std::move(kernel),
      static_cast<int>(seed.value_or(42))),
    mala_sigma(mala_sigma),
    mala_burn_in(mala_burn_in),
    mala_thinning(mala_thinning),
    tune_sigma(tune_sigma) {}
