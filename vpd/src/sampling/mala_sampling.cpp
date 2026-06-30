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
    double mala_sigma,
    int mala_burn_in,
    int mala_thinning,
    bool tune_sigma)
    : mala_sigma(mala_sigma),
      mala_burn_in(mala_burn_in),
      mala_thinning(mala_thinning),
      tune_sigma(tune_sigma) {}
