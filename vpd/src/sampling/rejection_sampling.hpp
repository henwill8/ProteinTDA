#pragma once

#include "sampling_method.hpp"

/**
 * @brief Rejection sampling for heat kernel weight generation.
 */
class RejectionSampling : public SamplingMethod {
private:
    int64_t ops_per_attempt_{0};
    std::atomic<int64_t> committed_ops_{0};
    std::atomic<int> attempts_completed_{0};

    void reset_progress() override;
    void sample() override;
    void update_total_ops();
    void reject_attempt();
    void accept_attempt();

public:
    int attempts_completed() const;
    double acceptance_rate() const;
    std::string progress_postfix() const override;
};
