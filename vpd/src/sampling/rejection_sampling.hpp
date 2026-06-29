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
    void on_progress_update() override;
    void sample() override;
    void update_total_ops();
    void reject_attempt();
    void accept_attempt();

public:
    /**
     * @brief Creates a new Heat_Kernel for persistent diagrams using rejection sampling.
     *
     * @param[in] kernel The heat kernel to sample thetas and compute weights for.
     * @param[in] seed (optional) A seed for reproducible randomness. Defaults to 42.
     * @param[in] progress_batch Batch size for weight creation progress updates.
     */
    RejectionSampling(
        std::shared_ptr<Heat_Kernel> kernel,
        std::optional<uint32_t> seed = std::nullopt,
        int progress_batch = DEFAULT_PROGRESS_BATCH);

    int attempts_completed() const;
    double acceptance_rate() const;
    std::string progress_postfix() const override;
};
