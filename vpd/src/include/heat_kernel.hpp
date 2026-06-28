#pragma once

#include <array>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <random>
#include <optional>
#include <numbers>
#include <iostream>

#include <torch/torch.h>
#include <torch/extension.h>

class VPD;
class Heat_KernelBuilder;

enum class SamplingMethod {
  Rejection,
  Metropolis_Hastings,
  MALA
};

class Heat_Kernel {
private:
    int R;
    int s;
    double tau;
    int dim;
    int n;
    int seed;
    double resolution;
    int axis_dim;
    double mcmc_sigma;
    int mcmc_burn_in;
    int mcmc_iter;
    std::vector<double> thetas;
    std::vector<double> weights;

    int points_per_axis() const;
    std::array<double, 2> node_at(int index) const;
    double dist_to_diagonal_grid(const std::array<double, 2>& p) const;
    double qdist(const std::array<double, 2>& p1, const std::array<double, 2>& p2) const;
    double laplacian_symbol(const double* theta, int n, Heat_KernelBuilder* builder = nullptr) const;
    double delta_laplacian_symbol(const double* theta, int k, double proposed_val) const;
    void generate_weights_rejection(Heat_KernelBuilder* builder = nullptr);
    void generate_weights_metropolis_hastings(Heat_KernelBuilder* builder = nullptr);
    void generate_weights(SamplingMethod method, Heat_KernelBuilder* builder = nullptr);
    void init_base(int n, int axis_dim, double resolution, int R, double s, double tau, int seed);
    void init_dim();

    friend class VPD;
    friend class Heat_KernelBuilder;

    Heat_Kernel() = default;

public:
    /**
     * @brief Creates a new Heat_Kernel for persistent diagrams using rejection sampling.
     *
     * @param[in] n The dimensionality of the points on our persistent diagram.
     * @param[in] axis_dim The size of all axes.
     * @param[in] resolution The number of points between any two integers on a axis of our grid.
     * @param[in] R The number of samples to take.
     * @param[in] s The s value used for character ewight calculation. 
     * @param[in] tau The time value to use for the heat kernel computations
     * @param[in] seed (optional) A seed for reproducible randomness. Defaults to 42.
    */
    Heat_Kernel(int n, int axis_dim, double resolution, int R, double s, double tau, std::optional<uint32_t> seed = std::nullopt);
    /**
     * @brief Creates a new Heat_Kernel for persistent diagrams using rejection sampling.
     *
     * @param[in] n The dimensionality of the points on our persistent diagram.
     * @param[in] axis_dim The size of all axes.
     * @param[in] resolution The number of points between any two integers on a axis of our grid.
     * @param[in] R The number of samples to take.
     * @param[in] s The s value used for character ewight calculation. 
     * @param[in] tau The time value to use for the heat kernel computations
     * @param[in] method The sampling method used for theta generation.
     * @param[in] mcmc_sigma The step size used for Metropolis-Hastings sampling. 
     * @param[in] mcmc_burn_in The amount of unused iterations for Metropolis-Hastings sampling. 
     * @param[in] mcmc_iter The amount of regular iterations used for Metropolis-Hastings sampling. 
     * @param[in] seed (optional) A seed for reproducible randomness. Defaults to 42.
    */
    Heat_Kernel(int n, int axis_dim, double resolution, int R, double s, double tau, double mcmc_sigma, int mcmc_burn_in, int mcmc_iter, std::optional<uint32_t> seed = std::nullopt);
    Heat_Kernel(int n, int axis_dim, double resolution, int R, double s, double tau, const std::vector<double>& thetas, const std::vector<double>& weights);
    const std::vector<double>& get_thetas() const { return thetas; }
    const std::vector<double>& get_weights() const { return weights; }
};
