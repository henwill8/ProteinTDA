#pragma once

#include <array>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <random>
#include <optional>
#include <numbers>

#include <torch/torch.h>
#include <torch/extension.h>

class VPD;
class Heat_KernelBuilder;

class Heat_Kernel {
private:
  int R;
  double tau;
  int dim;
  int n;
  int seed;
  double resolution;
  int axis_dim;
  std::vector<double> thetas;
  std::vector<double> lambdas;
  std::vector<double> weights;

  int points_per_axis() const;
  std::array<double, 2> node_at(int index) const;
  double dist_to_diagonal_grid(const std::array<double, 2>& p) const;
  double qdist(const std::array<double, 2>& p1, const std::array<double, 2>& p2) const;
  double laplacian_symbol(const double* theta, int n, Heat_KernelBuilder* builder = nullptr) const;
  std::vector<double> generate_random_thetas(Heat_KernelBuilder* builder = nullptr);
  std::vector<double> compute_lambdas(Heat_KernelBuilder* builder = nullptr);
  void apply_tau();
  void init_base(int n, int axis_dim, double resolution, int R, double tau, int seed);
  void init_dim();

  friend class VPD;
  friend class Heat_KernelBuilder;

  Heat_Kernel() = default;

public:
  /**
   * @brief Creates a new Heat_Kernel for persistent diagrams.
   *
   * @param[in] n The dimensionality of the points on our persistent diagram.
   * @param[in] axis_dim The size of all axes.
   * @param[in] resolution The number of points between any two integers on a axis of our grid.
   * @param[in] R The number of samples to take.
   * @param[in] tau The temperature value to use for the heat kernel computations
   * @param[in] mask (optional) A mask to make some of the edges 0
   * @param[in] seed (optional) A seed for reproducible randomness. Defaults to 42.
   */
  Heat_Kernel(int n, int axis_dim, double resolution, int R, double tau, const std::optional<std::vector<int>>& mask = std::nullopt, std::optional<uint32_t> seed = std::nullopt);
  Heat_Kernel(int n, int axis_dim, double resolution, int R, double tau, const std::vector<double>& thetas, const std::vector<double>& lambdas);
  const std::vector<double>& get_thetas() const { return thetas; }
  const std::vector<double>& get_lambdas() const { return lambdas; }
  const std::vector<double>& get_weights() const { return weights; }
};
