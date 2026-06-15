#pragma once

#include <array>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <random>
#include <optional>
#include <numbers>

#include <torch/torch.h>
#include <torch/extension.h>

class Heat_RFF {
private:
  int R;
  double tau;
  int dim;
  int n;
  int seed;
  double resolution;
  int axis_dim;
  std::vector<double> thetas;
  std::vector<double> weights;

  std::array<double, 2> node_at(int index) const;
  double qdist(const std::array<double, 2>& p1, const std::array<double, 2>& p2);
  double laplacian_symbol(const std::vector<double>& theta, int n);
  std::vector<double> generate_random_thetas();
  std::vector<double> compute_theta_weights();
  torch::Tensor align_pd_to_grid(torch::Tensor pd);
  torch::Tensor pd_to_vpd(torch::Tensor pd);
  torch::Tensor pd_diff(torch::Tensor pd1, torch::Tensor pd2);
  void init_dim();

public:
  /** 
   * @brief Creates a new Heat_RFF for persistent diagrams.
   *
   * @param[in] n The dimensionality of the points on our persistent diagram. 
   * @param[in] axis_dim The size of all axes.
   * @param[in] resolution The number of points between any two integers on a axis of our grid. 
   * @param[in] R The number of samples to take.
   * @param[in] tau The temperature value to use for the heat kernel computations
   * @param[in] mask (optional) A mask to make some of the edges 0
   * @param[in] seed (optional) A seed for reproducible randomness. Defaults to 42. 
   */
  Heat_RFF(int n, int axis_dim, double resolution, int R, double tau, const std::optional<std::vector<int>>& mask = std::nullopt, std::optional<uint32_t> seed = std::nullopt);
  Heat_RFF(int n, int axis_dim, double resolution, int R, double tau, const std::vector<double>& thetas, const std::vector<double>& weights);
  const std::vector<double>& get_thetas() const { return thetas; }
  const std::vector<double>& get_weights() const { return weights; }
  /**
   * @brief Computes the loss between two persistent diagrams and returns it as a vector in 2R dimensional space.
   *
   * @param[in] pd1 The first persistent diagram.
   * @paran[in] pd2 The second persistent diagram
   */
  torch::Tensor vpd_loss_vector_(torch::Tensor pd1, torch::Tensor pd2);
  /**
   * @brief Computes the loss between two persistent diagrams and returns its L2 norm squared.
   *
   * @param[in] pd1 The first persistent diagram.
   * @param[in] pd2 The second persistent diagram.
   *
   */
  torch::Tensor vpd_loss(torch::Tensor pd1, torch::Tensor pd2);
};
