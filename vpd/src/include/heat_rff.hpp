#pragma once

#include <array>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <random>
#include <optional>
#include <numbers>

#include <torch/torch.h>
#include <torch/extensions.h>

class Heat_RFF {
private:
  int R;
  double tau;
  int dim;
  int n;
  int seed;
  double resolution;
  std::vector<double> edges;
  std::vector<double> thetas;
  std::vector<double> weights;

  double qdist(const std::array<double, 2>& p1, const std::array<double, 2>& p2);
  std::vector<double> generate_edge_weights(const std::optional<std::vector<int>>& mask);
  double laplacian_symbol(const std::vector<double>& theta, int n);
  std::vector<double> generate_random_thetas();
  std::vector<double> compute_theta_weights();

public:
  Heat_RFF(int n, int axis_dim, double resolution, int R, double tau, const std::optional<std::vector<int>>& mask = std::nullopt, std::optional<uint32_t> seed = std::nullopt);
};
