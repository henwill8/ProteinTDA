#include "heat_kernel.hpp"
#include "heat_kernel_builder.hpp"
#include "include/heat_kernel.hpp"
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

int Heat_Kernel::points_per_axis() const {
    // subtract one to avoid going over the edge of the grid
    // though perhaps we might want to switch it to including the edge since we are discluding (0, 0)?
    return this->axis_dim * static_cast<int>(this->resolution) - 1;
}

double Heat_Kernel::dist_to_diagonal_grid(const std::array<double, 2>& p) const {
    // Project p onto the diagonal (t, t)
    double t = 0.5 * (p[0] + p[1]);

    double min_t = 0.0;
    double max_t = points_per_axis() * this->resolution;

    // Find closest grid value to (t, t)
    double d_grid = std::round((t - min_t) * this->resolution) / this->resolution + min_t;
    // Clamp to grid range
    d_grid = std::clamp(d_grid, min_t, max_t);

    double dx = p[0] - d_grid;
    double dy = p[1] - d_grid;
    return std::sqrt(dx * dx + dy * dy);
}

// Quotient distance
double Heat_Kernel::qdist(const std::array<double, 2>& p1, const std::array<double, 2>& p2) const {
    const auto dx = p2[0] - p1[0];
    const auto dy = p2[1] - p1[1];
    const auto d_euclidean = std::sqrt(dx * dx + dy * dy);
    // const auto d_line = (std::abs(p1[1] - p1[0]) / std::numbers::sqrt2) + (std::abs(p2[1] - p2[0]) / std::numbers::sqrt2);
    const auto d_line = dist_to_diagonal_grid(p1) + dist_to_diagonal_grid(p2);
    return std::min(d_euclidean, d_line);
}

std::array<double, 2> Heat_Kernel::node_at(int index) const {
    if (this->n == 1) {
        return {(index + 1) / this->resolution};
    }

    const int iy = static_cast<int>((std::sqrt(8.0 * index + 1.0) - 1.0) / 2.0); // solution to iy(iy + 1) / 2 <= index
    const int ix = index - iy * (iy + 1) / 2; // checks how many nodes were in the previous rows n(n + 1) / 2
    return {ix / this->resolution, iy / this->resolution};
}

double Heat_Kernel::laplacian_symbol(const double* theta, int n, Heat_KernelBuilder* builder) const {
    const int progress_batch = builder != nullptr ? builder->progress_batch_ : Heat_KernelBuilder::DEFAULT_PROGRESS_BATCH;
    double result = 0.0;

#pragma omp parallel reduction(+ : result)
    {
        int local_completed = 0;

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                double edge_weight = qdist(node_at(i), node_at(j));
                if (edge_weight != 0.0) {
                    double diff = theta[i] - theta[j];
                    result += edge_weight * (1.0 - std::cos(diff));
                }
                if (builder != nullptr) {
                    ++local_completed;
                    if (local_completed >= progress_batch) {
                        builder->add_laplacian_ops(local_completed);
                        local_completed = 0;
                    }
                }
            }
            double edge_weight = dist_to_diagonal_grid(node_at(i));
            result += edge_weight * (1.0 - std::cos(theta[i]));
        }

        if (builder != nullptr && local_completed > 0) {
            builder->add_laplacian_ops(local_completed);
        }
    }
    return result;
}

double Heat_Kernel::delta_laplacian_symbol(const double* theta, int k, double proposed_val) const {
  const auto k_node = node_at(k);
  double current_val = theta[k];
  double delta = 0;

  for (int i = 0; i < this->dim; ++i) {
    if (i == k) continue;
    double weight = qdist(k_node, node_at(i));
    if (weight == 0) continue;
    delta += 2 * weight * (std::cos(current_val - theta[i]) - std::cos(proposed_val - theta[i]));
  }

  double weight = dist_to_diagonal_grid(k_node);
  delta += weight * (std::cos(current_val) - std::cos(proposed_val));

  return delta;
}

void Heat_Kernel::generate_weights_rejection(Heat_KernelBuilder* builder) {
    const double TWO_PI = 2.0 * std::numbers::pi;
    const int total = this->R * this->dim;
    std::vector<double> total_thetas(total);
    std::vector<double> weights(this->R);

    std::mt19937 gen(static_cast<uint32_t>(this->seed));
    std::uniform_real_distribution<double> theta_dist(0.0, TWO_PI);
    std::uniform_real_distribution<double> acceptance_dist(0.0, 1.0);

    std::vector<double> thetas(this->dim);

    for (int r = 0; r < this->R; ++r) {
        for (;;) {
            for (int j = 0; j < this->dim; ++j) {
                thetas[j] = theta_dist(gen);
            }
            if (builder != nullptr) builder->add_theta_sampling_ops();

            double lambda = laplacian_symbol(thetas.data(), this->dim, builder);
            double weight = std::exp(-this->tau * lambda) * (1 - std::exp(-this->s * lambda));
            if (acceptance_dist(gen) <= weight) {
                weights[r] = weight;
                if (builder != nullptr) builder->accept_attempt();
                break;
            }
            if (builder != nullptr) builder->reject_attempt();
        }

        std::copy(thetas.begin(), thetas.end(), total_thetas.begin() + r * this->dim);
    }
    this->thetas = total_thetas;
    this->weights = weights;
}

void Heat_Kernel::generate_weights_metropolis_hastings(Heat_KernelBuilder* builder) {
  const double TWO_PI = 2.0 * std::numbers::pi;
  const int total = this->R * this->dim;

  std::mt19937 gen(static_cast<uint32_t>(this->seed));
  std::uniform_real_distribution<double> theta_dist(0.0, TWO_PI);
  std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

  std::vector<double> theta(this->dim);
  for (int j = 0; j < this->dim; ++j) {
    theta[j] = theta_dist(gen);
  }

  double curr_lambda = laplacian_symbol(theta.data(), this->dim, builder);

  std::vector<double> total_thetas(total);
  std::vector<double> weights(this->R, 1.0);

  auto mcmc_pass = [&]() {
    for (int k = 0; k < this->dim; ++k) {
      double prop = theta[k] + this->mcmc_sigma * (2 * uniform_dist(gen) - 1);
      prop = std::fmod(prop, TWO_PI);
      if (prop < 0.0) {
        prop += TWO_PI;
      }

      const double dL = delta_laplacian_symbol(theta.data(), k, prop);
      double next_lambda = curr_lambda + dL;

      double log_diff = -this->tau * dL + std::log1p(std::exp(-this->s * next_lambda)) - std::log1p(std::exp(-this->s * curr_lambda));
      if (std::log(uniform_dist(gen) > log_diff)) {
        theta[k] = prop;
        curr_lambda = next_lambda;
      }
    }
  };

  for (int s = 0; s < this->mcmc_burn_in; ++s) mcmc_pass();

  for (int r = 0; r < this->R; ++r) { 
    for (int s = 0; s < this->mcmc_iter; ++s) mcmc_pass();
    std::copy(theta.begin(), theta.end(), total_thetas.begin() + r * this->dim);
  }
  this->thetas = total_thetas;
  this->weights = weights;
}

void Heat_Kernel::generate_weights(SamplingMethod method, Heat_KernelBuilder* builder) {
  switch(method) {
    case SamplingMethod::Rejection:
      generate_weights_rejection(builder);
      break;
    case SamplingMethod::Metropolis_Hastings:
      generate_weights_metropolis_hastings(builder);
      break;
    case SamplingMethod::MALA:
      break;
  }
}

void Heat_Kernel::init_dim() {
    const int ppa = points_per_axis();
    if (this->n == 1) {
        this->dim = ppa;
    } else if (this->n == 2) {
        this->dim = ppa * (ppa + 1) / 2;
    } else {
        throw std::invalid_argument("n must be 1 or 2");
    }
}

void Heat_Kernel::init_base(int n, int axis_dim, double resolution, int R, double s, double tau, int seed) {
    this->n = n;
    this->R = R;
    this->s = s;
    this->tau = tau;
    this->resolution = resolution;
    this->axis_dim = axis_dim;
    this->seed = seed;
    init_dim();
}

Heat_Kernel::Heat_Kernel(int n, int axis_dim, double resolution, int R, double s, double tau, std::optional<uint32_t> seed) {
    init_base(n, axis_dim, resolution, R, s, tau, seed.value_or(42));
    double qdists = 0.0;
    double count = 0.0;
    for (int i = 0; i < resolution * axis_dim; ++i) {
        for (int64_t j = 0; j < resolution * axis_dim; ++j) {
          double dist = qdist(node_at(i), node_at(j));
          qdists += dist;
          count++;
        }
    }
    std::cout << "AVERAGE qdist" << qdists/count << std::endl;
    generate_weights(SamplingMethod::Rejection);
}

Heat_Kernel::Heat_Kernel(int n, int axis_dim, double resolution, int R, double s, double tau, double mcmc_sigma, int mcmc_burn_in, int mcmc_iter, std::optional<uint32_t> seed)
{
    init_base(n, axis_dim, resolution, s, R, tau, seed.value_or(42));
    this->mcmc_sigma = mcmc_sigma;
    this->mcmc_burn_in = mcmc_burn_in;
    this->mcmc_iter = mcmc_iter;
    generate_weights(SamplingMethod::Metropolis_Hastings);
}

Heat_Kernel::Heat_Kernel(int n, int axis_dim, double resolution, int R, double s, double tau, const std::vector<double>& thetas, const std::vector<double>& weights) {
    init_base(n, axis_dim, resolution, R, s, tau, 0);
    this->thetas = thetas;
    this->weights = weights;
}
