#include "heat_kernel.hpp"
#include "heat_kernel_builder.hpp"

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
    double d_grid = std::round((t - min_t) / this->resolution) * this->resolution + min_t;
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
        return {(index + 1) * this->resolution};
    }

    const int iy = static_cast<int>((std::sqrt(8.0 * index + 1.0) - 1.0) / 2.0); // solution to iy(iy + 1) / 2 <= index
    const int ix = index - iy * (iy + 1) / 2; // checks how many nodes were in the previous rows n(n + 1) / 2
    return {ix * this->resolution, iy * this->resolution};
}

double Heat_Kernel::laplacian_symbol(const double* theta, int n, Heat_KernelBuilder* builder) const {
    const int progress_batch = builder != nullptr ? builder->progress_batch_ : Heat_KernelBuilder::DEFAULT_PROGRESS_BATCH;
    double result = 0.0;

#pragma omp parallel reduction(+ : result)
    {
        int local_completed = 0;

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            for (int64_t j = i + 1; j < n; ++j) {
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
        }

        if (builder != nullptr && local_completed > 0) {
            builder->add_laplacian_ops(local_completed);
        }
    }
    return result;
}

// Returns theta of given dimension
std::vector<double> Heat_Kernel::generate_random_thetas(Heat_KernelBuilder* builder) {
    const double TWO_PI = 2.0 * std::numbers::pi;
    const int total = this->R * this->dim;
    const int progress_batch = builder != nullptr ? builder->progress_batch_ : Heat_KernelBuilder::DEFAULT_PROGRESS_BATCH;
    std::vector<double> thetas(total);

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        std::mt19937 gen(static_cast<uint32_t>(this->seed + tid)); // each thread needs a different generator
        std::uniform_real_distribution<double> theta_dist(0.0, TWO_PI);
        std::uniform_real_distribution<double> acceptance_dist(0.0, 1.0);

        std::vector<double> theta(this->dim);
        int local_completed = 0;

#pragma omp for schedule(dynamic)
        for (int r = 0; r < this->R; ++r) {
          for (;;) {
            for (int j = 0; j < this->dim; ++j) {
              theta[j] = theta_dist(gen);
            }

            double lambda = laplacian_symbol(theta.data(), this->dim, builder);
            double weight = std::exp(-this->tau * lambda);
            if (acceptance_dist(gen) <= weight) break;
          }

          std::copy(theta.begin(), theta.end(), thetas.begin() + r * this->dim);

          if (builder != nullptr) {
                ++local_completed;
                if (local_completed >= progress_batch) {
                    builder->add_theta_ops(local_completed);
                    local_completed = 0;
                }
            }
        }

        if (builder != nullptr && local_completed > 0) {
            builder->add_theta_ops(local_completed);
        }
    }
    return thetas;
}

std::vector<double> Heat_Kernel::compute_lambdas(Heat_KernelBuilder* builder) {
    std::vector<double> result(this->R);

    for (int r = 0; r < this->R; ++r) {
        const double* current_theta = this->thetas.data() + r * this->dim;
        result[r] = laplacian_symbol(current_theta, this->dim, builder);
        if (builder != nullptr) {
            builder->add_lambda_completed(1);
        }
    }
    return result;
}

void Heat_Kernel::apply_tau() {
    this->weights.resize(this->lambdas.size());
    for (size_t r = 0; r < this->lambdas.size(); ++r) {
        this->weights[r] = std::exp(-this->tau * this->lambdas[r]);
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

void Heat_Kernel::init_base(int n, int axis_dim, double resolution, int R, double tau, int seed) {
    this->n = n;
    this->R = R;
    this->tau = tau;
    this->resolution = resolution;
    this->axis_dim = axis_dim;
    this->seed = seed;
    init_dim();
}

Heat_Kernel::Heat_Kernel(int n, int axis_dim, double resolution, int R, double tau, const std::optional<std::vector<int>>& mask, std::optional<uint32_t> seed) {
    init_base(n, axis_dim, resolution, R, tau, seed.value_or(42));
    this->thetas = generate_random_thetas();
    this->lambdas = compute_lambdas();
    apply_tau();
}

Heat_Kernel::Heat_Kernel(int n, int axis_dim, double resolution, int R, double tau, const std::vector<double>& thetas, const std::vector<double>& lambdas) {
    init_base(n, axis_dim, resolution, R, tau, 0);
    this->thetas = thetas;
    this->lambdas = lambdas;
    apply_tau();
}
