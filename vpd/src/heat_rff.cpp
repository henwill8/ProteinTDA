#include "heat_rff.hpp"
#include <iostream>

// https://discuss.pytorch.org/t/torch-round-gradient/28628/9
torch::Tensor straight_through_round(torch::Tensor x) {
    return x + (torch::round(x) - x).detach();
}

class StraightThroughBincount : public torch::autograd::Function<StraightThroughBincount> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor indices, int64_t dim) {
            auto indices_int = torch::round(indices).to(torch::kInt64);
            
            ctx->save_for_backward({indices_int});
            return torch::bincount(indices_int, {}, dim).to(indices.options().dtype(torch::kFloat64));
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs) {
            auto indices_int = ctx->get_saved_variables()[0];
            auto grad_output = grad_outputs[0];
            auto grad_indices = torch::index_select(grad_output, 0, indices_int); // if loss increases for a bin, points in that bin get the gradient
            return {grad_indices, torch::Tensor()};
        }
};

torch::Tensor straight_through_bincount(torch::Tensor indices, int64_t dim) {
    return StraightThroughBincount::apply(indices, dim);
}

double Heat_RFF::dist_to_diagonal_grid(const std::array<double, 2>& p) const {
    // Project p onto the diagonal (t, t)
    double t = 0.5 * (p[0] + p[1]);
    const int points_per_axis = this->axis_dim * static_cast<int>(this->resolution);

    double min_t = 0.0;
    double max_t = (points_per_axis - 1) * this->resolution;

    // Find closest grid value to (t, t)
    double d_grid = std::round((t - min_t) / this->resolution) * this->resolution + min_t;
    // Clamp to grid range
    d_grid = std::clamp(d_grid, min_t, max_t);

    double dx = p[0] - d_grid;
    double dy = p[1] - d_grid;
    return std::sqrt(dx * dx + dy * dy);
}

// Quotient distance
double Heat_RFF::qdist(const std::array<double, 2>& p1, const std::array<double, 2>& p2) {
    const auto dx = p2[0] - p1[0];
    const auto dy = p2[1] - p1[1];
    const auto d_euclidean = std::sqrt(dx * dx + dy * dy);
    // const auto d_line = (std::abs(p1[1] - p1[0]) / std::numbers::sqrt2) + (std::abs(p2[1] - p2[0]) / std::numbers::sqrt2);
    const auto d_line = dist_to_diagonal_grid(p1) + dist_to_diagonal_grid(p2);
    return std::min(d_euclidean, d_line);
}

std::array<double, 2> Heat_RFF::node_at(int index) const {
    if (this->n == 1) {
        return {(index + 1) * this->resolution};
    }

    const int iy = static_cast<int>((std::sqrt(8.0 * index + 1.0) - 1.0) / 2.0); // solution to iy(iy + 1) / 2 <= index
    const int ix = index - iy * (iy + 1) / 2; // checks how many nodes were in the previous rows n(n + 1) / 2
    return {ix * this->resolution, iy * this->resolution};
}

double Heat_RFF::laplacian_symbol(const std::vector<double>& theta, int n) {
    if (theta.size() != static_cast<size_t>(n)) {
        throw std::invalid_argument("Size mismatch between theta and edges matrix.");
    }
    double result = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            double edge_weight = qdist(node_at(i), node_at(j));
            if (edge_weight != 0.0) {
                double diff = theta[i] - theta[j];
                result += edge_weight * (1.0 - std::cos(diff));
            }
        }
    }
    return result;
}

// Returns theta of given dimension
std::vector<double> Heat_RFF::generate_random_thetas() {
    std::mt19937 gen (this->seed);
    const double TWO_PI = 2.0 * std::numbers::pi;
    std::uniform_real_distribution<double> dist(0.0, TWO_PI);
    std::vector<double> thetas(this->R * this->dim);
    for (int i = 0; i < this->R * this->dim; ++i) {
        thetas[i] = dist(gen);
    }
    return thetas;
}

std::vector<double> Heat_RFF::compute_theta_weights() {
    std::vector<double> weights(this->R);
    std::vector<double> current_theta(this->dim);

    for (int r = 0; r < this->R; ++r) { 
        for (int i = 0; i < this->dim; ++i) {
            current_theta[i] = this->thetas[r * this->dim + i];
        }

        const double lambda = laplacian_symbol(current_theta, this->dim);
        weights[r] = std::exp(-this->tau * lambda);
    }
    return weights;
}

torch::Tensor Heat_RFF::align_pd_to_grid(torch::Tensor pd) {
    torch::Tensor aligned_pd = straight_through_round(pd * this->resolution);

    // It is likely that a few safety checks could be useful here. I'm not exactly sure though so we can talk about it later.

    return aligned_pd;
}

torch::Tensor Heat_RFF::pd_to_vpd(torch::Tensor pd) {
    torch::Tensor aligned_pd = align_pd_to_grid(pd);
    torch::Tensor grid_x = aligned_pd.select(1, 0);
    torch::Tensor grid_y = aligned_pd.select(1, 1);

    torch::Tensor indices;
    if (this->n == 1) {
        indices = grid_y - 1;
    } else {
        // Formula for indices in lexicographic order x <= y.
        indices = (grid_y * (grid_y - 1)) / 2 + grid_x;
    }

    return straight_through_bincount(indices, this->dim);
}

torch::Tensor Heat_RFF::pd_diff(torch::Tensor pd1, torch::Tensor pd2) {
    // Map both diagrams into the bounds of the grid [0, axis_dim] by the same scale factor (if this is not mathematically viable lemme know)
    torch::Tensor scale = torch::maximum(pd1.max(), pd2.max());
    scale = torch::clamp(scale, 1e-8);
    const double grid_max = static_cast<double>(this->axis_dim);
    torch::Tensor norm1 = pd1 / scale * grid_max;
    torch::Tensor norm2 = pd2 / scale * grid_max;

    torch::Tensor vpd1 = pd_to_vpd(norm1);
    torch::Tensor vpd2 = pd_to_vpd(norm2);
    // torch::Tensor vpd1 = pd_to_vpd(pd1);
    // torch::Tensor vpd2 = pd_to_vpd(pd2);

    return vpd1 - vpd2;
}

void Heat_RFF::init_dim() {
    if (this->n == 1) {
        this->dim = this->axis_dim * this->resolution - 1;
    } else if (this->n == 2) {
        const int points_per_axis = this->axis_dim * this->resolution;
        this->dim = (points_per_axis * points_per_axis - points_per_axis) / 2;
    } else {
        throw std::invalid_argument("n must be 1 or 2");
    }
}

void Heat_RFF::init_base(int n, int axis_dim, double resolution, int R, double tau, int seed) {
    this->n = n;
    this->R = R;
    this->tau = tau;
    this->resolution = resolution;
    this->axis_dim = axis_dim;
    this->seed = seed;
    init_dim();
}

Heat_RFF::Heat_RFF(int n, int axis_dim, double resolution, int R, double tau, const std::optional<std::vector<int>>& mask, std::optional<uint32_t> seed) {
    init_base(n, axis_dim, resolution, R, tau, seed.value_or(42));
    this->thetas = generate_random_thetas();
    this->weights = compute_theta_weights();
}

Heat_RFF::Heat_RFF(int n, int axis_dim, double resolution, int R, double tau, const std::vector<double>& thetas, const std::vector<double>& weights) {
    init_base(n, axis_dim, resolution, R, tau, 0);
    this->thetas = thetas;
    this->weights = weights;
}

torch::Tensor Heat_RFF::vpd_loss_vector_(torch::Tensor pd1, torch::Tensor pd2) {
    torch::Tensor difference_vpd = pd_diff(pd1, pd2);

    const auto tensor_options = difference_vpd.options().dtype(torch::kFloat64);
    torch::Tensor theta_tensor = torch::from_blob(this->thetas.data(), {this->R, this->dim}, torch::kFloat64).clone().to(tensor_options);
    torch::Tensor weights_tensor = torch::from_blob(this->weights.data(), {this->R}, torch::kFloat64).clone().to(tensor_options);

    // Calculating \langle \alpha, \theta^{(r)} \ranlge_{r = 1}^{R} (sorry I wrote it in LaTeX, I hope you understand it)
    // dim: [R, dim] x [dim] = [R], each ith entry is the ith dot product
    torch::Tensor dot_products = torch::matmul(theta_tensor, difference_vpd);

    // This approximates both the Monte Carlo sampling bias and the scaling by the measure v_t. 
    torch::Tensor scale = torch::sqrt(weights_tensor / static_cast<double>(this->R));

    torch::Tensor cos_vals = scale * (1 - torch::cos(dot_products));
    torch::Tensor sin_vals = scale * torch::sin(dot_products);

    return torch::cat({cos_vals, sin_vals});
}

torch::Tensor Heat_RFF::vpd_loss(torch::Tensor pd1, torch::Tensor pd2) {
    torch::Tensor vpd_loss_vector = vpd_loss_vector_(pd1, pd2);
    torch::Tensor loss = torch::sum(torch::square(vpd_loss_vector));
    return loss;
}

torch::Tensor Heat_RFF::get_vpd(torch::Tensor pd) {
    return pd_to_vpd(pd);
}

