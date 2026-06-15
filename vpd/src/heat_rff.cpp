#include "heat_rff.hpp"

// https://discuss.pytorch.org/t/torch-round-gradient/28628/9
torch::Tensor straight_through_round(torch::Tensor x) {
    return x + (torch::round(x) - x).detach();
}

class StraightThroughBincount : public torch::autograd::Function<StraightThroughBincount> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor indices, int64_t dim) {
            // indices might need to be rounded to int64 when passing to bincount
            // auto indices = torch::round(indices).to(torch::kInt64);
            
            ctx->save_for_backward({indices});
            return torch::bincount(indices, {}, dim).to(indices.options().dtype(torch::kFloat64));
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs) {
            auto indices = ctx->get_saved_variables()[0];
            auto grad_output = grad_outputs[0];
            auto grad_indices = torch::index_select(grad_output, 0, indices); // if loss increases for a bin, points in that bin get the gradient
            return {grad_indices, torch::Tensor()};
        }
};

torch::Tensor straight_through_bincount(torch::Tensor indices, int64_t dim) {
    return StraightThroughBincount::apply(indices, dim);
}

// Quotient distance
double Heat_RFF::dist_to_diagonal_grid(const std::array<double, 2>& p) const {
    const int points_per_axis = this->axis_dim * static_cast<int>(this->resolution);
    double best = std::numeric_limits<double>::infinity();
    for (int k = 0; k < points_per_axis; ++k) {
        const double d = k * this->resolution;
        const auto dx = p[0] - d;
        const auto dy = p[1] - d;
        best = std::min(best, std::sqrt(dx * dx + dy * dy));
    }
    return best;
}

double Heat_RFF::qdist(const std::array<double, 2>& p1, const std::array<double, 2>& p2) {
    const auto dx = p2[0] - p1[0];
    const auto dy = p2[1] - p1[1];
    const auto d_euclidean = std::sqrt(dx * dx + dy * dy);
    const auto d_line = dist_to_diagonal_grid(p1) + dist_to_diagonal_grid(p2);
    return std::min(d_euclidean, d_line);
}

std::array<double, 2> Heat_RFF::node_at(int index) const {
    if (this->n == 1) {
        return {0.0, index * this->resolution};
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

        double lambda = laplacian_symbol(current_theta, this->dim);
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
    torch::Tensor ix = straight_through_round(aligned_pd.select(1, 0));
    torch::Tensor iy = straight_through_round(aligned_pd.select(1, 1));

    torch::Tensor indices;
    if (this->n == 1) {
        indices = iy;
    } else {
        //Formula for indices in lexicograpghic order x<= y, I can convince you its right in person Monday lol.
        indices = (iy * (iy - 1))/2 + ix;
    }

    return straight_through_bincount(indices, this->dim);
}

torch::Tensor Heat_RFF::pd_diff(torch::Tensor pd1, torch::Tensor pd2) {
    torch::Tensor vpd1 = pd_to_vpd(pd1);
    torch::Tensor vpd2 = pd_to_vpd(pd2);
    
    return vpd1 - vpd2;
}

void Heat_RFF::init_dim() {
    if (this->n == 1) {
        this->dim = this->axis_dim * this->resolution;
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

void Heat_RFF::set_thetas_and_weights(const std::vector<double>& thetas, const std::vector<double>& weights) {
    if (thetas.size() != static_cast<size_t>(this->R * this->dim)) {
        throw std::invalid_argument("thetas size mismatch");
    }
    if (weights.size() != static_cast<size_t>(this->R)) {
        throw std::invalid_argument("weights size mismatch");
    }
    this->thetas = thetas;
    this->weights = weights;
}

Heat_RFF::Heat_RFF(int n, int axis_dim, double resolution, int R, double tau, const std::optional<std::vector<int>>& mask, std::optional<uint32_t> seed) {
    init_base(n, axis_dim, resolution, R, tau, seed.value_or(42));
    set_thetas_and_weights(generate_random_thetas(), compute_theta_weights());
}

Heat_RFF::Heat_RFF(int n, int axis_dim, double resolution, int R, double tau, const std::vector<double>& thetas, const std::vector<double>& weights) {
    init_base(n, axis_dim, resolution, R, tau, 0);
    set_thetas_and_weights(thetas, weights);
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

    torch::Tensor cos_vals = scale * torch::cos(dot_products);
    torch::Tensor sin_vals = scale * torch::sin(dot_products);

    return torch::cat({cos_vals, sin_vals});
}

torch::Tensor Heat_RFF::vpd_loss(torch::Tensor pd1, torch::Tensor pd2) {
    torch::Tensor vpd_loss_vector = vpd_loss_vector_(pd1, pd2);
    torch::Tensor loss = torch::sum(torch::square(vpd_loss_vector));
    return loss;
}

