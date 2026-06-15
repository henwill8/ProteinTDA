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
            return torch::bincount(indices, {}, dim).to(torch::kFloat64);
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
double Heat_RFF::qdist(const std::array<double, 2>& p1, const std::array<double, 2>& p2) {
    // Euclidean distance
    const auto dx = p2[0] - p1[0];
    const auto dy = p2[1] - p1[1];
    const auto d_euclidean = sqrt(dx * dx + dy * dy);

    // Distance to line y = x - should be careful this is the distance to a point on the grid
    const auto d_line = (std::abs(p1[1] - p1[0]) / std::numbers::sqrt2) + (std::abs(p2[1] - p2[0]) / std::numbers::sqrt2);

    // TODO: fix this
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

Heat_RFF::Heat_RFF(int n, int axis_dim, double resolution, int R, double tau, const std::optional<std::vector<int>>& mask, std::optional<uint32_t> seed) {
    this->n = n;
    this->R = R;
    this->tau = tau;
    this->resolution = resolution;
    this->axis_dim = axis_dim;
    if (seed == std::nullopt) { // < --- This could be changed to have true randomness instead
        this->seed = 42;
    } else {
        this->seed = seed.value();
    }
    if (n == 1) {
        this->dim = axis_dim * resolution;
    } else if (n == 2) {
        int points_per_axis = axis_dim * resolution;
        this->dim = (points_per_axis * points_per_axis - points_per_axis) / 2;
    }
    this->thetas = this->generate_random_thetas();
    this->weights = this->compute_theta_weights();
}

torch::Tensor Heat_RFF::vpd_loss_vector_(torch::Tensor pd1, torch::Tensor pd2) {
    torch::Tensor difference_vpd = pd_diff(pd1, pd2);

    torch::Tensor theta_tensor = torch::from_blob(this->thetas.data(), {this->R, this->dim}, torch::kFloat64);
    torch::Tensor weights_tensor = torch::from_blob(this->weights.data(), {this->R}, torch::kFloat64);

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

