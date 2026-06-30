#include "vpd.hpp"
#include "straight_through.hpp"

#include <iostream>
#include <tuple>

VPD::VPD(std::shared_ptr<Heat_Kernel> kernel) : kernel(std::move(kernel)) {}

torch::Tensor VPD::align_pd_to_grid(torch::Tensor pd) const {
    torch::Tensor aligned_pd = straight_through_round(pd * this->kernel->resolution);

    // It is likely that a few safety checks could be useful here. I'm not exactly sure though so we can talk about it later.

    // Remove any indices that are greater than dim - 1, as that would create a shape mismatch
    auto invalid_mask = aligned_pd > this->kernel->dim - 1;
    if (invalid_mask.any().item<bool>()) {
        auto invalid_indices = aligned_pd.masked_select(invalid_mask);
        std::cout << "Bincount: Removing "
                  << invalid_indices.size(0)
                  << " indices >= dim (" << this->kernel->dim << "): "
                  << invalid_indices << std::endl;
        aligned_pd = aligned_pd.masked_select(~invalid_mask);
    }
    return aligned_pd;
}

torch::Tensor VPD::pd_to_vpd(torch::Tensor pd) const {
    torch::Tensor aligned_pd = align_pd_to_grid(pd);
    torch::Tensor grid_x = aligned_pd.select(1, 0);
    torch::Tensor grid_y = aligned_pd.select(1, 1);

    torch::Tensor indices;
    if (this->kernel->n == 1) {
        indices = grid_y - 1;
    } else {
        // Formula for indices in lexicographic order x <= y.
        indices = (grid_y * (grid_y - 1)) / 2 + grid_x;
    }

    return straight_through_bincount(indices, this->kernel->dim);
}

torch::Tensor VPD::pd_diff(torch::Tensor pd1, torch::Tensor pd2) const {
    // Map both diagrams into the bounds of the grid using the same scale factor for both pd's
    torch::Tensor pd1_max = pd1.numel() == 0 ? torch::tensor(0.0, pd1.options().requires_grad(false)) : pd1.max();
    torch::Tensor pd2_max = pd2.numel() == 0 ? torch::tensor(0.0, pd2.options().requires_grad(false)) : pd2.max();
    torch::Tensor scale = torch::maximum(pd1_max, pd2_max);
    scale = torch::clamp(scale, 1e-8);
    // subtract 1 from points_per_axis as x starts at 0 and y starts at 1
    const double grid_max = static_cast<double>(this->kernel->points_per_axis() - 1) / this->kernel->resolution;
    torch::Tensor norm1 = pd1 / scale * grid_max;
    torch::Tensor norm2 = pd2 / scale * grid_max;

    torch::Tensor vpd1 = pd_to_vpd(norm1);
    torch::Tensor vpd2 = pd_to_vpd(norm2);
    // torch::Tensor vpd1 = pd_to_vpd(pd1);
    // torch::Tensor vpd2 = pd_to_vpd(pd2);

    return vpd1 - vpd2;
}

torch::Tensor VPD::vpd_loss_vector_(torch::Tensor pd1, torch::Tensor pd2, bool subtract_zero_embedding) const {
    torch::Tensor difference_vpd = pd_diff(pd1, pd2);

    const auto tensor_options = difference_vpd.options().dtype(torch::kFloat64);
    torch::Tensor theta_tensor = torch::from_blob(this->kernel->thetas.data(), {this->kernel->R, this->kernel->dim}, torch::kFloat64).clone().to(tensor_options);
    torch::Tensor weights_tensor = torch::from_blob(this->kernel->weights.data(), {this->kernel->R}, torch::kFloat64).clone().to(tensor_options);

    // Calculating \langle \alpha, \theta^{(r)} \ranlge_{r = 1}^{R} (sorry I wrote it in LaTeX, I hope you understand it)
    // dim: [R, dim] x [dim] = [R], each ith entry is the ith dot product
    torch::Tensor dot_products = torch::matmul(theta_tensor, difference_vpd);

    double scale = std::sqrt(1.0 / static_cast<double>(this->kernel->R));

    torch::Tensor cos_vals = scale * (torch::cos(dot_products) - (subtract_zero_embedding ? 1 : 0));
    torch::Tensor sin_vals = scale * torch::sin(dot_products);

    torch::Tensor vpd_loss_vector = torch::stack({cos_vals, sin_vals}, 1).view(-1);

    return vpd_loss_vector;
}

torch::Tensor VPD::vpd_loss(torch::Tensor pd1, torch::Tensor pd2) const {
    torch::Tensor vpd_loss_vector = vpd_loss_vector_(pd1, pd2, true);
    torch::Tensor loss = torch::sum(vpd_loss_vector.pow(2));
    return loss;
}

torch::Tensor VPD::get_vpd(torch::Tensor pd) const {
    return pd_to_vpd(pd);
}
