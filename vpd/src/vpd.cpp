#include "vpd.hpp"
#include "straight_through.hpp"

#include <iostream>
#include <tuple>

VPD::VPD(std::shared_ptr<Heat_Kernel> kernel) : kernel(std::move(kernel)) {}

torch::Tensor VPD::align_pd_to_grid(torch::Tensor pd) const {
    return straight_through_round(pd * this->kernel->resolution);
}

torch::Tensor VPD::pd_to_vpd(torch::Tensor pd) const {
    const int64_t D = this->kernel->dim;
    const int64_t P = this->kernel->points_per_axis();
    const auto opts = pd.options().dtype(torch::kFloat64);

    if (pd.numel() == 0) {
        return torch::zeros({D}, opts) + pd.sum().to(torch::kFloat64);
    }

    // Continuous grid coords in Double. No rounding: sub-cell position is kept.
    // The .to() is differentiable; autograd casts the grad back to pd's dtype.
    torch::Tensor g  = (pd * this->kernel->resolution).to(torch::kFloat64);
    torch::Tensor gx = g.select(1, 0);
    torch::Tensor gy = g.select(1, 1);

    // Cell identity: detached bookkeeping, carries no gradient.
    torch::Tensor x0 = torch::floor(gx).detach();
    torch::Tensor y0 = torch::floor(gy).detach();

    // Fractional parts: the ONLY differentiable path. d(fx)/d(gx) == 1 exactly,
    // symmetric in x and y, so no (y - 0.5) anisotropy can appear.
    torch::Tensor fx = gx - x0;
    torch::Tensor fy = gy - y0;

    // Corner -> flat bin. int64 throughout, clamped into the valid triangle
    // (replaces masked_select, so mass is conserved rather than deleted).
    auto tri = [&](torch::Tensor xi, torch::Tensor yi) {
        torch::Tensor yl = yi.clamp(1, P - 1).to(torch::kInt64);
        torch::Tensor xl = torch::minimum(xi.clamp(0, P - 2).to(torch::kInt64), yl - 1);
        // Use truncating div: Tensor / scalar promotes integers to float in libtorch.
        torch::Tensor idx = (this->kernel->n == 1)
            ? (yl - 1)
            : (at::div(yl * (yl - 1), 2, "trunc") + xl);
        return idx.to(torch::kInt64).clamp(0, D - 1);
    };

    torch::Tensor vpd = torch::zeros({D}, opts);

    if (this->kernel->n == 1) {
        // H0: 1-D linear interpolation over 2 neighbours.
        vpd = vpd.index_add(0, tri(x0, y0),     (1 - fy))
                 .index_add(0, tri(x0, y0 + 1), fy);
    } else {
        // H1: bilinear splat over the 4 surrounding cells.
        // Partition of unity: w00 + w10 + w01 + w11 == 1, so sum_j grad(w_j) == 0
        // and dL/dp is a finite difference of dL/dalpha across adjacent cells.
        torch::Tensor w00 = (1 - fx) * (1 - fy);
        torch::Tensor w10 =      fx  * (1 - fy);
        torch::Tensor w01 = (1 - fx) *      fy;
        torch::Tensor w11 =      fx  *      fy;

        vpd = vpd.index_add(0, tri(x0,     y0),     w00)
                 .index_add(0, tri(x0 + 1, y0),     w10)
                 .index_add(0, tri(x0,     y0 + 1), w01)
                 .index_add(0, tri(x0 + 1, y0 + 1), w11);
    }

    TORCH_CHECK(vpd.dtype() == torch::kFloat64, "vpd must be Double");
    TORCH_INTERNAL_ASSERT(
        std::abs(vpd.sum().item<double>() - static_cast<double>(pd.size(0))) < 1e-6,
        "mass leak: partition of unity violated");

    return vpd;
}

torch::Tensor VPD::pd_diff(torch::Tensor pd1, torch::Tensor pd2) const {
    // // Map both diagrams into the bounds of the grid using the same scale factor for both pd's
    // torch::Tensor pd1_max = pd1.numel() == 0 ? torch::tensor(0.0, pd1.options().requires_grad(false)) : pd1.max();
    // torch::Tensor pd2_max = pd2.numel() == 0 ? torch::tensor(0.0, pd2.options().requires_grad(false)) : pd2.max();
    // torch::Tensor scale = torch::maximum(pd1_max, pd2_max);
    // scale = torch::clamp(scale, 1e-8);
    // // subtract 1 from points_per_axis as x starts at 0 and y starts at 1
    // const double grid_max = static_cast<double>(this->kernel->points_per_axis() - 1) / this->kernel->resolution;
    // torch::Tensor norm1 = pd1 / scale * grid_max;
    // torch::Tensor norm2 = pd2 / scale * grid_max;

    // torch::Tensor vpd1 = pd_to_vpd(norm1);
    // torch::Tensor vpd2 = pd_to_vpd(norm2);
    torch::Tensor vpd1 = pd_to_vpd(pd1);
    torch::Tensor vpd2 = pd_to_vpd(pd2);

    return vpd1 - vpd2;
}

torch::Tensor VPD::vpd_loss_vector_(torch::Tensor pd1, torch::Tensor pd2, bool subtract_zero_embedding) const {
    torch::Tensor difference_vpd = pd_diff(pd1, pd2);

    const auto tensor_options = difference_vpd.options().dtype(torch::kFloat64);
    torch::Tensor theta_tensor = torch::from_blob(this->kernel->thetas.data(), {this->kernel->R, this->kernel->dim}, torch::kFloat64).clone().to(tensor_options);
    // torch::Tensor weights_tensor = torch::from_blob(this->kernel->weights.data(), {this->kernel->R}, torch::kFloat64).clone().to(tensor_options);

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
