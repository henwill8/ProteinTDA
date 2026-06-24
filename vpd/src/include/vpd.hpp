#pragma once

#include <memory>
#include <tuple>

#include <torch/torch.h>
#include <torch/extension.h>

#include "heat_kernel.hpp"

class VPD {
private:
    std::shared_ptr<Heat_Kernel> kernel;

    torch::Tensor align_pd_to_grid(torch::Tensor pd) const;
    torch::Tensor pd_to_vpd(torch::Tensor pd) const;
    torch::Tensor pd_diff(torch::Tensor pd1, torch::Tensor pd2) const;

public:
	explicit VPD(std::shared_ptr<Heat_Kernel> kernel);

	/**
	 * @brief Computes the loss between two persistent diagrams
	 *
	 * This function computes the VPD algebraic distance loss between two persistent diagrams.
	 *
	 * @param[in] pd1 The first persistent diagram.
	 * @param[in] pd2 The second persistent diagram.
	 * @param[in] subtract_zero_embedding Whether to subtract the zero embedding from the cosine values.
	 *
	 * @return TODO: add description once fixed implementation is added
	*/
	torch::Tensor vpd_loss_vector_(torch::Tensor pd1, torch::Tensor pd2, bool subtract_zero_embedding = false) const;
	
	/**
	 * @brief Computes the vpd loss between two persistent diagrams.
	 *
	 * @param[in] pd1 The first persistent diagram.
	 * @param[in] pd2 The second persistent diagram.
	 *
	 * @return The VPD algebraic distance loss between the two persistent diagrams.
	*/
	torch::Tensor vpd_loss(torch::Tensor pd1, torch::Tensor pd2) const;

	torch::Tensor get_vpd(torch::Tensor pd) const;

	const std::vector<double>& get_thetas() const { return kernel->get_thetas(); }
	const std::vector<double>& get_weights() const { return kernel->get_weights(); }
};
