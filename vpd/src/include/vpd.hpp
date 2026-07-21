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
    torch::Tensor pd_diff(torch::Tensor pd1, torch::Tensor pd2) const;
   torch::Tensor pd_to_vpd(torch::Tensor pd) const;

public:
	explicit VPD(std::shared_ptr<Heat_Kernel> kernel);

	/**
	 * @brief Computes the loss between two persistence diagrams
	 *
	 * This function computes the VPD algebraic distance loss between two persistence diagrams.
	 *
	 * @param[in] pd1 The first persistence diagram.
	 * @param[in] pd2 The second persistence diagram.
	 * @param[in] subtract_zero_embedding Whether to subtract the zero embedding from the cosine values.
	 *
	 * @return TODO: add description once fixed implementation is added
	*/
	torch::Tensor vpd_loss_vector_(torch::Tensor pd1, torch::Tensor pd2, bool subtract_zero_embedding = false) const;
	
	/**
	 * @brief Computes the vpd loss between two persistence diagrams.
	 *
	 * @param[in] pd1 The first persistence diagram.
	 * @param[in] pd2 The second persistence diagram.
	 *
	 * @return The VPD algebraic distance loss between the two persistence diagrams.
	*/
	torch::Tensor vpd_loss(torch::Tensor pd1, torch::Tensor pd2) const;

	/**
	 * @brief Computes the VPD representation of a persistence diagram.
	 *
	 * This function returns a VPD vector representing the persistence diagram. 
	 *
	 * @param[in] pd The first persistence diagram.
	 *
   * @return The VPD vector representing the persistence diagram.
	*/
	torch::Tensor get_vpd(torch::Tensor pd) const;

	const std::vector<double>& get_thetas() const { return kernel->thetas; }
	const std::vector<double>& get_weights() const { return kernel->weights; }
};
