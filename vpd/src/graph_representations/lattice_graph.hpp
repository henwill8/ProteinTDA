#pragma once

#include "graph_representation.hpp"
#include "heat_kernel.hpp"
#include<array>
#include <stdint.h>

class LatticeGraph : public GraphRepresentation {
private: 
    std::array<double, 2> node_xy_indices(int index) const;
public:
    LatticeGraph(Heat_Kernel &kernel, bool normalized_lambdas) : GraphRepresentation(kernel, normalized_lambdas) {
        this->scale = (normalized_lambdas) ? 1.0 / compute_total_edge_weights() : 1.0;
    }

    double laplacian_symbol(const double* theta) const override;
    double delta_laplacian_symbol(const double* theta, int k, double proposed_val) const override;
    void grad_laplacian_symbol(const double* theta, double* grad) const override;
    double compute_total_edge_weights() const override;
    int64_t ops_per_lambda() const override;
};
