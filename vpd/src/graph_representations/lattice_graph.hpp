#pragma once

#include "graph_representation.hpp"
#include <stdint.h>

class LatticeGraph: public GraphRepresentation {
public:
    LatticeGraph();

    double laplacian_symbol(const double* theta) const override;
    double delta_laplacian_symbol(const double* theta, int k, double proposed_val) const override;
    void grad_laplacian_symbol(const double* theta, double* grad) const override;
    double compute_total_edge_weights() const override;
    int64_t ops_per_lambda() const override;
};
