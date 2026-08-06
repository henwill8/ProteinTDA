#pragma once

#include "lattice_graph.hpp"
#include <array>
#include <cmath>

std::array<int, 2> LatticeGraph::node_xy_indices(int index) const {
    if (this->n == 1) {
        return {0, index + 1};
    }

    const int iy = static_cast<int>((std::sqrt(8.0 * index + 1.0) - 1.0) / 2.0); // solution to iy(iy + 1) / 2 <= index
    const int ix = index - iy * (iy + 1) / 2; // checks how many nodes were in the previous rows n(n + 1) / 2
    return {ix, iy};
}

double LatticeGraph::laplacian_symbol(const double* theta) const{
    double result = 0.0;
    const int n = this->dim;
    
    // In this implementation, there is no qdist as the quotient shouldn't ever be the fastest path.
    
    const double edge_weight = std::sqrt(1.0 / (this->resolution * this->resolution)); 

    double diff;

    for (int i = 0; i < n; ++i) {
        std::array<int, 2> coords = node_xy_indices(i);  
        double theta_i = theta[i];
        bool diagonal = false;

        if (coords[0] > 0) { // Left
            diff = theta_i - theta[i - 1]; 
            result += 2 * edge_weight * (1 - std::cos(diff));
        } else diagonal = true;

        if (coords[1] < this->ppa) { // Up
            diff = theta_i - theta[i + coords[1]];
            result += 2 * edge_weight * (1 - std::cos(diff));
        } else diagonal = true;

        if (coords[0] + 1 < coords[1]) { 
            diff = theta_i - theta[i + 1];//right
            result += 2 * edge_weight * (1 - std::cos(diff));
            diff = theta_i - theta[i - coords[1]+ 1];
            result += 2 * edge_weight * (1 - std::cos(diff));
        } else diagonal = true;

        if (diagonal) {
            double diag_weight = this->dist_to_diagonal_grid(coords);
            result += 2 * diag_weight * (1 - std::cos(theta_i));
        }
    }

    return result * this->scale;
}
