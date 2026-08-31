#pragma once

#include "complete_graph.hpp"
#include <cmath>

double CompleteGraph::laplacian_symbol(const double* theta) const{
    double result = 0.0;
    const int n = this->dim;

#pragma omp parallel reduction(+ : result)
    {
#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            for (int64_t j = i + 1; j < n; ++j) {
                double edge_weight = qdist(this->node_at(i), this->node_at(j));
                if (edge_weight != 0.0) {
                    double diff = theta[i] - theta[j];
                    result += 2 * edge_weight * (1.0 - std::cos(diff));
                }
                // If adding op everytime to global counter is too slow, switch to batched local counter
            }
            
            double edge_weight = dist_to_diagonal_grid(this->node_at(i));
            result += 2 * edge_weight * (1.0 - std::cos(theta[i]));
            
        }
    }
    return result * this->scale;
}

double CompleteGraph::delta_laplacian_symbol(const double* theta, int k, double proposed_val) const {
    const auto k_node = this->node_at(k);
    double current_val = theta[k];
    double delta = 0;

    for (int i = 0; i < this->dim; ++i) {
        
        if (i == k) continue;
        double weight = qdist(k_node, this->node_at(i));
        if (weight == 0) continue;
        delta += 2 * weight * (std::cos(current_val - theta[i]) - std::cos(proposed_val - theta[i]));
    }

    double weight = dist_to_diagonal_grid(k_node);
    delta += 2 * weight * (std::cos(current_val) - std::cos(proposed_val));
    
    return delta * this->scale;
}

double CompleteGraph::compute_total_edge_weights() const {
    double total = 0.0;
    const int n = this->dim;

#pragma omp parallel reduction(+ : total)
    {
#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            for (int64_t j = i + 1; j < n; ++j) {
                double edge_weight = qdist(this->node_at(i), this->node_at(j));
                if (edge_weight != 0.0) {
                    total += 2 * edge_weight;
                }
            }

            double edge_weight = dist_to_diagonal_grid(this->node_at(i));
            total += 2 * edge_weight;
        }
    }
    return total;
}

int64_t CompleteGraph::ops_per_lambda() const {
    // Mirrors laplacian_symbol's double loop: dim*(dim-1)/2 pairwise terms plus one diagonal term per node.
    return static_cast<int64_t>(this->dim) * (this->dim + 1) / 2;
}

void CompleteGraph::grad_laplacian_symbol(const double* theta, double* grad) const {
#pragma omp parallel 
    {
#pragma omp for schedule(static)
        for (int i = 0; i < this->dim; ++i) {
            double d_i = 0.0;
            for (int j = 0; j < this->dim; ++j) {
                if (i == j) continue;
                double weight = qdist(this->node_at(i), this->node_at(j));
                if (weight == 0) continue;
                d_i += 2 * weight * std::sin(theta[i] - theta[j]);
            }
            
            
            double weight = dist_to_diagonal_grid(this->node_at(i));
            d_i += 2 * weight * std::sin(theta[i]);
            grad[i] = d_i * this->scale;
        }
    } 
}
