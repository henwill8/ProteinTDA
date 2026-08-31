#pragma once

#include "lattice_graph.hpp"
#include <array>
#include <cmath>

std::array<double, 2> LatticeGraph::node_xy_indices(int index) const {
    if (this->n == 1) {
        return {0, index + 1};
    }

    const int iy = static_cast<int>((1.0 + std::sqrt(1.0 + 8.0 * index)) / 2.0); // solution to iy(iy - 1) / 2 <= index; no on-diagonal nodes
    const int ix = index - iy * (iy - 1) / 2; // checks how many nodes were in the previous rows, excluding the diagonal
    return {(double)ix, (double)iy};
}

double LatticeGraph::laplacian_symbol(const double* theta) const {
    double result = 0.0;
    const int n = this->dim;
    const double edge_weight = std::sqrt(1.0 / (this->resolution * this->resolution)); 

#pragma omp parallel reduction(+ : result)
    {
#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            std::array<double, 2> coords = node_xy_indices(i);  
            double theta_i = theta[i];
            double diff;

            if (this->n == 1) {
                if (coords[1] == 1) {
                    result += 2 * edge_weight * (1 - std::cos(theta_i));
                } else {
                    diff = theta_i - theta[i - 1];
                    result += 2 * edge_weight * (1 - std::cos(diff));
                }

                if (coords[1] < this->ppa) {
                    diff = theta_i - theta[i + 1];
                    result += 2 * edge_weight * (1 - std::cos(diff));
                }

            }else if (this->n == 2) {
                bool diagonal = false;
                if (coords[0] > 0) { // Left
                    diff = theta_i - theta[i - 1]; 
                    result += 2 * edge_weight * (1 - std::cos(diff));
                } 
                if (coords[1] + 1 < this->ppa) { // Up
                    diff = theta_i - theta[i + static_cast<int>(coords[1])];
                    result += 2 * edge_weight * (1 - std::cos(diff));
                }
                if (coords[0] + 1 < coords[1]) { // Right and Down
                    diff = theta_i - theta[i + 1];
                    result += 2 * edge_weight * (1 - std::cos(diff));
                    diff = theta_i - theta[i - static_cast<int>(coords[1]) + 1];
                    result += 2 * edge_weight * (1 - std::cos(diff));
                } else diagonal = true; // this node sits one step from the (quotiented) diagonal
                if (diagonal) {
                    result += 2 * edge_weight * (1 - std::cos(theta_i));
                }
            }
        }
    }
    return result * this->scale;
}

double LatticeGraph::delta_laplacian_symbol(const double*theta, int k, double proposed_val) const {
    double delta = 0;

    double current_val = theta[k];
    std::array<double, 2> coords = node_xy_indices(k); 
    const double edge_weight = std::sqrt(1.0 / (this->resolution * this->resolution)); 
    const double w = 2 * edge_weight;

    double current_diff;
    double proposed_diff;
    
    if (this->n == 1) {
        if (coords[1] == 1) {
            delta += w * (std::cos(current_val) - std::cos(proposed_val));
        } else {
            current_diff = current_val - theta[k - 1];
            proposed_diff = proposed_val - theta[k - 1];
            delta += w * (std::cos(current_diff) - std::cos(proposed_diff));
        }

        if (coords[1] < this->ppa) {
            current_diff = current_val - theta[k + 1];
            proposed_diff = proposed_val - theta[k + 1];
            delta += w * (std::cos(current_diff) - std::cos(proposed_diff));
        }
    } else if (this->n == 2) {
        bool diagonal = false;

        if (coords[0] > 0) { // Left
            current_diff = current_val - theta[k - 1]; 
            proposed_diff = proposed_val - theta[k - 1];
            delta += w * (std::cos(current_diff) - std::cos(proposed_diff));
        } 

        if (coords[1] + 1 < this->ppa) { // Up
            current_diff = current_val - theta[k + static_cast<int>(coords[1])];
            proposed_diff = proposed_val - theta[k + static_cast<int>(coords[1])];
            delta += w * (std::cos(current_diff) - std::cos(proposed_diff));
        }

        if (coords[0] + 1 < coords[1]) { // Right and Down
            current_diff = current_val - theta[k + 1];
            proposed_diff = proposed_val - theta[k + 1];
            delta += w * (std::cos(current_diff) - std::cos(proposed_diff));
            current_diff = current_val - theta[k - static_cast<int>(coords[1]) + 1]; // Down
            proposed_diff = proposed_val - theta[k - static_cast<int>(coords[1]) + 1];
            delta += w * (std::cos(current_diff) - std::cos(proposed_diff));
        } else diagonal = true; // this node sits one step from the (quotiented) diagonal

        if(diagonal) {
            delta += w * (std::cos(current_val) - std::cos(proposed_val));
        }
    }

    return delta * this->scale;
}

void LatticeGraph::grad_laplacian_symbol(const double* theta, double* grad) const {
    const double edge_weight = std::sqrt(1.0 / (this->resolution * this->resolution)); 
    const double w = 2 * edge_weight;
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for(int i = 0; i < this->dim; ++i) {
            double diff;
            double d_i = 0.0;

            std::array<double, 2> coords = node_xy_indices(i);  
            double theta_i = theta[i];

            if (this->n == 1) {
                if (coords[1] == 1) {
                    d_i += w * std::sin(theta_i);
                } else {
                    diff = theta_i - theta[i - 1];
                    d_i += w * std::sin(diff);
                }

                if (coords[1] < this->ppa) {
                    diff = theta_i - theta[i + 1];
                    d_i += w * std::sin(diff);
                }
            } else if (this->n == 2) {
                bool diagonal = false;

                if (coords[0] > 0) { // Left
                    diff = theta_i - theta[i - 1]; 
                    d_i += w * std::sin(diff);
                } 

                if (coords[1] + 1 < this->ppa) { // Up
                    diff = theta_i - theta[i + static_cast<int>(coords[1])];
                    d_i += w * std::sin(diff);
                }

                if (coords[0] + 1 < coords[1]) { // Right and Down
                    diff = theta_i - theta[i + 1];
                    d_i += w * std::sin(diff);
                    diff = theta_i - theta[i - static_cast<int>(coords[1]) + 1]; // Down
                    d_i += w * std::sin(diff);
                } else diagonal = true; // this node sits one step from the (quotiented) diagonal

                if (diagonal) {
                    d_i += w * std::sin(theta_i);
                }
            }

            grad[i] = d_i * this->scale;
        }
    }

}

double LatticeGraph::compute_total_edge_weights() const {
    double total = 0.0;
    const int n = this->dim;
    const double edge_weight = std::sqrt(1.0 / (this->resolution * this->resolution));

#pragma omp parallel reduction(+ : total)
    {
#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            std::array<double, 2> coords = node_xy_indices(i);

            if (this->n == 1) {
                total += edge_weight; // origin edge: always exactly one of these fires
                if (coords[1] < this->ppa) {
                    total += edge_weight; // Right
                }
            } else if (this->n == 2) {
                if (coords[0] > 0) total += edge_weight; // Left
                if (coords[1] + 1 < this->ppa) total += edge_weight; // Up
                if (coords[0] + 1 < coords[1]) {
                    total += 2 * edge_weight; // Right and Down
                } else {
                    total += edge_weight; // Delta edge
                }
            }
        }
    }
    return 2.0 * total;
}

int64_t LatticeGraph::ops_per_lambda() const {
    // Mirrors laplacian_symbol: at most 4 weighted terms (Left, Up, Right, Down, or the Delta fallback) per node.
    return static_cast<int64_t>(this->dim) * 4;
}
