#pragma once

#include "heat_kernel.hpp"
#include <array>
#include <stdint.h>

class GraphRepresentation {
public:
    GraphRepresentation(Heat_Kernel &kernel, bool normalized_lambdas) : dim(kernel.dim), resolution(kernel.resolution), ppa(kernel.ppa), n(kernel.n), normalized_lambdas(normalized_lambdas) {}
    virtual ~GraphRepresentation () = default;

    int dim;
    int ppa;
    int n;
    bool normalized_lambdas;
    int resolution;
    double scale;

    std::array<double, 2> node_at(int index) const;
    double dist_to_diagonal_grid(const std::array<double, 2>& p) const;
    double qdist(const std::array<double, 2>& p1, const std::array<double, 2>& p2) const;
    
    virtual double laplacian_symbol(const double* theta) const = 0;
    virtual double delta_laplacian_symbol(const double* theta, int k, double proposed_val) const = 0;
    
    virtual void grad_laplacian_symbol(const double* theta, double* grad) const = 0;
    virtual double compute_total_edge_weights() const = 0;
    virtual int64_t ops_per_lambda() const = 0;
};
