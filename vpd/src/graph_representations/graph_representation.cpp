#pragma once

#include "graph_representation.hpp"
#include <array>

double GraphRepresentation::dist_to_diagonal_grid(const std::array<double, 2>& p) const {
    // Project p onto the diagonal (t, t)
    double t = 0.5 * (p[0] + p[1]);

    double min_t = 0.0;
    double max_t = kernel->points_per_axis() * kernel->resolution;

    // Find closest grid value to (t, t)
    double d_grid = std::round((t - min_t) * kernel->resolution) / kernel->resolution + min_t;
    // Clamp to grid range
    d_grid = std::clamp(d_grid, min_t, max_t);

    double dx = p[0] - d_grid;
    double dy = p[1] - d_grid;
    return std::sqrt(dx * dx + dy * dy);
}

// Quotient distance
double GraphRepresentation::qdist(const std::array<double, 2>& p1, const std::array<double, 2>& p2) const {
    const auto dx = p2[0] - p1[0];
    const auto dy = p2[1] - p1[1];
    const auto d_euclidean = std::sqrt(dx * dx + dy * dy);
    const auto d_line = dist_to_diagonal_grid(p1) + dist_to_diagonal_grid(p2);
    return std::min(d_euclidean, d_line);
}

std::array<double, 2> GraphRepresentation::node_at(int index) const {
    if (this->n == 1) {
        return {(index + 1) / kernel->resolution};
    }

    const int iy = static_cast<int>((std::sqrt(8.0 * index + 1.0) - 1.0) / 2.0); // solution to iy(iy + 1) / 2 <= index
    const int ix = index - iy * (iy + 1) / 2; // checks how many nodes were in the previous rows n(n + 1) / 2
    return {ix / kernel->resolution, iy / kernel->resolution};
}
