#include "heat_kernel.hpp"

#include <cmath>
#include <stdexcept>

Heat_Kernel::Heat_Kernel(int n, int axis_dim, double resolution, int R, double s, double t)
    : n(n),
      axis_dim(axis_dim),
      resolution(resolution),
      R(R),
      s(s),
      t(t)
{
    init_dim();
}

Heat_Kernel::Heat_Kernel(
    int n,
    int axis_dim,
    double resolution,
    int R,
    double s,
    double t,
    const std::vector<double>& thetas,
    const std::vector<double>& weights)
    : Heat_Kernel(n, axis_dim, resolution, R, s, t)
{
    this->thetas = thetas;
    this->weights = weights;
}

int Heat_Kernel::points_per_axis() const {
    // subtract one to avoid going over the edge of the grid
    // though perhaps we might want to switch it to including the edge since we are discluding (0, 0)?
    return this->axis_dim * static_cast<int>(this->resolution) - 1;
}

void Heat_Kernel::init_dim() {
    const int ppa = points_per_axis();
    if (this->n == 1) {
        this->dim = ppa;
    } else if (this->n == 2) {
        this->dim = ppa * (ppa + 1) / 2;
    } else {
        throw std::invalid_argument("n must be 1 or 2");
    }
}
