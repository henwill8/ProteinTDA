#pragma once

#include <vector>

struct Heat_Kernel {
    int n;
    int axis_dim;
    int ppa;
    double resolution;
    int R;
    double s;
    double t;
    int dim;
    std::vector<double> thetas;
    std::vector<double> weights;

    /**
     * @brief Defines the parameters of the heat kernel.
     *
     * @param[in] n The dimensionality of the points on our persistent diagram.
     * @param[in] axis_dim The size of all axes.
     * @param[in] resolution The number of points between any two integers on a axis of our grid.
     * @param[in] R The number of samples to take.
     * @param[in] s The s value used for character weight calculation.
     * @param[in] t The time value to use for the heat kernel computations
     */
    Heat_Kernel(int n, int axis_dim, double resolution, int R, double s, double t);

    /**
     * @brief Creates a Heat_Kernel from precomputed thetas and weights.
     */
    Heat_Kernel(
        int n,
        int axis_dim,
        double resolution,
        int R,
        double s,
        double t,
        const std::vector<double>& thetas,
        const std::vector<double>& weights);

    int points_per_axis() const;
    void init_dim();
};
