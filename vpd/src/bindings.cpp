#include <torch/extension.h>
#include "heat_flow.hpp"
#include "heat_rff.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Heat_RFF>(m, "Heat_RFF")
    .def(py::init<int, int, double, int, double, const std::optional<std::vector<int>>&, std::optional<uint32_t>>(),
        py::arg("n"),
        py::arg("axis_dim"),
        py::arg("resolution"),
        py::arg("R"),
        py::arg("tau"),
        py::arg("mask") = py::none(),
        py::arg("seed") = py::none())
    .def(py::init<int, int, double, int, double, const std::vector<double>&, const std::vector<double>&>(),
        py::arg("n"),
        py::arg("axis_dim"),
        py::arg("resolution"),
        py::arg("R"),
        py::arg("tau"),
        py::arg("thetas"),
        py::arg("weights"))
    .def_property_readonly("thetas", &Heat_RFF::get_thetas)
    .def_property_readonly("weights", &Heat_RFF::get_weights)

    .def("vpd_loss_vector", &Heat_RFF::vpd_loss_vector_,
        py::arg("pd1"),
        py::arg("pd2"))

    .def("vpd_loss", &Heat_RFF::vpd_loss,
        py::arg("pd1"),
        py::arg("pd2"));
    m.def(
        "graph_laplacian",
        &graph_laplacian,
        "Compute graph Laplacian",
        py::arg("adjacency"),
        py::arg("normalized") = true
    );
    m.def(
        "heat_kernel",
        &heat_kernel,
        "Compute heat kernel H(tau) = exp(-tau * L)",
        py::arg("L"),
        py::arg("tau")
    );
    m.def(
        "heat_edge_weights",
        &heat_edge_weights,
        "Compute heat-based edge weights w_tau(u,v) = H(tau)_{uv}",
        py::arg("adjacency"),
        py::arg("tau") = 1.0,
        py::arg("normalized") = true
    );
    m.def(
        "heat_vertex_function",
        &heat_vertex_function,
        "Compute heat-derived vertex function for lower-star filtration",
        py::arg("adjacency"),
        py::arg("tau") = 1.0,
        py::arg("source") = py::none(),
        py::arg("method") = "content",
        py::arg("normalize") = "rank"
    );
    m.def(
        "lower_star_filtration_value",
        &lower_star_filtration_value,
        "Compute lower-star filtration value: f(sigma) = max_{v in sigma} f(v)",
        py::arg("clique_vertices"),
        py::arg("vertex_function")
    );
}
