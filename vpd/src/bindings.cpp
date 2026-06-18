#include <torch/extension.h>
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
        py::arg("pd2"))

    .def("get_vpd", &Heat_RFF::get_vpd,
        py::arg("pd"));
}
