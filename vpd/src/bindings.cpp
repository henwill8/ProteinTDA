#include <torch/extension.h>
#include "heat_kernel.hpp"
#include "heat_kernel_builder.hpp"
#include "vpd.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Heat_KernelBuilder>(m, "Heat_KernelBuilder")
    .def(py::init<int, int, double, int, double, const std::optional<std::vector<int>>&, std::optional<uint32_t>, int>(),
        py::arg("n"),
        py::arg("axis_dim"),
        py::arg("resolution"),
        py::arg("R"),
        py::arg("tau"),
        py::arg("mask") = py::none(),
        py::arg("seed") = py::none(),
        py::arg("progress_batch") = Heat_KernelBuilder::DEFAULT_PROGRESS_BATCH)
    .def("build", &Heat_KernelBuilder::build, py::call_guard<py::gil_scoped_release>())
    .def("kernel", &Heat_KernelBuilder::kernel)
    .def_property_readonly("completed_ops", &Heat_KernelBuilder::completed_ops)
    .def_property_readonly("total_ops", &Heat_KernelBuilder::total_ops)
    .def_property_readonly("thetas_completed", &Heat_KernelBuilder::thetas_completed)
    .def_property_readonly("weights_completed", &Heat_KernelBuilder::weights_completed)
    .def_property_readonly("total_thetas", &Heat_KernelBuilder::total_thetas)
    .def_property_readonly("total_weights", &Heat_KernelBuilder::total_weights)
    .def_property_readonly("fraction", &Heat_KernelBuilder::fraction)
    .def_property_readonly("done", &Heat_KernelBuilder::done)
    .def_property_readonly("phase", &Heat_KernelBuilder::phase);

  py::class_<Heat_Kernel, std::shared_ptr<Heat_Kernel>>(m, "Heat_Kernel")
    .def(py::init<int, int, double, int, double, const std::optional<std::vector<int>>&, std::optional<uint32_t>>(),
        py::arg("n"),
        py::arg("axis_dim"),
        py::arg("resolution"),
        py::arg("R"),
        py::arg("tau"),
        py::arg("mask") = py::none(),
        py::arg("seed") = py::none(),
        py::call_guard<py::gil_scoped_release>())
    .def(py::init<int, int, double, int, double, const std::vector<double>&, const std::vector<double>&>(),
        py::arg("n"),
        py::arg("axis_dim"),
        py::arg("resolution"),
        py::arg("R"),
        py::arg("tau"),
        py::arg("thetas"),
        py::arg("weights"))
    .def_property_readonly("thetas", &Heat_Kernel::get_thetas)
    .def_property_readonly("weights", &Heat_Kernel::get_weights);

  py::class_<VPD>(m, "VPD")
    .def(py::init<std::shared_ptr<Heat_Kernel>>(), py::arg("kernel"))
    .def("vpd_loss_vector", &VPD::vpd_loss_vector_,
        py::arg("pd1"),
        py::arg("pd2"),
        py::arg("subtract_zero_embedding") = false)
    .def("vpd_loss", &VPD::vpd_loss,
        py::arg("pd1"),
        py::arg("pd2"))
    .def("get_vpd", &VPD::get_vpd,
        py::arg("pd"))
    .def_property_readonly("thetas", &VPD::get_thetas)
    .def_property_readonly("weights", &VPD::get_weights);
}
