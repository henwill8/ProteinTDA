#include <torch/extension.h>
#include "heat_kernel.hpp"
#include "sampling_method.hpp"
#include "rejection_sampling.hpp"
#include "metropolis_hastings_sampling.hpp"
#include "mala_sampling.hpp"
#include "vpd.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::enum_<Device>(m, "Device")
    .value("CPU", Device::CPU)
    .value("CUDA", Device::CUDA);
  py::class_<Heat_Kernel, std::shared_ptr<Heat_Kernel>>(m, "Heat_Kernel")

    .def(py::init<int, int, double, int, double, double>(),
        py::arg("n"),
        py::arg("axis_dim"),
        py::arg("resolution"),
        py::arg("R"),
        py::arg("s"),
        py::arg("t"))
    .def(py::init<int, int, double, int, double, double, const std::vector<double>&, const std::vector<double>&>(),
        py::arg("n"),
        py::arg("axis_dim"),
        py::arg("resolution"),
        py::arg("R"),
        py::arg("s"),
        py::arg("t"),
        py::arg("thetas"),
        py::arg("weights"))
    .def_readonly("thetas", &Heat_Kernel::thetas)
    .def_readonly("weights", &Heat_Kernel::weights);

  py::class_<SamplingMethod, std::shared_ptr<SamplingMethod>>(m, "SamplingMethod")
    .def("init", &SamplingMethod::init,
        py::arg("kernel"),
        py::arg("normalized_lambdas") = true,
        py::arg("seed") = 42,
        py::arg("device") = Device::CPU)
    .def("build", &SamplingMethod::build, py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("completed_ops", &SamplingMethod::completed_ops)
    .def_property_readonly("total_ops", &SamplingMethod::total_ops)
    .def_property_readonly("weights_completed", &SamplingMethod::weights_completed)
    .def_property_readonly("total_weights", &SamplingMethod::total_weights)
    .def("progress_postfix", &SamplingMethod::progress_postfix);

  py::class_<RejectionSampling, SamplingMethod, std::shared_ptr<RejectionSampling>>(m, "RejectionSamplingKernel")
    .def(py::init<>())
    .def_property_readonly("attempts_completed", &RejectionSampling::attempts_completed)
    .def_property_readonly("acceptance_rate", &RejectionSampling::acceptance_rate);

  py::class_<MetropolisHastingsSampling, SamplingMethod, std::shared_ptr<MetropolisHastingsSampling>>(m, "MetropolisHastingsSamplingKernel")
    .def(py::init<double, int, int>(),
        py::arg("sigma"),
        py::arg("burn_in"),
        py::arg("thinning"));

  py::class_<MALASampling, SamplingMethod, std::shared_ptr<MALASampling>>(m, "MALASamplingKernel")
    .def(py::init<double, int, int, bool>(),
        py::arg("sigma"),
        py::arg("burn_in"),
        py::arg("thinning"),
        py::arg("tune_sigma"));

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
