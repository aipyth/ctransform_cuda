#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

#include "ctransform.hpp"

namespace py = pybind11;

py::array_t<double> ctransform_1d(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> Y,
    py::array_t<double, py::array::c_style | py::array::forcecast> phi
    ) {
  if (X.ndim() != 1 || Y.ndim() != 1 || phi.ndim() != 1)
    throw std::invalid_argument("X, Y, phi must be 1D");
  if (phi.shape(0) != X.shape(0))
    throw std::invalid_argument("phi.shape must match X.shape");

  Grid1D grid(
      static_cast<std::size_t>(X.shape(0)),
      static_cast<std::size_t>(Y.shape(0))
      );
  py::array_t<double> out(Y.shape(0));

  quadraticCTransform1D<double>(X.data(), Y.data(), phi.data(), out.mutable_data(), grid);

  return out;
}

py::array_t<double> ctransform_2d(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xaxis0,
    py::array_t<double, py::array::c_style | py::array::forcecast> Xaxis1,
    py::array_t<double, py::array::c_style | py::array::forcecast> Yaxis0,
    py::array_t<double, py::array::c_style | py::array::forcecast> Yaxis1,
    py::array_t<double, py::array::c_style | py::array::forcecast> phi,
    std::string method = "naive"
    ) {
  if (Xaxis0.ndim() != 1  || Xaxis1.ndim() != 1 || Yaxis0.ndim() != 1 || Yaxis1.ndim() != 1)
    throw std::invalid_argument("Xaxis0, Xaxis1, Yaxis0, Yaxis1 must be 1D");
  if (phi.ndim() != 2)
    throw std::invalid_argument("phi must be 2D");
  if (phi.shape(0) != Xaxis0.shape(0) || phi.shape(1) != Xaxis1.shape(0))
    throw std::invalid_argument("phi.shape must match (Xaxis0.shape[0], Xaxis1.shape[1])");
  if (method != "naive" && method != "separable")
    throw std::invalid_argument("method must be 'naive' or 'separable'");

  Grid2D grid(
      static_cast<std::size_t>(Xaxis0.shape(0)),
      static_cast<std::size_t>(Xaxis1.shape(0)),
      static_cast<std::size_t>(Yaxis0.shape(0)),
      static_cast<std::size_t>(Yaxis1.shape(0))
      );
  py::array_t<double> out({Yaxis0.shape(0), Yaxis1.shape(0)});

  if (method == "naive") {
    quadraticCTransform2D<double>(
        Xaxis0.data(), Xaxis1.data(), Yaxis0.data(), Yaxis1.data(),
        phi.data(), out.mutable_data(), grid);
  } else if (method == "separable") {
    quadraticCTransform2DSeparable<double>(
        Xaxis0.data(), Xaxis1.data(), Yaxis0.data(), Yaxis1.data(),
        phi.data(), out.mutable_data(), grid);
  }

  return out;
}

PYBIND11_MODULE(ctransform_cuda_py, m) {
  m.def("ctransform_1d", &ctransform_1d, "1D quadratic c-transform",
      py::arg("X"), py::arg("Y"), py::arg("phi"));
  m.def("ctransform_2d", &ctransform_2d, "2D quadratic c-transform",
      py::arg("Xaxis0"), py::arg("Xaxis1"), py::arg("Yaxis0"), py::arg("Yaxis1"),
      py::arg("phi"), py::arg("method") = "naive");
}
