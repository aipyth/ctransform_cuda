#include "xla/ffi/api/ffi.h"
#include <pybind11/pybind11.h>
#include "ctransform.hpp"
#include <cstddef>
#include <optional>

namespace ffi = xla::ffi;
namespace py = pybind11;

ffi::Error CTransform1DHandler_f64(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64> X,
    ffi::Buffer<ffi::F64> Y,
    ffi::Buffer<ffi::F64> phi,
    ffi::ResultBuffer<ffi::F64> out
    ) {
  Grid1D grid{
    static_cast<std::size_t>(X.dimensions()[0]),
    static_cast<std::size_t>(Y.dimensions()[0])
  };

  quadraticCTransform1D_launch(
      X.typed_data(), Y.typed_data(),
      phi.typed_data(), out->typed_data(), grid, stream);

  return ffi::Error::Success();
}

ffi::Error CTransform2DHandler_f64(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64> Xaxis0,
    ffi::Buffer<ffi::F64> Xaxis1,
    ffi::Buffer<ffi::F64> Yaxis0,
    ffi::Buffer<ffi::F64> Yaxis1,
    ffi::Buffer<ffi::F64> phi,
    ffi::ResultBuffer<ffi::F64> out
    ) {
  Grid2D grid{
    static_cast<std::size_t>(Xaxis0.dimensions()[0]),
    static_cast<std::size_t>(Xaxis1.dimensions()[0]),
    static_cast<std::size_t>(Yaxis0.dimensions()[0]),
    static_cast<std::size_t>(Yaxis1.dimensions()[0])
  };

  quadraticCTransform2D_launch(
      Xaxis0.typed_data(), Xaxis1.typed_data(),
      Yaxis0.typed_data(), Yaxis1.typed_data(),
      phi.typed_data(), out->typed_data(), grid, stream);

  return ffi::Error::Success();
}

ffi::Error CTransform2DSeparableHandler_f64(
    cudaStream_t stream,
    ffi::ScratchAllocator scratchG,
    ffi::Buffer<ffi::F64> Xaxis0,
    ffi::Buffer<ffi::F64> Xaxis1,
    ffi::Buffer<ffi::F64> Yaxis0,
    ffi::Buffer<ffi::F64> Yaxis1,
    ffi::Buffer<ffi::F64> phi,
    ffi::ResultBuffer<ffi::F64> out
    ) {
  Grid2D grid{
    static_cast<std::size_t>(Xaxis0.dimensions()[0]),
    static_cast<std::size_t>(Xaxis1.dimensions()[0]),
    static_cast<std::size_t>(Yaxis0.dimensions()[0]),
    static_cast<std::size_t>(Yaxis1.dimensions()[0])
  };
  const std::size_t scratchBytes = grid.nx0 * grid.ny1 * sizeof(double);
  std::optional<void*> maybeG = scratchG.Allocate(scratchBytes, alignof(double));
  if (!maybeG.has_value()) {
    return ffi::Error::Internal("ctransform: scratch allocation failed");
  }
  double* dScratchG = static_cast<double*>(*maybeG);

  quadraticCTransform2DSeparable_launch(
      Xaxis0.typed_data(), Xaxis1.typed_data(),
      Yaxis0.typed_data(), Yaxis1.typed_data(),
      phi.typed_data(), out->typed_data(),
      dScratchG, grid, stream);

  return ffi::Error::Success();
}
 
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CTransform1D_f64, CTransform1DHandler_f64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()               // X
        .Arg<ffi::Buffer<ffi::F64>>()               // Y
        .Arg<ffi::Buffer<ffi::F64>>()               // phi
        .Ret<ffi::Buffer<ffi::F64>>()               // out
    );

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CTransform2D_f64, CTransform2DHandler_f64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()               // Xaxis0
        .Arg<ffi::Buffer<ffi::F64>>()               // Xaxis1
        .Arg<ffi::Buffer<ffi::F64>>()               // Yaxis0
        .Arg<ffi::Buffer<ffi::F64>>()               // Yaxis1
        .Arg<ffi::Buffer<ffi::F64>>()               // phi
        .Ret<ffi::Buffer<ffi::F64>>()               // out
    );

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CTransform2DSeparable_f64, CTransform2DSeparableHandler_f64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()               // scratchG
        .Arg<ffi::Buffer<ffi::F64>>()               // Xaxis0
        .Arg<ffi::Buffer<ffi::F64>>()               // Xaxis1
        .Arg<ffi::Buffer<ffi::F64>>()               // Yaxis0
        .Arg<ffi::Buffer<ffi::F64>>()               // Yaxis1
        .Arg<ffi::Buffer<ffi::F64>>()               // phi
        .Ret<ffi::Buffer<ffi::F64>>()               // out
    );

PYBIND11_MODULE(_ctransform_ffi, m) {
  m.def("ctransform_1d_f64", []() {
      return py::capsule(reinterpret_cast<void*>(CTransform1D_f64),
          "xla._CUSTOM_CALL_TARGET");
      });

  m.def("ctransform_2d_f64", []() {
      return py::capsule(reinterpret_cast<void*>(CTransform2D_f64),
          "xla._CUSTOM_CALL_TARGET");
      });

  m.def("ctransform_2d_separable_f64", []() {
      return py::capsule(reinterpret_cast<void*>(CTransform2DSeparable_f64),
          "xla._CUSTOM_CALL_TARGET");
      });
}
