#include "xla/ffi/api/ffi.h"
#include <pybind11/pybind11.h>
#include "ctransform.hpp"
#include <cstddef>
#include <optional>
#include <exception>
#include <string>

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

  try {
    quadraticCTransform1D_launch(
      X.typed_data(), Y.typed_data(),
      phi.typed_data(), out->typed_data(), grid, stream);
  } catch (const std::exception& e) {
    return ffi::Error::Internal(std::string("ctransform: ") + e.what());
  } catch (...) {
    return ffi::Error::Internal("ctransform: unknown exception");
  }

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

  try {
    quadraticCTransform2D_launch(
      Xaxis0.typed_data(), Xaxis1.typed_data(),
      Yaxis0.typed_data(), Yaxis1.typed_data(),
      phi.typed_data(), out->typed_data(), grid, stream);
  } catch (const std::exception& e) {
    return ffi::Error::Internal(std::string("ctransform: ") + e.what());
  } catch (...) {
    return ffi::Error::Internal("ctransform: unknown exception");
  }

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

  try {
    quadraticCTransform2DSeparable_launch(
      Xaxis0.typed_data(), Xaxis1.typed_data(),
      Yaxis0.typed_data(), Yaxis1.typed_data(),
      phi.typed_data(), out->typed_data(),
      dScratchG, grid, stream);
  } catch (const std::exception& e) {
    return ffi::Error::Internal(std::string("ctransform: ") + e.what());
  } catch (...) {
    return ffi::Error::Internal("ctransform: unknown exception");
  }

  return ffi::Error::Success();
}

// ---- 3D ----

namespace {

// Reads the grid from the six axis buffers and checks that phi and out have the matching
// 3D shapes. The Python wrappers check the same thing first; this second check protects
// the GPU from any caller that reaches the FFI target directly, because a wrong shape here
// would otherwise become an out-of-bounds read or write inside the kernel.
std::optional<ffi::Error> readGrid3D(
    const ffi::Buffer<ffi::F64>& Xaxis0, const ffi::Buffer<ffi::F64>& Xaxis1,
    const ffi::Buffer<ffi::F64>& Xaxis2, const ffi::Buffer<ffi::F64>& Yaxis0,
    const ffi::Buffer<ffi::F64>& Yaxis1, const ffi::Buffer<ffi::F64>& Yaxis2,
    const ffi::Buffer<ffi::F64>& phi, ffi::ResultBuffer<ffi::F64>& out,
    Grid3D& grid) {
  for (const auto* axis : {&Xaxis0, &Xaxis1, &Xaxis2, &Yaxis0, &Yaxis1, &Yaxis2}) {
    if (axis->dimensions().size() != 1)
      return ffi::Error::InvalidArgument("ctransform_3d: every axis must be 1-dimensional");
  }
  grid = Grid3D{
    static_cast<std::size_t>(Xaxis0.dimensions()[0]),
    static_cast<std::size_t>(Xaxis1.dimensions()[0]),
    static_cast<std::size_t>(Xaxis2.dimensions()[0]),
    static_cast<std::size_t>(Yaxis0.dimensions()[0]),
    static_cast<std::size_t>(Yaxis1.dimensions()[0]),
    static_cast<std::size_t>(Yaxis2.dimensions()[0])
  };

  const auto p = phi.dimensions();
  if (p.size() != 3 ||
      static_cast<std::size_t>(p[0]) != grid.nx0 ||
      static_cast<std::size_t>(p[1]) != grid.nx1 ||
      static_cast<std::size_t>(p[2]) != grid.nx2)
    return ffi::Error::InvalidArgument("ctransform_3d: phi must have shape (nx0, nx1, nx2)");

  const auto o = out->dimensions();
  if (o.size() != 3 ||
      static_cast<std::size_t>(o[0]) != grid.ny0 ||
      static_cast<std::size_t>(o[1]) != grid.ny1 ||
      static_cast<std::size_t>(o[2]) != grid.ny2)
    return ffi::Error::InvalidArgument("ctransform_3d: out must have shape (ny0, ny1, ny2)");

  return std::nullopt;
}

}  // namespace

ffi::Error CTransform3DHandler_f64(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64> Xaxis0,
    ffi::Buffer<ffi::F64> Xaxis1,
    ffi::Buffer<ffi::F64> Xaxis2,
    ffi::Buffer<ffi::F64> Yaxis0,
    ffi::Buffer<ffi::F64> Yaxis1,
    ffi::Buffer<ffi::F64> Yaxis2,
    ffi::Buffer<ffi::F64> phi,
    ffi::ResultBuffer<ffi::F64> out
    ) {
  Grid3D grid{};
  if (auto err = readGrid3D(Xaxis0, Xaxis1, Xaxis2, Yaxis0, Yaxis1, Yaxis2, phi, out, grid))
    return *err;

  try {
    quadraticCTransform3D_launch(
      Xaxis0.typed_data(), Xaxis1.typed_data(), Xaxis2.typed_data(),
      Yaxis0.typed_data(), Yaxis1.typed_data(), Yaxis2.typed_data(),
      phi.typed_data(), out->typed_data(), grid, stream);
  } catch (const std::exception& e) {
    return ffi::Error::Internal(std::string("ctransform: ") + e.what());
  } catch (...) {
    return ffi::Error::Internal("ctransform: unknown exception");
  }

  return ffi::Error::Success();
}

ffi::Error CTransform3DSeparableHandler_f64(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::Buffer<ffi::F64> Xaxis0,
    ffi::Buffer<ffi::F64> Xaxis1,
    ffi::Buffer<ffi::F64> Xaxis2,
    ffi::Buffer<ffi::F64> Yaxis0,
    ffi::Buffer<ffi::F64> Yaxis1,
    ffi::Buffer<ffi::F64> Yaxis2,
    ffi::Buffer<ffi::F64> phi,
    ffi::ResultBuffer<ffi::F64> out
    ) {
  Grid3D grid{};
  if (auto err = readGrid3D(Xaxis0, Xaxis1, Xaxis2, Yaxis0, Yaxis1, Yaxis2, phi, out, grid))
    return *err;

  const std::size_t scratchBytes = quadraticCTransform3DSeparable_scratchSize(grid) * sizeof(double);
  double* dScratch = nullptr;
  if (scratchBytes > 0) {
    std::optional<void*> maybe = scratch.Allocate(scratchBytes, alignof(double));
    if (!maybe.has_value())
      return ffi::Error::Internal("ctransform: scratch allocation failed");
    dScratch = static_cast<double*>(*maybe);
  }

  try {
    quadraticCTransform3DSeparable_launch(
      Xaxis0.typed_data(), Xaxis1.typed_data(), Xaxis2.typed_data(),
      Yaxis0.typed_data(), Yaxis1.typed_data(), Yaxis2.typed_data(),
      phi.typed_data(), out->typed_data(), dScratch, grid, stream);
  } catch (const std::exception& e) {
    return ffi::Error::Internal(std::string("ctransform: ") + e.what());
  } catch (...) {
    return ffi::Error::Internal("ctransform: unknown exception");
  }

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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CTransform3D_f64, CTransform3DHandler_f64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()               // Xaxis0
        .Arg<ffi::Buffer<ffi::F64>>()               // Xaxis1
        .Arg<ffi::Buffer<ffi::F64>>()               // Xaxis2
        .Arg<ffi::Buffer<ffi::F64>>()               // Yaxis0
        .Arg<ffi::Buffer<ffi::F64>>()               // Yaxis1
        .Arg<ffi::Buffer<ffi::F64>>()               // Yaxis2
        .Arg<ffi::Buffer<ffi::F64>>()               // phi
        .Ret<ffi::Buffer<ffi::F64>>()               // out
    );

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CTransform3DSeparable_f64, CTransform3DSeparableHandler_f64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()               // scratch for g1 and g2
        .Arg<ffi::Buffer<ffi::F64>>()               // Xaxis0
        .Arg<ffi::Buffer<ffi::F64>>()               // Xaxis1
        .Arg<ffi::Buffer<ffi::F64>>()               // Xaxis2
        .Arg<ffi::Buffer<ffi::F64>>()               // Yaxis0
        .Arg<ffi::Buffer<ffi::F64>>()               // Yaxis1
        .Arg<ffi::Buffer<ffi::F64>>()               // Yaxis2
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

  m.def("ctransform_3d_f64", []() {
      return py::capsule(reinterpret_cast<void*>(CTransform3D_f64),
          "xla._CUSTOM_CALL_TARGET");
      });

  m.def("ctransform_3d_separable_f64", []() {
      return py::capsule(reinterpret_cast<void*>(CTransform3DSeparable_f64),
          "xla._CUSTOM_CALL_TARGET");
      });
}
