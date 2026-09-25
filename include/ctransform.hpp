#pragma once

#include <cstddef>

struct Grid1D { std::size_t nx, ny; };

struct Grid2D { std::size_t nx0, nx1, ny0, ny1; };

struct Grid3D { std::size_t nx0, nx1, nx2, ny0, ny1, ny2; };


// public host wrapper function
template <typename T>
void quadraticCTransform1D(
    const T* X,
    const T* Y,
    const T* phi,
    T* out,
    Grid1D grid
);

template <typename T>
void quadraticCTransformCPU1D(
    const T* X,
    const T* Y,
    const T* Phi,
    T* out,
    Grid1D grid
);

template <typename T>
void quadraticCTransform2D(
    const T* Xaxis0, const T* Xaxis1,
    const T* Yaxis0, const T* Yaxis1,
    const T* Phi,
    T* out,
    Grid2D grid
    );

template <typename T>
void quadraticCTransformCPU2D(
    const T* Xaxis0, const T* Xaxis1,
    const T* Yaxis0, const T* Yaxis1,
    const T* Phi,
    T* out,
    Grid2D grid
    );

template <typename T>
void quadraticCTransform2DSeparable(
    const T* Xaxis0, const T* Xaxis1,
    const T* Yaxis0, const T* Yaxis1,
    const T* Phi,
    T* out,
    Grid2D grid
    );

typedef struct CUstream_st* cudaStream_t;

template <typename T>
void quadraticCTransform1D_launch(
    const T* dXaxis, const T* dYaxis,     // DEVICE pointers
    const T* dPhi,
    T* dOut,
    Grid1D grid,
    cudaStream_t stream = 0
    );

template <typename T>
void quadraticCTransform2D_launch(
    const T* dXaxis0, const T* dXaxis1,     // DEVICE pointers
    const T* dYaxis0, const T* dYaxis1,
    const T* dPhi,
    T* dOut,
    Grid2D grid,
    cudaStream_t stream = 0
    );

template <typename T>
void quadraticCTransform2DSeparable_launch(
    const T* dXaxis0, const T* dXaxis1,     // DEVICE pointers
    const T* dYaxis0, const T* dYaxis1,
    const T* dPhi,
    T* dOut,
    T* dScratchG,
    Grid2D grid,
    cudaStream_t stream = 0
    );

template <typename T>
void quadraticCTransformCPU3D(
  const T* Xaxis0,  // shape (nx0,)
  const T* Xaxis1,  // shape (nx1,)
  const T* Xaxis2,  // shape (nx2,)
  const T* Yaxis0,  // shape (ny0,)
  const T* Yaxis1,  // shape (ny1,)
  const T* Yaxis2,  // shape (ny2,)
  const T* Phi,     // shape (nx0, nx1, nx2)
  T* out,           // shape (ny0, ny1, ny2)
  Grid3D grid
  );
template <typename T>
void quadraticCTransform3D(
  const T* Xaxis0,
  const T* Xaxis1,
  const T* Xaxis2,
  const T* Yaxis0,
  const T* Yaxis1,
  const T* Yaxis2,
  const T* Phi,
  T* Out,
  Grid3D grid
  );
template <typename T>
void quadraticCTransform3D_launch(
  const T* dXaxis0,         // DEVICE pointers
  const T* dXaxis1,
  const T* dXaxis2,
  const T* dYaxis0,
  const T* dYaxis1,
  const T* dYaxis2,
  const T* dPhi,
  T* dOut,
  Grid3D grid,
  cudaStream_t stream = 0
  );
