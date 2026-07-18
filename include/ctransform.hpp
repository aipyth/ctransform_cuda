#pragma once

#include <cstddef>

struct Grid1D { std::size_t nx, ny; };

struct Grid2D { std::size_t nx0, nx1, ny0, ny1; };


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
