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
