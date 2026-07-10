#include <iostream>
#include <vector>
#include <limits>
#include <cmath>

#include "ctransform.hpp"


template <typename T>
void quadraticCTransformCPU (
    const T* Xaxis,     // shape (nx,)
    const T* Yaxis,     // shape (ny,)
    const T* Phi,       // shape (nx,)
    T* out,             // shape (ny,)
    Grid1D grid
) {
    if (grid.nx == 0) return;

    for (std::size_t j = 0; j < grid.ny; j++) {
        T yi = Yaxis[j];
        T best = std::numeric_limits<T>::infinity();

        for (std::size_t i = 0; i < grid.nx; i++) {
            T dx = Xaxis[i] - yi;
            T candidate = 0.5 * dx * dx - Phi[i];
            best = std::fmin(best, candidate);
        }

        out[j] = best;
    }
}

template <typename T>
void quadraticCTransformCPU (
    const T* Xaxis0,     // shape (nx0,)
    const T* Xaxis1,     // shape (nx1,)
    const T* Yaxis0,     // shape (ny0,)
    const T* Yaxis1,     // shape (ny1,)
    const T* Phi,       // shape (nx0, nx1)
    T* out,             // shape (ny0, ny1)
    Grid2D grid
) {

    T yi0, yi1;
    T best;
    T d0, d1;
    T candidate;

    for (std::size_t iy0 = 0; iy0 < grid.ny0; iy0++) {
        yi0 = Yaxis0[iy0];
        for (std::size_t iy1 = 0; iy1 < grid.ny1; iy1++) {
            yi1 = Yaxis1[iy1];
            best = std::numeric_limits<T>::infinity();
            for (std::size_t ix0 = 0; ix0 < grid.nx0; ix0++) {
                d0 = Xaxis0[ix0] - yi0;
                for (std::size_t ix1 = 0; ix1 < grid.nx1; ix1++) {
                    d1 = Xaxis1[ix1] - yi1;
                    candidate = 0.5 * (d0 * d0 + d1 * d1) - Phi[ix0 * grid.nx1 + ix1];
                    best = std::fmin(best, candidate);
                }
            }
            out[iy0 * grid.ny1 + iy1] = best;
        }
    }
}

template void quadraticCTransformCPU(const float*, const float*, const float*, float*, Grid1D);
template void quadraticCTransformCPU(const double*, const double*, const double*, double*, Grid1D);

template void quadraticCTransformCPU(const  float*, const float*, const float*, const float*, const float*, float*, Grid2D);
template void quadraticCTransformCPU(const  double*, const double*, const double*, const double*, const double*, double*, Grid2D);
