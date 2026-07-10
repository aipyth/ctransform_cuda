#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <chrono>

#include "print_utils.hpp"
#include "cuda_utils.cuh"
#include "ctransform.hpp"


// === Naive 2D kernel - c-tranform ===
// the Kantorovich dual admits the following c-tranform
// for a potentials phi(x)
//      phi^c (y) = min_x c(x, y) - phi(x)
// 
// Assuming the quadratic cost
//      c(x, y) = 1/2 || x - y ||^2 = 0.5 ( (x_0 - y_0)^2 + (x_1 - y_1)^2 )
// one obtains
//      phi^c _{iy} = min_{ix}
//                      0.5 ( (x_{ix0} - y_{iy0})^2 + (x_{ix1} - y_{iy1})^2 )
//                      - phi_{ix0 * nx1 + ix1}

template <typename T>
__global__ void quadraticCTransform2DKernel (
        const T* __restrict__ Xaxis0,       // target coords
        const T* __restrict__ Xaxis1,       // target coords
        const T* __restrict__ Yaxis0,       // source coords
        const T* __restrict__ Yaxis1,       // source coords
        const T* __restrict__ Phi,          // source values
        T* out,
        const Grid2D grid
        ) {
    // within a warp axis x (theadIdx.x) is varying faster
    // but we're accessing the out array as iy*grid.ny1 + iy1
    // meaning that we're varying faster in y axis
    // => iy1 is a fast-varying (contiguous) index
    // => to have coaleasced access we need to vary accordingly
    std::size_t iy0 = threadIdx.y + blockDim.y * blockIdx.y;
    std::size_t iy1 = threadIdx.x + blockDim.x * blockIdx.x;

    if (iy0 >= grid.ny0 || iy1 >= grid.ny1) return;

    T best = INFINITY;
    T yi0 = Yaxis0[iy0];       // load INTO REGISTRY !
    T yi1 = Yaxis1[iy1];       // load INTO REGISTRY !
    T d0, d1;

    for (std::size_t ix0 = 0; ix0 < grid.nx0; ix0++) {
        d0 = Xaxis0[ix0] - yi0;
        for (std::size_t ix1 = 0; ix1 < grid.nx1; ix1++) {
            d1 = Xaxis1[ix1] - yi1;
            best = min(
                best,
                0.5 * (d0 * d0 + d1 * d1)
                    - Phi[ix0 * grid.nx1 + ix1]
            );
        }
    }

    out[iy0 * grid.ny1 + iy1] = best;
}

template <typename T>
void quadraticCTransform(
    const T* Xaxis0,
    const T* Xaxis1,
    const T* Yaxis0,
    const T* Yaxis1,
    const T* Phi,
    T* out,
    Grid2D grid
) {
    DeviceBuffer<T> devX0(grid.nx0);
    DeviceBuffer<T> devX1(grid.nx1);
    DeviceBuffer<T> devY0(grid.ny0);
    DeviceBuffer<T> devY1(grid.ny1);
    DeviceBuffer<T> devPhi(grid.nx0 * grid.nx1);
    DeviceBuffer<T> devOut(grid.ny0 * grid.ny1);

    CUDA_CHECK(cudaMemcpy(devX0.get(), Xaxis0, grid.nx0 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devX1.get(), Xaxis1, grid.nx1 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devY0.get(), Yaxis0, grid.ny0 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devY1.get(), Yaxis1, grid.ny1 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devPhi.get(), Phi, grid.nx0 * grid.nx1 * sizeof(T), cudaMemcpyHostToDevice));

    // call the kernel
    dim3 threads(16, 16);
    dim3 blocks(
        (grid.ny1 + threads.x - 1) / threads.x,
        (grid.ny0 + threads.y - 1) / threads.y
        );
    quadraticCTransform2DKernel<T><<<blocks, threads>>>(
        devX0.get(), devX1.get(),
        devY0.get(), devY1.get(),
        devPhi.get(), devOut.get(), grid
        );
    
    // check for errors and copy the result
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, devOut.get(), grid.ny0 * grid.ny1 * sizeof(T), cudaMemcpyDeviceToHost));
}

template void quadraticCTransform(const float*, const float*, const float*, const float*, const float*, float*, Grid2D);
template void quadraticCTransform(const double*, const double*, const double*, const double*, const double*, double*, Grid2D);
