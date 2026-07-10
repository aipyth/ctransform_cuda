#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "print_utils.hpp"
#include "cuda_utils.cuh"

#include "ctransform.hpp"



// === Naive 1D kernel - c-tranform ===
// the Kantorovich dual admits the following c-tranform
// for a potentials phi(x)
//      phi^c (y) = min_x c(x, y) - phi(x)
// 
// Assuming the quadratic cost
//      c(x, y) = 1/2 | x - y |^2
// one obtains
//      phi^c _{iy} = min_{ix} 1/2 (x_{ix} - y_{iy})^2 - phi_{ix}

template <typename T>
__global__ void quadraticCTransform1DKernel (
        const T* __restrict__ X,     // target coords
        const T* __restrict__ Y,     // source coords
        const T* __restrict__ Phi,   // source values
        T* out,
        const Grid1D grid
        ) {
    int iy = threadIdx.x + blockDim.x * blockIdx.x;
    if (iy >= grid.ny) return;

    T best = INFINITY;
    T yi = Y[iy];       // load INTO REGISTRY !
    T dx;
    T candidate;

    for (int ix = 0; ix < grid.nx; ix++) {
        dx = X[ix] - yi;
        candidate = 0.5 * dx * dx - Phi[ix];
        best = min(best, candidate);
    }

    out[iy] = best;
}

template <typename T>
void quadraticCTransform(
    const T* X,
    const T* Y,
    const T* phi,
    T* out,
    Grid1D grid
) {
    DeviceBuffer<T> devX(grid.nx);
    DeviceBuffer<T> devY(grid.ny);
    DeviceBuffer<T> devPhi(grid.nx);
    DeviceBuffer<T> devOut(grid.ny);

    CUDA_CHECK(cudaMemcpy(devX.get(), X, grid.nx * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devY.get(), Y, grid.ny * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devPhi.get(), phi, grid.nx * sizeof(T), cudaMemcpyHostToDevice));

    // call the kernel
    int threads = 256;
    int blocks = (static_cast<int>(grid.ny) + threads - 1) / threads;
    quadraticCTransform1DKernel<T><<<blocks, threads>>>(
        devX.get(), devY.get(), devPhi.get(), devOut.get(), grid
        );
    
    // check for errors and copy the result
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, devOut.get(), grid.ny * sizeof(T), cudaMemcpyDeviceToHost));
}

template void quadraticCTransform(const float*, const float*, const float*, float*, Grid1D);
template void quadraticCTransform(const double*, const double*, const double*, double*, Grid1D);
