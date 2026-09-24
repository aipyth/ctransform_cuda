#include <cuda_runtime.h>

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
    const T* __restrict__ Xaxis,     // target coords
    const T* __restrict__ Yaxis,     // source coords
    const T* __restrict__ Phi,   // source values
    T* out,
    const Grid1D grid
  ) {
  std::size_t iy = threadIdx.x + static_cast<std::size_t>(blockDim.x) * blockIdx.x;
  if (iy >= grid.ny) return;

  T best = INFINITY;
  T yi = Yaxis[iy];       // load INTO REGISTRY !
  T dx;
  T candidate;

  for (std::size_t ix = 0; ix < grid.nx; ix++) {
      dx = Xaxis[ix] - yi;
      candidate = T(0.5) * dx * dx - Phi[ix];
      best = min(best, candidate);
  }

  out[iy] = best;
}

template <typename T>
void quadraticCTransform1D_launch(
    const T* dXaxis, const T* dYaxis,
    const T* dPhi,
    T* dOut,
    const Grid1D grid,
    cudaStream_t stream
  ) {
  int threads = 256;
  std::size_t blocks = (grid.ny + static_cast<std::size_t>(threads) - 1) / threads;
  quadraticCTransform1DKernel<T><<<blocks, threads, 0, stream>>>(
      dXaxis, dYaxis, dPhi, dOut, grid
      );
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void quadraticCTransform1D(
    const T* Xaxis,
    const T* Yaxis,
    const T* phi,
    T* out,
    Grid1D grid
) {
    CudaStream s;
    DeviceBuffer<T> devXaxis(grid.nx);
    DeviceBuffer<T> devYaxis(grid.ny);
    DeviceBuffer<T> devPhi(grid.nx);
    DeviceBuffer<T> devOut(grid.ny);

    CUDA_CHECK(cudaMemcpyAsync(devXaxis.get(), Xaxis, grid.nx * sizeof(T), cudaMemcpyHostToDevice, s.get()));
    CUDA_CHECK(cudaMemcpyAsync(devYaxis.get(), Yaxis, grid.ny * sizeof(T), cudaMemcpyHostToDevice, s.get()));
    CUDA_CHECK(cudaMemcpyAsync(devPhi.get(), phi, grid.nx * sizeof(T), cudaMemcpyHostToDevice, s.get()));

    quadraticCTransform1D_launch<T>(
        devXaxis.get(), devYaxis.get(), devPhi.get(), devOut.get(), grid, /*stream=*/s.get()
        );
    
    CUDA_CHECK(cudaMemcpyAsync(out, devOut.get(), grid.ny * sizeof(T), cudaMemcpyDeviceToHost, s.get()));
    CUDA_CHECK(cudaStreamSynchronize(s.get()));
}

template void quadraticCTransform1D_launch(const double*, const double*, const double*, double*, Grid1D, cudaStream_t);
template void quadraticCTransform1D_launch(const float*, const float*, const float*, float*, Grid1D, cudaStream_t);
template void quadraticCTransform1D(const float*, const float*, const float*, float*, Grid1D);
template void quadraticCTransform1D(const double*, const double*, const double*, double*, Grid1D);
