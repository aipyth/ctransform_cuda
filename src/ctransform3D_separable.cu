#include <cuda_runtime.h>
#include "ctransform.hpp"
#include "cuda_utils.cuh"


std::size_t quadraticCTransform3DSeparable_scratchSize(Grid3D grid) {
  return grid.nx0 * grid.nx1 * grid.ny2 + grid.nx0 * grid.ny1 * grid.ny2;
}

template <typename T>
__global__ void collapseAxisKernel(
    const T* __restrict__ dXaxis,   // shape = (n,)
    const T* __restrict__ dYaxis,   // shape = (m,)
    const T* __restrict__ dIn,      // (B, n, F), row-major
    T* __restrict__ dOut,           // (B, m, F), row-major
    std::size_t B, std::size_t n, std::size_t m, std::size_t F,
    bool subtractInput
    ) {

  std::size_t t = threadIdx.x + blockDim.x * static_cast<std::size_t>(blockIdx.x);
  if (t >= B * m * F) return;

  std::size_t f = t % F;
  std::size_t j = (t / F) % m;
  std::size_t b = t / (m * F);

  T y = dYaxis[j];
  T best = INFINITY;
  T v, d;

  for (std::size_t i = 0; i < n; ++i) {
    v = dIn[(b*n + i)*F + f];
    d = dXaxis[i] - y;
    best = min(best, T(0.5) * d * d + (subtractInput ? -v : v));
  }
  dOut[t] = best;
}

template <typename T>
void quadraticCTransform3DSeparable_launch(
    const T* dXaxis0, const T* dXaxis1, const T* dXaxis2,
    const T* dYaxis0, const T* dYaxis1, const T* dYaxis2,
    const T* Phi, T* out, T* dScratch,
    Grid3D grid, cudaStream_t stream
    ) {
  if (grid.ny0 * grid.ny1 * grid.ny2 == 0) return;
  std::size_t N;
  std::size_t threads = 256;
  std::size_t blocks;
  // Pass 1: reduction over raw phi[x_0, x_1, x_2]
  N = grid.nx0 * grid.nx1 * grid.ny2;
  blocks = (N + threads - 1) / threads;
  if (blocks > kMaxGridDimX)
    throw std::runtime_error("grid too large");
  if (N > 0) {
    // B = nx_0 nx_1, n = nx_2, m = ny_2, F = 1
    collapseAxisKernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        dXaxis2, dYaxis2, Phi, dScratch,
        grid.nx0 * grid.nx1, grid.nx2, grid.ny2, 1, true);
    CUDA_CHECK(cudaGetLastError());
  }

  // Pass 2: reduction over scratch g_1[x_0, x_1, y_2]
  N = grid.nx0 * grid.ny1 * grid.ny2;
  blocks = (N + threads - 1) / threads;
  if (blocks > kMaxGridDimX)
    throw std::runtime_error("grid too large");
  T* dScratchOffsetted = dScratch + grid.nx0 * grid.nx1 * grid.ny2;
  if (N > 0) {
    // B = nx_0, n = nx_1, m = ny_1, F = ny_2
    collapseAxisKernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        dXaxis1, dYaxis1, dScratch, dScratchOffsetted,
        grid.nx0, grid.nx1, grid.ny1, grid.ny2, false);
    CUDA_CHECK(cudaGetLastError());
  }
  
  // Pass 3: reduction over scratch g_2[x_0, y_1, y_2]
  N = grid.ny0 * grid.ny1 * grid.ny2;
  blocks = (N + threads - 1) / threads;
  if (blocks > kMaxGridDimX)
    throw std::runtime_error("grid too large");
  // B = 1, n = nx_0, m = ny_0, F = ny_1 ny_2
  collapseAxisKernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      dXaxis0, dYaxis0, dScratchOffsetted, out,
      1, grid.nx0, grid.ny0, grid.ny1 * grid.ny2, false);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void quadraticCTransform3DSeparable(
    const T* Xaxis0, const T* Xaxis1, const T* Xaxis2,
    const T* Yaxis0, const T* Yaxis1, const T* Yaxis2,
    const T* Phi, T* out, Grid3D grid
    ) {
  CudaStream s;
  DeviceBuffer<T> dXaxis0(grid.nx0);
  DeviceBuffer<T> dXaxis1(grid.nx1);
  DeviceBuffer<T> dXaxis2(grid.nx2);
  DeviceBuffer<T> dYaxis0(grid.ny0);
  DeviceBuffer<T> dYaxis1(grid.ny1);
  DeviceBuffer<T> dYaxis2(grid.ny2);
  DeviceBuffer<T> dPhi(grid.nx0 * grid.nx1 * grid.nx2);
  DeviceBuffer<T> dOut(grid.ny0 * grid.ny1 * grid.ny2);
  DeviceBuffer<T> dScratch(quadraticCTransform3DSeparable_scratchSize(grid));

  CUDA_CHECK(cudaMemcpyAsync(dXaxis0.get(), Xaxis0, grid.nx0 * sizeof(T), cudaMemcpyHostToDevice, s.get()));
  CUDA_CHECK(cudaMemcpyAsync(dXaxis1.get(), Xaxis1, grid.nx1 * sizeof(T), cudaMemcpyHostToDevice, s.get()));
  CUDA_CHECK(cudaMemcpyAsync(dXaxis2.get(), Xaxis2, grid.nx2 * sizeof(T), cudaMemcpyHostToDevice, s.get()));
  CUDA_CHECK(cudaMemcpyAsync(dYaxis0.get(), Yaxis0, grid.ny0 * sizeof(T), cudaMemcpyHostToDevice, s.get()));
  CUDA_CHECK(cudaMemcpyAsync(dYaxis1.get(), Yaxis1, grid.ny1 * sizeof(T), cudaMemcpyHostToDevice, s.get()));
  CUDA_CHECK(cudaMemcpyAsync(dYaxis2.get(), Yaxis2, grid.ny2 * sizeof(T), cudaMemcpyHostToDevice, s.get()));
  CUDA_CHECK(cudaMemcpyAsync(dPhi.get(), Phi, grid.nx0 * grid.nx1 * grid.nx2 * sizeof(T), cudaMemcpyHostToDevice, s.get()));

  quadraticCTransform3DSeparable_launch(
      dXaxis0.get(), dXaxis1.get(), dXaxis2.get(),
      dYaxis0.get(), dYaxis1.get(), dYaxis2.get(),
      dPhi.get(), dOut.get(), dScratch.get(), grid, s.get());
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(out, dOut.get(), grid.ny0*grid.ny1*grid.ny2*sizeof(T), cudaMemcpyDeviceToHost, s.get()));
  CUDA_CHECK(cudaStreamSynchronize(s.get()));
}

template void quadraticCTransform3DSeparable(const float *Xaxis0, const float *Xaxis1, const float *Xaxis2, const float *Yaxis0, const float *Yaxis1, const float *Yaxis2, const float *Phi, float *Out, Grid3D grid);
template void quadraticCTransform3DSeparable(const double *Xaxis0, const double *Xaxis1, const double *Xaxis2, const double *Yaxis0, const double *Yaxis1, const double *Yaxis2, const double *Phi, double *Out, Grid3D grid);

template void quadraticCTransform3DSeparable_launch(const float *dXaxis0, const float *dXaxis1, const float *dXaxis2, const float *dYaxis0, const float *dYaxis1, const float *dYaxis2, const float *dPhi, float *dOut, float *dScratch, Grid3D grid, cudaStream_t);
template void quadraticCTransform3DSeparable_launch(const double *dXaxis0, const double *dXaxis1, const double *dXaxis2, const double *dYaxis0, const double *dYaxis1, const double *dYaxis2, const double *dPhi, double *dOut, double *dScratch, Grid3D grid, cudaStream_t);
