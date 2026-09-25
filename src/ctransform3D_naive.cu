#include <cuda_runtime.h>
#include <stdexcept>

#include "ctransform.hpp"
#include "cuda_utils.cuh"


template <typename T>
__global__ void quadraticCTransform3DKernel(
  const T* dXaxis0, const T* dXaxis1, const T* dXaxis2,
  const T* dYaxis0, const T* dYaxis1, const T* dYaxis2,
  const T* dPhi, T* dOut, Grid3D grid
  ) {
  std::size_t N = grid.ny0 * grid.ny1 * grid.ny2;
  std::size_t t = threadIdx.x + blockDim.x * static_cast<std::size_t>(blockIdx.x);
  if (t >= N) return;
  std::size_t iy2 = t % grid.ny2;
  std::size_t iy1 = (t / grid.ny2) % grid.ny1;
  std::size_t iy0 = t / (grid.ny1 * grid.ny2);

  T yi0 = dYaxis0[iy0];
  T yi1 = dYaxis1[iy1];
  T yi2 = dYaxis2[iy2];

  T d0, d1, d2;
  T best = INFINITY;

  for (std::size_t ix0 = 0; ix0 < grid.nx0; ++ix0) {
    d0 = dXaxis0[ix0] - yi0;
    for (std::size_t ix1 = 0; ix1 < grid.nx1; ++ix1) {
      d1 = dXaxis1[ix1] - yi1;
      for (std::size_t ix2 = 0; ix2 < grid.nx2; ++ix2) {
        d2 = dXaxis2[ix2] - yi2;
        best = min(best,
                   T(0.5)*(d0*d0 + d1*d1 + d2*d2)
                   - dPhi[(ix0*grid.nx1 + ix1)*grid.nx2 + ix2]);
      }
    }
  }

  dOut[t] = best;
}

template <typename T>
void quadraticCTransform3D_launch(
  const T* dXaxis0, const T* dXaxis1, const T* dXaxis2,
  const T* dYaxis0, const T* dYaxis1, const T* dYaxis2,
  const T* dPhi, T* dOut, Grid3D grid,
  cudaStream_t stream
  ) {
  std::size_t N = grid.ny0 * grid.ny1 * grid.ny2;
  if (N == 0) return;
  std::size_t threads = 256;
  std::size_t blocks = (N + threads - 1) / threads;
  if (blocks > kMaxGridDimX)
    throw std::runtime_error("grid too large");

  quadraticCTransform3DKernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      dXaxis0, dXaxis1, dXaxis2,
      dYaxis0, dYaxis1, dYaxis2,
      dPhi, dOut, grid);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void quadraticCTransform3D(
  const T* Xaxis0, const T* Xaxis1, const T* Xaxis2,
  const T* Yaxis0, const T* Yaxis1, const T* Yaxis2,
  const T* Phi, T* Out, Grid3D grid
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

  CUDA_CHECK(cudaMemcpyAsync(dXaxis0.get(), Xaxis0, grid.nx0*sizeof(T), cudaMemcpyHostToDevice, s.get()));
  CUDA_CHECK(cudaMemcpyAsync(dXaxis1.get(), Xaxis1, grid.nx1*sizeof(T), cudaMemcpyHostToDevice, s.get()));
  CUDA_CHECK(cudaMemcpyAsync(dXaxis2.get(), Xaxis2, grid.nx2*sizeof(T), cudaMemcpyHostToDevice, s.get()));
  CUDA_CHECK(cudaMemcpyAsync(dYaxis0.get(), Yaxis0, grid.ny0*sizeof(T), cudaMemcpyHostToDevice, s.get()));
  CUDA_CHECK(cudaMemcpyAsync(dYaxis1.get(), Yaxis1, grid.ny1*sizeof(T), cudaMemcpyHostToDevice, s.get()));
  CUDA_CHECK(cudaMemcpyAsync(dYaxis2.get(), Yaxis2, grid.ny2*sizeof(T), cudaMemcpyHostToDevice, s.get()));
  CUDA_CHECK(cudaMemcpyAsync(dPhi.get(), Phi, grid.nx0*grid.nx1*grid.nx2*sizeof(T), cudaMemcpyHostToDevice, s.get()));

  quadraticCTransform3D_launch(
      dXaxis0.get(), dXaxis1.get(), dXaxis2.get(),
      dYaxis0.get(), dYaxis1.get(), dYaxis2.get(),
      dPhi.get(), dOut.get(), grid, /*stream=*/s.get()
      );
  CUDA_CHECK(cudaMemcpyAsync(Out, dOut.get(), grid.ny0*grid.ny1*grid.ny2*sizeof(T), cudaMemcpyDeviceToHost, s.get()));
  CUDA_CHECK(cudaStreamSynchronize(s.get()));
}


template void quadraticCTransform3D(
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, float*, Grid3D);
template void quadraticCTransform3D(
    const double*, const double*, const double*,
    const double*, const double*, const double*,
    const double*, double*, Grid3D);

template void quadraticCTransform3D_launch(
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, float*, Grid3D, cudaStream_t);
template void quadraticCTransform3D_launch(
    const double*, const double*, const double*,
    const double*, const double*, const double*,
    const double*, double*, Grid3D, cudaStream_t);
