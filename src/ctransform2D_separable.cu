#include <cuda_runtime.h>

#include "ctransform.hpp"
#include "cuda_utils.cuh"
#include "print_utils.hpp"


template <typename T>
__global__ void separablePass1Kernel (
    const T* __restrict__ dXaxis1,     // DEVICE pointers
    const T* __restrict__ dYaxis1,
    const T* __restrict__ dPhi,
    T* __restrict__ dScratchG,
    Grid2D grid
  ){
  std::size_t iy1 = threadIdx.x + blockDim.x * blockIdx.x;
  std::size_t ix0 = threadIdx.y + blockDim.y * blockIdx.y;
  if (iy1 >= grid.ny1) return;
  if (ix0 >= grid.nx0) return;

  T best = INFINITY;
  T yi1 = dYaxis1[iy1];       // load INTO REGISTRY !
  T dx1;

  for (std::size_t ix1 = 0; ix1 < grid.nx1; ++ix1) {
      dx1 = dXaxis1[ix1] - yi1;
      best = min(best, 0.5 * dx1 * dx1 - dPhi[ix0 * grid.nx1 + ix1]);
  }

  dScratchG[ix0 * grid.ny1 + iy1] = best;
}

template <typename T>
__global__ void separablePass2Kernel (
    const T* __restrict__ dXaxis0,                   // DEVICE pointers
    const T* __restrict__ dYaxis0,
    const T* __restrict__ dScratchG,
    T* dOut,
    Grid2D grid
  ) {
  std::size_t iy1 = threadIdx.x + blockDim.x * blockIdx.x;
  std::size_t iy0 = threadIdx.y + blockDim.y * blockIdx.y;
  if (iy1 >= grid.ny1) return;
  if (iy0 >= grid.ny0) return;

  T best = INFINITY;
  T yi0 = dYaxis0[iy0];       // load INTO REGISTRY !
  T dx0;

  for (std::size_t ix0 = 0; ix0 < grid.nx0; ++ix0) {
      dx0 = dXaxis0[ix0] - yi0;
      best = min(best, 0.5 * dx0 * dx0
          + dScratchG[ix0 * grid.ny1 + iy1]);
  }

  dOut[iy0 * grid.ny1 + iy1] = best;
}

template <typename T>
void quadraticCTransform2DSeparable_launch(
    const T* dXaxis0, const T* dXaxis1,     // DEVICE pointers
    const T* dYaxis0, const T* dYaxis1,
    const T* dPhi,
    T* dOut,
    T* dScratchG,
    Grid2D grid,
    cudaStream_t stream
  ) {
  dim3 threads(16, 16);
  // PASS 1 sized over (ny1, nx0)
  dim3 blocks1(
      (grid.ny1 + threads.x - 1) / threads.x,
      (grid.nx0 + threads.y - 1) / threads.y
      );
  separablePass1Kernel<<<blocks1, threads, 0, stream>>>(
      dXaxis1, dYaxis1, dPhi, dScratchG, grid
      );
  CUDA_CHECK(cudaGetLastError());
  // PASS 2 sized over (ny1, ny0)
  dim3 blocks2(
      (grid.ny1 + threads.x - 1) / threads.x,
      (grid.ny0 + threads.y - 1) / threads.y
      );
  separablePass2Kernel<<<blocks2, threads, 0, stream>>>(
      dXaxis0, dYaxis0, dScratchG, dOut, grid
      );
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void quadraticCTransform2DSeparable(
    const T* Xaxis0, const T* Xaxis1,
    const T* Yaxis0, const T* Yaxis1,
    const T* Phi,
    T* out,
    Grid2D grid
    ) {
    DeviceBuffer<T> devX0(grid.nx0);
    DeviceBuffer<T> devX1(grid.nx1);
    DeviceBuffer<T> devY0(grid.ny0);
    DeviceBuffer<T> devY1(grid.ny1);
    DeviceBuffer<T> devPhi(grid.nx0 * grid.nx1);
    DeviceBuffer<T> devG(grid.nx0 * grid.ny1);      // scratch g, device-only
    DeviceBuffer<T> devOut(grid.ny0 * grid.ny1);

    CUDA_CHECK(cudaMemcpy(devX0.get(),  Xaxis0, grid.nx0 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devX1.get(),  Xaxis1, grid.nx1 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devY0.get(),  Yaxis0, grid.ny0 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devY1.get(),  Yaxis1, grid.ny1 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devPhi.get(), Phi,    grid.nx0 * grid.nx1 * sizeof(T), cudaMemcpyHostToDevice));

    quadraticCTransform2DSeparable_launch<T>(
        devX0.get(), devX1.get(),
        devY0.get(), devY1.get(),
        devPhi.get(), devOut.get(), devG.get(),
        grid, /*stream=*/0
        );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, devOut.get(), grid.ny0 * grid.ny1 * sizeof(T), cudaMemcpyDeviceToHost));
}

template void quadraticCTransform2DSeparable(const double*, const double*, const double*, const double*, const double*, double*, Grid2D);
template void quadraticCTransform2DSeparable(const float*, const float*, const float*, const float*, const float*, float*, Grid2D);

template void quadraticCTransform2DSeparable_launch(const double*, const double*, const double*, const double*, const double*, double*, double*, Grid2D, cudaStream_t);
template void quadraticCTransform2DSeparable_launch(const float*, const float*, const float*, const float*, const float*, float*, float*, Grid2D, cudaStream_t);
