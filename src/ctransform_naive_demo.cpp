#include <vector>

#include "ctransform.hpp"
#include "print_utils.hpp"


int main(void) {
  std::vector<double> X = { -0.5, 0.0, 0.5 };
  std::vector<double> Y = { -0.2, 0.0, 0.2, 0.5 };
  std::vector<double> phi = { 0.2, 0.0, 0.3 };
  std::vector<double> psi(Y.size());

  Grid1D grid { X.size(), Y.size() };

  quadraticCTransform1D(X.data(), Y.data(), phi.data(), psi.data(), grid);

  printVector("X  ", X);
  printVector("Y  ", Y);
  printVector("phi", phi);
  printVector("psi", psi);

  return 0;
}

// int main (void) {


//     std::vector<double> hostX = { -0.5, 0.0, 0.5 };
//     std::vector<double> hostY = { -0.2, 0.0, 0.2, 0.5 };

//     std::size_t nx = hostX.size();
//     std::size_t ny = hostY.size();
//     Grid grid { nx, ny };

//     std::size_t sizeX = grid.nx * sizeof(hostX[0]);
//     std::size_t sizeY = grid.ny * sizeof(hostY[0]);


//     std::vector<double> hostPhi = { 0.2, 0., 0.3 };

//     printVector("x   grid", hostX);
//     printVector("y   grid", hostY);
//     printVector("Phi grid", hostPhi);

//     DeviceBuffer devX(grid.nx);
//     DeviceBuffer devY(grid.ny);
//     DeviceBuffer devPhi(grid.nx);
//     DeviceBuffer devPsi(grid.ny);

//     CUDA_CHECK(cudaMemcpy(devX.get(), hostX.data(), sizeX, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(devY.get(), hostY.data(), sizeY, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(devPhi.get(), hostPhi.data(), sizeX, cudaMemcpyHostToDevice));

//     int threads = 256;
//     int blocks = (grid.ny + threads - 1) / threads;

//     quadraticCTransform1D<<<blocks, threads>>> (devX.get(), devY.get(), devPhi.get(), devPsi.get(), grid);

//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     std::vector<double> hostPsi(grid.ny);

//     CUDA_CHECK(cudaMemcpy(hostPsi.data(), devPsi.get(), sizeY, cudaMemcpyDeviceToHost));

//     printVector("psi", hostPsi);

//     return 0;
// }



// int main (void) {

//     std::vector<TensType> hostXaxis0 = { -0.5 , 0.5 };
//     std::vector<TensType> hostXaxis1 = { 0.0, 1.0, 2.0 };
//     std::vector<TensType> hostYaxis0 = { -0.5 , 0.5 };
//     std::vector<TensType> hostYaxis1 = { -0.5 , 0.5 };

//     std::size_t nx0 = hostXaxis0.size();
//     std::size_t nx1 = hostXaxis1.size();
//     std::size_t ny0 = hostYaxis0.size();
//     std::size_t ny1 = hostYaxis1.size();
//     Grid2D grid { .nx0=nx0, .nx1=nx1, .ny0=ny0, .ny1=ny1 };

//     std::size_t sizeX0 = grid.nx0 * sizeof(hostXaxis0[0]);
//     std::size_t sizeX1 = grid.nx1 * sizeof(hostXaxis1[0]);
//     std::size_t sizeY0 = grid.ny0 * sizeof(hostYaxis0[0]);
//     std::size_t sizeY1 = grid.ny1 * sizeof(hostYaxis1[0]);

//     std::vector<double> hostPhi = {
//         0.2, 0.3, 0.4,
//         0.3, 0.1, 0.2
//     };
//     std::size_t sizePhi = grid.nx0 * grid.nx1 * sizeof(hostPhi[0]);

//     // printVector2D("Xaxis0", hostX, grid.nx0, grid.nx1);
//     // printVector2D("y   grid", hostY, grid.ny0, grid.ny1);
//     printVector2D("Phi grid", hostPhi, grid.nx0, grid.nx1);

//     DeviceBuffer devXaxis0(grid.nx0);
//     DeviceBuffer devXaxis1(grid.nx1);
//     DeviceBuffer devYaxis0(grid.ny0);
//     DeviceBuffer devYaxis1(grid.ny1);
//     DeviceBuffer devPhi(grid.nx0 * grid.nx1);
//     DeviceBuffer devPsi(grid.ny0 * grid.ny1);

//     CUDA_CHECK(cudaMemcpy(devXaxis0.get(), hostXaxis0.data(), sizeX0, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(devXaxis1.get(), hostXaxis1.data(), sizeX1, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(devYaxis0.get(), hostYaxis0.data(), sizeY0, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(devYaxis1.get(), hostYaxis1.data(), sizeY1, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(devPhi.get(), hostPhi.data(), sizePhi, cudaMemcpyHostToDevice));

//     dim3 threads(16, 16);
//     dim3 blocks(
//         (grid.ny0 + threads.x - 1) / threads.x,
//         (grid.ny1 + threads.y - 1) / threads.y
//     );

//     quadraticCTransform2D<<<blocks, threads>>> (
//         devXaxis0.get(), devXaxis1.get(), devYaxis0.get(), devYaxis1.get(),
//         devPhi.get(), devPsi.get(), grid);

//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     std::vector<double> hostPsi(grid.ny0 * grid.ny1);
//     std::size_t sizePsi = grid.ny0 * grid.ny1 * sizeof(hostPsi[0]);

//     CUDA_CHECK(cudaMemcpy(hostPsi.data(), devPsi.get(), sizePsi, cudaMemcpyDeviceToHost));

//     printVector2D("psi", hostPsi, grid.ny0, grid.ny1);

//     return 0;
// }
