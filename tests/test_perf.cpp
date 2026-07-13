#include <vector>
#include <chrono>
#include <cstdio>
#include "ctransform.hpp"

static void bench1D(std::size_t nx, std::size_t ny) {
    std::vector<double> X(nx), Y(ny), phi(nx), out_cpu(ny), out_gpu(ny);
    for (std::size_t i = 0; i < nx; ++i) X[i]   = -1.0 + 2.0 * i / (nx - 1);
    for (std::size_t j = 0; j < ny; ++j) Y[j]   = -1.0 + 2.0 * j / (ny - 1);
    for (std::size_t i = 0; i < nx; ++i) phi[i] = 0.0;
    Grid1D grid{nx, ny};

    auto t0 = std::chrono::high_resolution_clock::now();
    quadraticCTransformCPU1D(X.data(), Y.data(), phi.data(), out_cpu.data(), grid);
    auto t1 = std::chrono::high_resolution_clock::now();
    quadraticCTransform1D   (X.data(), Y.data(), phi.data(), out_gpu.data(), grid);
    auto t2 = std::chrono::high_resolution_clock::now();

    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_gpu = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::printf("1D  nx=%5zu  ny=%5zu   CPU %8.2f ms   GPU %8.2f ms   speedup %.1fx\n",
                nx, ny, ms_cpu, ms_gpu, ms_cpu / ms_gpu);
}

int main() {
    bench1D(1000,  1000);
    bench1D(5000,  5000);
    bench1D(10000, 10000);
    return 0;
}
