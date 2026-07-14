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

static void bench2D(std::size_t nx0, std::size_t nx1, std::size_t ny0, std::size_t ny1) {
    std::vector<double> X0(nx0), X1(nx1), Y0(ny0), Y1(ny1);
    std::vector<double> phi(nx0 * nx1, 0.0);
    std::vector<double> out_cpu(ny0 * ny1), out_naive(ny0 * ny1), out_sep(ny0 * ny1);

    for (std::size_t i = 0; i < nx0; ++i) X0[i] = -1.0 + 2.0 * i / (nx0 - 1);
    for (std::size_t i = 0; i < nx1; ++i) X1[i] = -1.0 + 2.0 * i / (nx1 - 1);
    for (std::size_t j = 0; j < ny0; ++j) Y0[j] = -1.0 + 2.0 * j / (ny0 - 1);
    for (std::size_t j = 0; j < ny1; ++j) Y1[j] = -1.0 + 2.0 * j / (ny1 - 1);
    Grid2D grid{nx0, nx1, ny0, ny1};

    // warm-up (CUDA context init + first-launch cost) — not timed
    quadraticCTransform2D         (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), out_naive.data(), grid);
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), out_sep.data(),   grid);

    auto t0 = std::chrono::high_resolution_clock::now();
    quadraticCTransformCPU2D      (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), out_cpu.data(),   grid);
    auto t1 = std::chrono::high_resolution_clock::now();
    quadraticCTransform2D         (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), out_naive.data(), grid);
    auto t2 = std::chrono::high_resolution_clock::now();
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), out_sep.data(),   grid);
    auto t3 = std::chrono::high_resolution_clock::now();

    double ms_cpu   = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_naive = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double ms_sep   = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::printf("2D  src %zux%zu  tgt %zux%zu   CPU %9.2f ms   naive %8.2f ms   sep %8.2f ms"
                "   sep-vs-naive %.1fx\n",
                nx0, nx1, ny0, ny1, ms_cpu, ms_naive, ms_sep, ms_naive / ms_sep);
}

int main() {
    bench1D(1000,  1000);
    bench1D(5000,  5000);
    bench1D(10000, 10000);
    bench2D(32,  32,  32,  32);
    bench2D(64,  64,  64,  64);
    bench2D(128, 128, 128, 128);
    return 0;
}
