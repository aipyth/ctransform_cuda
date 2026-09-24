#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include "ctransform.hpp"
#include "cuda_utils.cuh"

namespace {

using Clock = std::chrono::high_resolution_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void linspace(std::vector<double>& v, double a, double b) {
    const std::size_t n = v.size();
    for (std::size_t i = 0; i < n; ++i)
        v[i] = (n == 1)
             ? a
             : a + (b - a) * static_cast<double>(i) / static_cast<double>(n - 1);
}

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

// Times `launch` (a callable taking a cudaStream_t) with kw warm-up
// iterations discarded, then kr timed iterations; returns ms per iteration.
template <typename Launch>
double timeKernelMs(Launch&& launch, cudaStream_t stream,
                    std::size_t kw, std::size_t kr) {
    for (std::size_t i = 0; i < kw; ++i) launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CudaEvent start, stop;
    CUDA_CHECK(cudaEventRecord(start.get(), stream));
    for (std::size_t i = 0; i < kr; ++i) launch(stream);
    CUDA_CHECK(cudaEventRecord(stop.get(), stream));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));

    float ms = 0.0f;   // cudaEventElapsedTime writes a float, not a double
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));
    return static_cast<double>(ms) / static_cast<double>(kr);
}

// diff < 0 means "no CPU reference was run for this configuration".
void reportRow(const char* label, std::size_t nout,
               double ms_kernel, double ms_e2e, double ms_cpu, double diff) {
    const double ns_per_out = ms_kernel * 1.0e6 / static_cast<double>(nout);

    char cpu_ratio[32];
    if (ms_cpu > 0.0)
        std::snprintf(cpu_ratio, sizeof cpu_ratio, "%9.1fx", ms_cpu / ms_kernel);
    else
        std::snprintf(cpu_ratio, sizeof cpu_ratio, "%10s", "n/a");

    char err[32];
    if (diff >= 0.0) std::snprintf(err, sizeof err, "%.2e", diff);
    else             std::snprintf(err, sizeof err, "%s", "n/a");

    std::printf("    %-10s kernel %10.4f ms  %9.2f ns/out | e2e %10.3f ms |"
                " cpu/kernel %s | e2e/kernel %7.1fx | err %s\n",
                label, ms_kernel, ns_per_out, ms_e2e, cpu_ratio,
                ms_e2e / ms_kernel, err);
}

// ---------------------------------------------------------------- 1D ------

void bench1D(std::size_t nx, std::size_t ny, std::size_t kw, std::size_t kr) {
    std::vector<double> X(nx), Y(ny), phi(nx, 0.0);
    std::vector<double> out_cpu(ny, 0.0), out_gpu(ny, 0.0);
    linspace(X, -1.0, 1.0);
    linspace(Y, -1.0, 1.0);
    const Grid1D grid{nx, ny};

    auto t0 = Clock::now();
    quadraticCTransformCPU1D(X.data(), Y.data(), phi.data(), out_cpu.data(), grid);
    const double ms_cpu = msSince(t0);

    // One discarded call first: the process's first GPU call pays CUDA
    // context creation, which would otherwise land inside ms_e2e.
    quadraticCTransform1D(X.data(), Y.data(), phi.data(), out_gpu.data(), grid);
    t0 = Clock::now();
    quadraticCTransform1D(X.data(), Y.data(), phi.data(), out_gpu.data(), grid);
    const double ms_e2e = msSince(t0);

    CudaStream s;
    DeviceBuffer<double> dX(nx), dY(ny), dPhi(nx), dOut(ny);
    CUDA_CHECK(cudaMemcpyAsync(dX.get(),   X.data(),   nx * sizeof(double),
                               cudaMemcpyHostToDevice, s.get()));
    CUDA_CHECK(cudaMemcpyAsync(dY.get(),   Y.data(),   ny * sizeof(double),
                               cudaMemcpyHostToDevice, s.get()));
    CUDA_CHECK(cudaMemcpyAsync(dPhi.get(), phi.data(), nx * sizeof(double),
                               cudaMemcpyHostToDevice, s.get()));
    CUDA_CHECK(cudaStreamSynchronize(s.get()));

    const double ms_kernel = timeKernelMs(
        [&](cudaStream_t st) {
            quadraticCTransform1D_launch<double>(
                dX.get(), dY.get(), dPhi.get(), dOut.get(), grid, st);
        },
        s.get(), kw, kr);

    std::printf("1D  nx=%zu ny=%zu  (%zu outputs)  cpu %10.3f ms\n",
                nx, ny, ny, ms_cpu);
    reportRow("naive", ny, ms_kernel, ms_e2e, ms_cpu,
              maxAbsDiff(out_cpu, out_gpu));
}

// ---------------------------------------------------------------- 2D ------

struct Bench2DOpts {
    bool run_cpu   = true;
    bool run_naive = true;
    bool run_sep   = true;
    std::size_t kw = 5;
    std::size_t kr = 50;
};

void bench2D(std::size_t nx0, std::size_t nx1,
             std::size_t ny0, std::size_t ny1, Bench2DOpts o) {
    const std::size_t nout = ny0 * ny1;

    std::vector<double> X0(nx0), X1(nx1), Y0(ny0), Y1(ny1);
    std::vector<double> phi(nx0 * nx1, 0.0);
    std::vector<double> out_cpu(nout, 0.0);
    std::vector<double> out_naive(nout, 0.0), out_sep(nout, 0.0);
    linspace(X0, -1.0, 1.0);
    linspace(X1, -1.0, 1.0);
    linspace(Y0, -1.0, 1.0);
    linspace(Y1, -1.0, 1.0);
    const Grid2D grid{nx0, nx1, ny0, ny1};

    double ms_cpu = 0.0;
    if (o.run_cpu) {
        auto t0 = Clock::now();
        quadraticCTransformCPU2D(X0.data(), X1.data(), Y0.data(), Y1.data(),
                                 phi.data(), out_cpu.data(), grid);
        ms_cpu = msSince(t0);
    }

    std::printf("2D  src %zux%zu tgt %zux%zu  (%zu outputs)  ",
                nx0, nx1, ny0, ny1, nout);
    if (o.run_cpu) std::printf("cpu %10.3f ms\n", ms_cpu);
    else           std::printf("cpu skipped (O(n^4) on host)\n");

    CudaStream s;
    DeviceBuffer<double> dX0(nx0), dX1(nx1), dY0(ny0), dY1(ny1);
    DeviceBuffer<double> dPhi(nx0 * nx1), dOut(nout), dG(nx0 * ny1);
    CUDA_CHECK(cudaMemcpyAsync(dX0.get(),  X0.data(),  nx0 * sizeof(double),
                               cudaMemcpyHostToDevice, s.get()));
    CUDA_CHECK(cudaMemcpyAsync(dX1.get(),  X1.data(),  nx1 * sizeof(double),
                               cudaMemcpyHostToDevice, s.get()));
    CUDA_CHECK(cudaMemcpyAsync(dY0.get(),  Y0.data(),  ny0 * sizeof(double),
                               cudaMemcpyHostToDevice, s.get()));
    CUDA_CHECK(cudaMemcpyAsync(dY1.get(),  Y1.data(),  ny1 * sizeof(double),
                               cudaMemcpyHostToDevice, s.get()));
    CUDA_CHECK(cudaMemcpyAsync(dPhi.get(), phi.data(), nx0 * nx1 * sizeof(double),
                               cudaMemcpyHostToDevice, s.get()));
    CUDA_CHECK(cudaStreamSynchronize(s.get()));

    if (o.run_naive) {
        quadraticCTransform2D(X0.data(), X1.data(), Y0.data(), Y1.data(),
                              phi.data(), out_naive.data(), grid);   // discard
        auto t0 = Clock::now();
        quadraticCTransform2D(X0.data(), X1.data(), Y0.data(), Y1.data(),
                              phi.data(), out_naive.data(), grid);
        const double ms_e2e = msSince(t0);

        const double ms_kernel = timeKernelMs(
            [&](cudaStream_t st) {
                quadraticCTransform2D_launch<double>(
                    dX0.get(), dX1.get(), dY0.get(), dY1.get(),
                    dPhi.get(), dOut.get(), grid, st);
            },
            s.get(), o.kw, o.kr);

        reportRow("naive", nout, ms_kernel, ms_e2e, ms_cpu,
                  o.run_cpu ? maxAbsDiff(out_cpu, out_naive) : -1.0);
    }

    if (o.run_sep) {
        quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(),
                                       phi.data(), out_sep.data(), grid);  // discard
        auto t0 = Clock::now();
        quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(),
                                       phi.data(), out_sep.data(), grid);
        const double ms_e2e = msSince(t0);

        const double ms_kernel = timeKernelMs(
            [&](cudaStream_t st) {
                quadraticCTransform2DSeparable_launch<double>(
                    dX0.get(), dX1.get(), dY0.get(), dY1.get(),
                    dPhi.get(), dOut.get(), dG.get(), grid, st);
            },
            s.get(), o.kw, o.kr);

        reportRow("separable", nout, ms_kernel, ms_e2e, ms_cpu,
                  o.run_cpu ? maxAbsDiff(out_cpu, out_sep) : -1.0);
    }
}

}  // namespace

int main() {
    std::printf("kernel = CUDA events around _launch on staged buffers;"
                " e2e = host wrapper (alloc + copies + sync)\n");
    std::printf("err = max|gpu - cpu reference|, a guard against timing a"
                " broken kernel\n\n");

    bench1D(1000,  1000,  10, 200);
    bench1D(5000,  5000,   5,  50);
    bench1D(10000, 10000,  3,  20);

    std::printf("\n");

    bench2D(32,  32,  32,  32,  {.kw = 10, .kr = 200});
    bench2D(64,  64,  64,  64,  {.kw = 10, .kr = 100});
    bench2D(128, 128, 128, 128, {.kw =  5, .kr =  20});

    // Past 128 the naive path is O(n^4) and the host reference likewise, so
    // both are dropped; the separable path is O(n^3) and is the one that has
    // to reach solver-relevant sizes.
    bench2D(256, 256, 256, 256,
            {.run_cpu = false, .run_naive = false, .kw = 5, .kr = 50});
    bench2D(512, 512, 512, 512,
            {.run_cpu = false, .run_naive = false, .kw = 3, .kr = 20});

    return 0;
}
