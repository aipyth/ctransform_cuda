// Exercises the device-pointer launch layer (*_launch) directly on pre-staged
// device buffers, bypassing the host wrapper's alloc/copy/sync. This is the
// code path an external caller (e.g. a JAX/XLA FFI shim, see
// docs/engineering/jax_ffi_integration.md) actually uses.
//
// Unlike the other test_*.cpp files, this must be a .cu file: it stages device
// buffers itself via cuda_runtime.h/cuda_utils.cuh, which the g++-compiled
// .cpp tests deliberately avoid (see test_strategy.md).

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

#include "ctransform.hpp"
#include "cuda_utils.cuh"
#include "helpers_3d.hpp"

template <typename T>
static T maxAbsErr(const std::vector<T>& a, const std::vector<T>& b) {
    T err = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

// ---- 1D ----

TEST(LaunchDirect1D, ZeroPhi) {
    std::vector<double> X   = {-0.5, 0.0, 0.5};
    std::vector<double> Y   = {-0.3, 0.0, 0.3, 0.6};
    std::vector<double> phi(X.size(), 0.0);
    Grid1D grid(X.size(), Y.size());

    std::vector<double> cpu(Y.size());
    quadraticCTransformCPU1D(X.data(), Y.data(), phi.data(), cpu.data(), grid);

    DeviceBuffer<double> dX(X.size()), dY(Y.size()), dPhi(X.size()), dOut(Y.size());
    CUDA_CHECK(cudaMemcpy(dX.get(),   X.data(),   X.size()   * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY.get(),   Y.data(),   Y.size()   * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPhi.get(), phi.data(), phi.size() * sizeof(double), cudaMemcpyHostToDevice));

    quadraticCTransform1D_launch<double>(dX.get(), dY.get(), dPhi.get(), dOut.get(), grid, /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gpu(Y.size());
    CUDA_CHECK(cudaMemcpy(gpu.data(), dOut.get(), Y.size() * sizeof(double), cudaMemcpyDeviceToHost));

    EXPECT_LT(maxAbsErr(cpu, gpu), 1e-12);
}

TEST(LaunchDirect1D, NonZeroPhi) {
    std::vector<double> X   = {-1.0, 0.0, 1.0};
    std::vector<double> Y   = {-1.0, 0.0, 1.0, 2.0};
    std::vector<double> phi = { 0.2, -0.5, 0.6};
    Grid1D grid(X.size(), Y.size());

    std::vector<double> cpu(Y.size());
    quadraticCTransformCPU1D(X.data(), Y.data(), phi.data(), cpu.data(), grid);

    DeviceBuffer<double> dX(X.size()), dY(Y.size()), dPhi(X.size()), dOut(Y.size());
    CUDA_CHECK(cudaMemcpy(dX.get(),   X.data(),   X.size()   * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY.get(),   Y.data(),   Y.size()   * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPhi.get(), phi.data(), phi.size() * sizeof(double), cudaMemcpyHostToDevice));

    quadraticCTransform1D_launch<double>(dX.get(), dY.get(), dPhi.get(), dOut.get(), grid, /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gpu(Y.size());
    CUDA_CHECK(cudaMemcpy(gpu.data(), dOut.get(), Y.size() * sizeof(double), cudaMemcpyDeviceToHost));

    EXPECT_LT(maxAbsErr(cpu, gpu), 1e-12);
}

// ---- 2D ----

TEST(LaunchDirect2D, ZeroPhi) {
    std::vector<double> X0 = {-0.5, 0.0, 0.5};
    std::vector<double> X1 = {0.0, 0.5, 1.0};
    std::vector<double> Y0 = {-0.3, 0.0, 0.3};
    std::vector<double> Y1 = {0.1, 0.5, 0.9};
    std::vector<double> phi(X0.size() * X1.size(), 0.0);
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::size_t nout = Y0.size() * Y1.size();
    std::vector<double> cpu(nout);
    quadraticCTransformCPU2D(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), cpu.data(), grid);

    DeviceBuffer<double> dX0(X0.size()), dX1(X1.size()), dY0(Y0.size()), dY1(Y1.size());
    DeviceBuffer<double> dPhi(phi.size()), dOut(nout);
    CUDA_CHECK(cudaMemcpy(dX0.get(),  X0.data(),  X0.size()  * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dX1.get(),  X1.data(),  X1.size()  * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY0.get(),  Y0.data(),  Y0.size()  * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY1.get(),  Y1.data(),  Y1.size()  * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPhi.get(), phi.data(), phi.size() * sizeof(double), cudaMemcpyHostToDevice));

    quadraticCTransform2D_launch<double>(
        dX0.get(), dX1.get(), dY0.get(), dY1.get(), dPhi.get(), dOut.get(), grid, /*stream=*/0
        );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gpu(nout);
    CUDA_CHECK(cudaMemcpy(gpu.data(), dOut.get(), nout * sizeof(double), cudaMemcpyDeviceToHost));

    EXPECT_LT(maxAbsErr(cpu, gpu), 1e-12);
}

TEST(LaunchDirect2D, NonSquareGrid) {
    // ny0 != ny1: catches a blocks/threads axis-swap bug in *_launch specifically
    // (same fixture as CudaVsCPU2D.NonSquareGrid).
    std::vector<double> X0 = {0.0, 1.0};
    std::vector<double> X1 = {0.0, 1.0, 2.0};
    std::vector<double> Y0 = {0.25, 0.5, 0.75};         // ny0 = 3
    std::vector<double> Y1 = {0.1, 0.5, 0.9, 1.5};     // ny1 = 4
    std::vector<double> phi(X0.size() * X1.size(), 0.0);
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::size_t nout = Y0.size() * Y1.size();
    std::vector<double> cpu(nout);
    quadraticCTransformCPU2D(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), cpu.data(), grid);

    DeviceBuffer<double> dX0(X0.size()), dX1(X1.size()), dY0(Y0.size()), dY1(Y1.size());
    DeviceBuffer<double> dPhi(phi.size()), dOut(nout);
    CUDA_CHECK(cudaMemcpy(dX0.get(),  X0.data(),  X0.size()  * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dX1.get(),  X1.data(),  X1.size()  * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY0.get(),  Y0.data(),  Y0.size()  * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY1.get(),  Y1.data(),  Y1.size()  * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPhi.get(), phi.data(), phi.size() * sizeof(double), cudaMemcpyHostToDevice));

    quadraticCTransform2D_launch<double>(
        dX0.get(), dX1.get(), dY0.get(), dY1.get(), dPhi.get(), dOut.get(), grid, /*stream=*/0
        );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gpu(nout);
    CUDA_CHECK(cudaMemcpy(gpu.data(), dOut.get(), nout * sizeof(double), cudaMemcpyDeviceToHost));

    EXPECT_LT(maxAbsErr(cpu, gpu), 1e-12);
}

// ---- 3D ----
// These call quadraticCTransform3D_launch directly on device buffers the test allocates
// itself, the way a GPU-resident solver would.

namespace {

// The 3D problem already copied to the device, plus a device output buffer.
struct Device3D {
    Grid3D grid;
    DeviceBuffer<double> x0, x1, x2, y0, y1, y2, phi, out;

    Device3D(const t3d::Axes3<double>& X, const t3d::Axes3<double>& Y,
             const std::vector<double>& hostPhi)
        : grid{X.a0.size(), X.a1.size(), X.a2.size(), Y.a0.size(), Y.a1.size(), Y.a2.size()},
          x0(X.a0.size()), x1(X.a1.size()), x2(X.a2.size()),
          y0(Y.a0.size()), y1(Y.a1.size()), y2(Y.a2.size()),
          phi(hostPhi.size()), out(Y.size()) {
        upload(x0, X.a0); upload(x1, X.a1); upload(x2, X.a2);
        upload(y0, Y.a0); upload(y1, Y.a1); upload(y2, Y.a2);
        upload(phi, hostPhi);
    }

    // Sets every byte of the output to 0xFF. As a double, that bit pattern is a NaN, so any
    // output entry the kernel forgets to write stays NaN and fails the comparison.
    void poisonOutput() {
        CUDA_CHECK(cudaMemset(out.get(), 0xFF, outputCount() * sizeof(double)));
    }

    void launch(cudaStream_t stream) {
        quadraticCTransform3D_launch<double>(x0.get(), x1.get(), x2.get(),
                                             y0.get(), y1.get(), y2.get(),
                                             phi.get(), out.get(), grid, stream);
    }

    std::vector<double> download() const {
        std::vector<double> h(outputCount());
        CUDA_CHECK(cudaMemcpy(h.data(), out.get(), h.size() * sizeof(double), cudaMemcpyDeviceToHost));
        return h;
    }

    std::size_t outputCount() const { return grid.ny0 * grid.ny1 * grid.ny2; }

private:
    static void upload(DeviceBuffer<double>& d, const std::vector<double>& h) {
        CUDA_CHECK(cudaMemcpy(d.get(), h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
};

}  // namespace

TEST(LaunchDirect3D, PoisonedOutputFullyWritten) {
    // 11 * 13 * 37 outputs end in a partial block, the place where an entry is most likely
    // to be skipped.
    std::mt19937_64 rng(41);
    const t3d::Axes3<double> X = t3d::randomAxes(5, 7, 9, rng);
    const t3d::Axes3<double> Y = t3d::randomAxes(11, 13, 37, rng);
    const std::vector<double> phi = t3d::uniform(X.size(), rng, -1.0, 1.0);
    const std::vector<double> cpu = t3d::run(&quadraticCTransformCPU3D<double>, X, Y, phi);

    Device3D d(X, Y, phi);
    d.poisonOutput();
    d.launch(/*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<double> gpu = d.download();

    for (std::size_t i = 0; i < gpu.size(); ++i)
        ASSERT_FALSE(std::isnan(gpu[i])) << "output " << i << " was never written";
    EXPECT_LT(t3d::maxAbsErr(cpu, gpu), 1e-12);
}

TEST(LaunchDirect3D, NonDefaultStream) {
    // Launch on a caller-created stream and wait on that stream only. On its own this test
    // can pass even if the launch layer ignores its stream argument and runs on stream 0.
    // That mistake is caught by the host-wrapper tests instead: the wrapper copies the
    // result back on its own non-blocking stream, which does not wait for stream 0.
    std::mt19937_64 rng(43);
    const t3d::Axes3<double> X = t3d::randomAxes(4, 6, 5, rng);
    const t3d::Axes3<double> Y = t3d::randomAxes(7, 3, 9, rng);
    const std::vector<double> phi = t3d::uniform(X.size(), rng, -1.0, 1.0);
    const std::vector<double> cpu = t3d::run(&quadraticCTransformCPU3D<double>, X, Y, phi);

    Device3D d(X, Y, phi);
    d.poisonOutput();
    CudaStream stream;
    d.launch(stream.get());
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    EXPECT_LT(t3d::maxAbsErr(cpu, d.download()), 1e-12);
}

TEST(LaunchDirect3D, EmptyTargetIsNoOp) {
    // ny1 = 0 means zero output points. The launch layer must return without launching:
    // a launch with zero blocks is an error (cudaErrorInvalidConfiguration).
    const t3d::Axes3<double> X{{0.0, 1.0}, {0.5}, {0.5}};
    const t3d::Axes3<double> Y{{0.0}, {}, {1.0}};
    const std::vector<double> phi(X.size(), 0.0);

    Device3D d(X, Y, phi);
    EXPECT_NO_THROW(d.launch(/*stream=*/0));
    CUDA_CHECK(cudaDeviceSynchronize());
}
