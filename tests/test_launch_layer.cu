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
#include <cuda_runtime.h>

#include "ctransform.hpp"
#include "cuda_utils.cuh"

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
