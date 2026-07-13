#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "ctransform.hpp"

template <typename T>
static T maxAbsErr(const std::vector<T>& a, const std::vector<T>& b) {
    T err = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

TEST(CudaVsCPU1D, ZeroPhi) {
    std::vector<double> X   = {-0.5, 0.0, 0.5};
    std::vector<double> Y   = {-0.3, 0.0, 0.3, 0.6};
    std::vector<double> phi(X.size(), 0.0);
    Grid1D grid(X.size(), Y.size());

    std::vector<double> cpu(Y.size()), gpu(Y.size());
    quadraticCTransformCPU1D(X.data(), Y.data(), phi.data(), cpu.data(), grid);
    quadraticCTransform1D   (X.data(), Y.data(), phi.data(), gpu.data(), grid);

    EXPECT_LT(maxAbsErr(cpu, gpu), 1e-12);
}

TEST(CudaVsCPU1D, NonZeroPhi) {
    std::vector<double> X   = {-1.0, 0.0, 1.0};
    std::vector<double> Y   = {-1.0, 0.0, 1.0, 2.0};
    std::vector<double> phi = { 0.2, -0.5, 0.6};
    Grid1D grid(X.size(), Y.size());

    std::vector<double> cpu(Y.size()), gpu(Y.size());
    quadraticCTransformCPU1D(X.data(), Y.data(), phi.data(), cpu.data(), grid);
    quadraticCTransform1D   (X.data(), Y.data(), phi.data(), gpu.data(), grid);

    EXPECT_LT(maxAbsErr(cpu, gpu), 1e-12);
}

TEST(CudaVsCPU1D, SameGrid) {
    std::vector<double> X   = {0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> phi = {0.1,  0.3, -0.2, 0.0,  0.5};
    Grid1D grid(X.size(), X.size());

    std::vector<double> cpu(X.size()), gpu(X.size());
    quadraticCTransformCPU1D(X.data(), X.data(), phi.data(), cpu.data(), grid);
    quadraticCTransform1D   (X.data(), X.data(), phi.data(), gpu.data(), grid);

    EXPECT_LT(maxAbsErr(cpu, gpu), 1e-12);
}

// ---- 2D CPU vs CUDA tests ----

TEST(CudaVsCPU2D, ZeroPhi) {
    std::vector<double> X0 = {-0.5, 0.0, 0.5};
    std::vector<double> X1 = {0.0, 0.5, 1.0};
    std::vector<double> Y0 = {-0.3, 0.0, 0.3};
    std::vector<double> Y1 = {0.1, 0.5, 0.9};
    std::vector<double> phi(X0.size() * X1.size(), 0.0);
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::size_t nout = Y0.size() * Y1.size();
    std::vector<double> cpu(nout), gpu(nout);
    quadraticCTransformCPU2D(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), cpu.data(), grid);
    quadraticCTransform2D   (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), gpu.data(), grid);

    EXPECT_LT(maxAbsErr(cpu, gpu), 1e-12);
}

TEST(CudaVsCPU2D, NonZeroPhi) {
    std::vector<double> X0 = {-1.0, 0.0, 1.0};
    std::vector<double> X1 = {-1.0, 0.0, 1.0};
    std::vector<double> Y0 = {-0.5, 0.5};
    std::vector<double> Y1 = {-0.5, 0.5};
    std::vector<double> phi = {0.2, -0.1, 0.3, 0.0, 0.5, -0.2, 0.1, 0.4, -0.3};
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::size_t nout = Y0.size() * Y1.size();
    std::vector<double> cpu(nout), gpu(nout);
    quadraticCTransformCPU2D(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), cpu.data(), grid);
    quadraticCTransform2D   (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), gpu.data(), grid);

    EXPECT_LT(maxAbsErr(cpu, gpu), 1e-12);
}

TEST(CudaVsCPU2D, NonSquareGrid) {
    // Specifically tests the ny0 != ny1 case that would catch the blocks swap bug
    std::vector<double> X0 = {0.0, 1.0};
    std::vector<double> X1 = {0.0, 1.0, 2.0};
    std::vector<double> Y0 = {0.25, 0.5, 0.75};         // ny0 = 3
    std::vector<double> Y1 = {0.1, 0.5, 0.9, 1.5};     // ny1 = 4
    std::vector<double> phi(X0.size() * X1.size(), 0.0);
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::size_t nout = Y0.size() * Y1.size();
    std::vector<double> cpu(nout), gpu(nout);
    quadraticCTransformCPU2D(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), cpu.data(), grid);
    quadraticCTransform2D   (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), gpu.data(), grid);

    EXPECT_LT(maxAbsErr(cpu, gpu), 1e-12);
}
