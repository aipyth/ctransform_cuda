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

// ---- 2D separable tests ----
// Exercises quadraticCTransform2DSeparable; see docs/engineering/separable_kernel_spec.md.

// Hand-computed fixtures (reused from test_cpu_reference.cpp, already hand-verified
// there) — these assert a literal expected value, not just agreement with another
// implementation, so a bug shared between the separable and naive/CPU kernels can't
// hide behind a cross-comparison.

TEST(Separable2D, KnownZeroPhi) {
    // Same fixture as CPUReference2D.Separability: phi=0 -> per-axis split
    // dim0 min = 0.125, dim1 min = 0.125 -> 0.25
    std::vector<double> X0 = {-0.5, 0.5};
    std::vector<double> X1 = {0.0, 1.0};
    std::vector<double> Y0 = {0.0};
    std::vector<double> Y1 = {0.5};
    std::vector<double> phi(X0.size() * X1.size(), 0.0);
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::vector<double> out(Y0.size() * Y1.size());
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), out.data(), grid);

    EXPECT_NEAR(out[0], 0.25, 1e-12);
}

TEST(Separable2D, KnownNonZeroPhi) {
    // Same fixture as CPUReference2D.KnownNonZeroPhi:
    // phi^c(0.5,0.5) = 0.25 - max(phi) = 0.25 - 0.3 = -0.05
    std::vector<double> X0 = {0.0, 1.0};
    std::vector<double> X1 = {0.0, 1.0};
    std::vector<double> Y0 = {0.5};
    std::vector<double> Y1 = {0.5};
    std::vector<double> phi = {0.1, 0.3, 0.0, 0.2};
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::vector<double> out(Y0.size() * Y1.size());
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), out.data(), grid);

    EXPECT_NEAR(out[0], -0.05, 1e-12);
}

// ---- Separable vs CPU reference (same fixtures as CudaVsCPU2D) ----

TEST(CudaSeparableVsCPU2D, ZeroPhi) {
    std::vector<double> X0 = {-0.5, 0.0, 0.5};
    std::vector<double> X1 = {0.0, 0.5, 1.0};
    std::vector<double> Y0 = {-0.3, 0.0, 0.3};
    std::vector<double> Y1 = {0.1, 0.5, 0.9};
    std::vector<double> phi(X0.size() * X1.size(), 0.0);
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::size_t nout = Y0.size() * Y1.size();
    std::vector<double> cpu(nout), separable(nout);
    quadraticCTransformCPU2D      (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), cpu.data(), grid);
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), separable.data(), grid);

    EXPECT_LT(maxAbsErr(cpu, separable), 1e-12);
}

TEST(CudaSeparableVsCPU2D, NonZeroPhi) {
    std::vector<double> X0 = {-1.0, 0.0, 1.0};
    std::vector<double> X1 = {-1.0, 0.0, 1.0};
    std::vector<double> Y0 = {-0.5, 0.5};
    std::vector<double> Y1 = {-0.5, 0.5};
    std::vector<double> phi = {0.2, -0.1, 0.3, 0.0, 0.5, -0.2, 0.1, 0.4, -0.3};
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::size_t nout = Y0.size() * Y1.size();
    std::vector<double> cpu(nout), separable(nout);
    quadraticCTransformCPU2D      (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), cpu.data(), grid);
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), separable.data(), grid);

    EXPECT_LT(maxAbsErr(cpu, separable), 1e-12);
}

TEST(CudaSeparableVsCPU2D, NonSquareGrid) {
    // nx0 != ny0 and ny0 != ny1 -- catches a pass-1 launch-grid sizing bug
    // (nx0 vs ny0), see docs/engineering/separable_kernel_spec.md.
    std::vector<double> X0 = {0.0, 1.0};
    std::vector<double> X1 = {0.0, 1.0, 2.0};
    std::vector<double> Y0 = {0.25, 0.5, 0.75};
    std::vector<double> Y1 = {0.1, 0.5, 0.9, 1.5};
    std::vector<double> phi(X0.size() * X1.size(), 0.0);
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::size_t nout = Y0.size() * Y1.size();
    std::vector<double> cpu(nout), separable(nout);
    quadraticCTransformCPU2D      (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), cpu.data(), grid);
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), separable.data(), grid);

    EXPECT_LT(maxAbsErr(cpu, separable), 1e-12);
}

// ---- Separable vs naive GPU kernel (both GPU implementations should agree) ----

TEST(CudaSeparableVsNaive2D, ZeroPhi) {
    std::vector<double> X0 = {-0.5, 0.0, 0.5};
    std::vector<double> X1 = {0.0, 0.5, 1.0};
    std::vector<double> Y0 = {-0.3, 0.0, 0.3};
    std::vector<double> Y1 = {0.1, 0.5, 0.9};
    std::vector<double> phi(X0.size() * X1.size(), 0.0);
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::size_t nout = Y0.size() * Y1.size();
    std::vector<double> naive(nout), separable(nout);
    quadraticCTransform2D         (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), naive.data(), grid);
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), separable.data(), grid);

    EXPECT_LT(maxAbsErr(naive, separable), 1e-12);
}

TEST(CudaSeparableVsNaive2D, NonZeroPhi) {
    std::vector<double> X0 = {-1.0, 0.0, 1.0};
    std::vector<double> X1 = {-1.0, 0.0, 1.0};
    std::vector<double> Y0 = {-0.5, 0.5};
    std::vector<double> Y1 = {-0.5, 0.5};
    std::vector<double> phi = {0.2, -0.1, 0.3, 0.0, 0.5, -0.2, 0.1, 0.4, -0.3};
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::size_t nout = Y0.size() * Y1.size();
    std::vector<double> naive(nout), separable(nout);
    quadraticCTransform2D         (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), naive.data(), grid);
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), separable.data(), grid);

    EXPECT_LT(maxAbsErr(naive, separable), 1e-12);
}

TEST(CudaSeparableVsNaive2D, NonSquareGrid) {
    std::vector<double> X0 = {0.0, 1.0};
    std::vector<double> X1 = {0.0, 1.0, 2.0};
    std::vector<double> Y0 = {0.25, 0.5, 0.75};
    std::vector<double> Y1 = {0.1, 0.5, 0.9, 1.5};
    std::vector<double> phi(X0.size() * X1.size(), 0.0);
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::size_t nout = Y0.size() * Y1.size();
    std::vector<double> naive(nout), separable(nout);
    quadraticCTransform2D         (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), naive.data(), grid);
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), separable.data(), grid);

    EXPECT_LT(maxAbsErr(naive, separable), 1e-12);
}

// ---- Separable edge cases ----

TEST(Separable2D, SingletonSourceAxis1) {
    // nx1 = 1: pass 1 collapses a single-element axis
    std::vector<double> X0 = {-0.5, 0.0, 0.5};
    std::vector<double> X1 = {0.25};                   // nx1 = 1
    std::vector<double> Y0 = {-0.2, 0.3};
    std::vector<double> Y1 = {0.0, 0.5, 1.0};
    std::vector<double> phi = {0.1, -0.2, 0.3};         // nx0*nx1 = 3
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::size_t nout = Y0.size() * Y1.size();
    std::vector<double> cpu(nout), sep(nout);
    quadraticCTransformCPU2D      (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), cpu.data(), grid);
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), sep.data(), grid);
    EXPECT_LT(maxAbsErr(cpu, sep), 1e-12);
}

TEST(Separable2D, SingletonTargetAxis0) {
    // ny0 = 1: pass 2 output is a single row
    std::vector<double> X0 = {0.0, 1.0};
    std::vector<double> X1 = {0.0, 0.5, 1.0};
    std::vector<double> Y0 = {0.5};                     // ny0 = 1
    std::vector<double> Y1 = {0.1, 0.9};
    std::vector<double> phi = {0.0, 0.2, -0.1, 0.3, 0.0, 0.4};  // nx0*nx1 = 6
    Grid2D grid{X0.size(), X1.size(), Y0.size(), Y1.size()};

    std::size_t nout = Y0.size() * Y1.size();
    std::vector<double> cpu(nout), sep(nout);
    quadraticCTransformCPU2D      (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), cpu.data(), grid);
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), sep.data(), grid);
    EXPECT_LT(maxAbsErr(cpu, sep), 1e-12);
}

TEST(Separable2D, LargeNonSquareMultiBlock) {
    // Every axis > 16 and all distinct -> multiple blocks in both passes, non-square.
    const std::size_t nx0 = 20, nx1 = 24, ny0 = 18, ny1 = 40;
    std::vector<double> X0(nx0), X1(nx1), Y0(ny0), Y1(ny1), phi(nx0 * nx1);
    for (std::size_t i = 0; i < nx0; ++i) X0[i] = -1.0 + 2.0 * i / (nx0 - 1);
    for (std::size_t i = 0; i < nx1; ++i) X1[i] = -1.0 + 2.0 * i / (nx1 - 1);
    for (std::size_t j = 0; j < ny0; ++j) Y0[j] = -1.0 + 2.0 * j / (ny0 - 1);
    for (std::size_t j = 0; j < ny1; ++j) Y1[j] = -1.0 + 2.0 * j / (ny1 - 1);
    for (std::size_t k = 0; k < nx0 * nx1; ++k) phi[k] = 0.1 * std::sin(0.7 * static_cast<double>(k));
    Grid2D grid{nx0, nx1, ny0, ny1};

    std::size_t nout = ny0 * ny1;
    std::vector<double> cpu(nout), sep(nout);
    quadraticCTransformCPU2D      (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), cpu.data(), grid);
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), sep.data(), grid);
    EXPECT_LT(maxAbsErr(cpu, sep), 1e-12);
}
