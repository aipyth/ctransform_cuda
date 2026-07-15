#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "ctransform.hpp"

namespace {

template <typename T>
T maxAbsErr(const std::vector<T>& a, const std::vector<T>& b) {
    T err = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

template <typename T>
bool allFinite(const std::vector<T>& v) {
    for (T x : v) if (!std::isfinite(x)) return false;
    return true;
}

void ramp(std::vector<double>& v) {
    std::size_t n = v.size();
    for (std::size_t i = 0; i < n; ++i)
        v[i] = (n > 1) ? (-1.0 + 2.0 * i / (n - 1)) : 0.0;
}

}  // namespace

// Tier 4: 1D medium, 25M pairs — compare to CPU
TEST(Stress1D, MediumMatchesCpu) {
    const std::size_t N = 5000, M = 5000;
    std::vector<double> X(N), Y(M), phi(N, 0.0), cpu(M), gpu(M);
    ramp(X); ramp(Y);
    Grid1D grid{N, M};
    quadraticCTransformCPU1D(X.data(), Y.data(), phi.data(), cpu.data(), grid);
    quadraticCTransform1D   (X.data(), Y.data(), phi.data(), gpu.data(), grid);
    ASSERT_TRUE(allFinite(gpu));
    EXPECT_LT(maxAbsErr(cpu, gpu), 1e-12);
}

// Tier 4: 1D large, 100M pairs — GPU only, finiteness
TEST(Stress1D, LargeFinite) {
    const std::size_t N = 10000, M = 10000;
    std::vector<double> X(N), Y(M), phi(N, 0.0), gpu(M);
    ramp(X); ramp(Y);
    Grid1D grid{N, M};
    quadraticCTransform1D(X.data(), Y.data(), phi.data(), gpu.data(), grid);
    EXPECT_TRUE(allFinite(gpu));
}

// Tier 4: 2D medium, 50^4 — naive + separable vs CPU
TEST(Stress2D, MediumMatchesCpu) {
    const std::size_t nx0 = 50, nx1 = 50, ny0 = 50, ny1 = 50;
    std::vector<double> X0(nx0), X1(nx1), Y0(ny0), Y1(ny1), phi(nx0 * nx1, 0.0);
    ramp(X0); ramp(X1); ramp(Y0); ramp(Y1);
    Grid2D grid{nx0, nx1, ny0, ny1};
    std::size_t nout = ny0 * ny1;
    std::vector<double> cpu(nout), naive(nout), sep(nout);
    quadraticCTransformCPU2D      (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), cpu.data(),   grid);
    quadraticCTransform2D         (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), naive.data(), grid);
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), sep.data(),   grid);
    ASSERT_TRUE(allFinite(naive));
    ASSERT_TRUE(allFinite(sep));
    EXPECT_LT(maxAbsErr(cpu, naive), 1e-12);
    EXPECT_LT(maxAbsErr(cpu, sep),   1e-12);
}

// Tier 4: 2D large, 128^4 — GPU only, finiteness
TEST(Stress2D, LargeFinite) {
    const std::size_t nx0 = 128, nx1 = 128, ny0 = 128, ny1 = 128;
    std::vector<double> X0(nx0), X1(nx1), Y0(ny0), Y1(ny1), phi(nx0 * nx1, 0.0);
    ramp(X0); ramp(X1); ramp(Y0); ramp(Y1);
    Grid2D grid{nx0, nx1, ny0, ny1};
    std::size_t nout = ny0 * ny1;
    std::vector<double> naive(nout), sep(nout);
    quadraticCTransform2D         (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), naive.data(), grid);
    quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), sep.data(),   grid);
    EXPECT_TRUE(allFinite(naive));
    EXPECT_TRUE(allFinite(sep));
}
