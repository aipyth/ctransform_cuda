#include <gtest/gtest.h>
#include <vector>
#include <random>
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

void fillUniform(std::vector<double>& v, std::mt19937_64& rng, double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    for (double& x : v) x = dist(rng);
}

}  // namespace

// Tier 3: 1D randomized — GPU vs CPU reference, 10 seeds x size grid
TEST(Randomized1D, GpuMatchesCpu) {
    const std::size_t sizes[] = {10, 100, 500};
    for (unsigned seed = 0; seed < 10; ++seed) {
        std::mt19937_64 rng(seed);
        for (std::size_t N : sizes) {
            for (std::size_t M : sizes) {
                std::vector<double> X(N), Y(M), phi(N), cpu(M), gpu(M);
                fillUniform(X, rng, 0.0, 1.0);
                fillUniform(Y, rng, 0.0, 1.0);
                fillUniform(phi, rng, -1.0, 1.0);
                Grid1D grid{N, M};

                quadraticCTransformCPU1D(X.data(), Y.data(), phi.data(), cpu.data(), grid);
                quadraticCTransform1D   (X.data(), Y.data(), phi.data(), gpu.data(), grid);

                ASSERT_TRUE(allFinite(gpu)) << "seed=" << seed << " N=" << N << " M=" << M;
                EXPECT_LT(maxAbsErr(cpu, gpu), 1e-12) << "seed=" << seed << " N=" << N << " M=" << M;
            }
        }
    }
}

// Tier 3: 2D randomized — naive AND separable vs CPU reference
TEST(Randomized2D, GpuMatchesCpu) {
    const std::size_t sizes[] = {4, 16, 32};
    for (unsigned seed = 0; seed < 10; ++seed) {
        std::mt19937_64 rng(seed);
        for (std::size_t nx0 : sizes)
        for (std::size_t nx1 : sizes)
        for (std::size_t ny0 : sizes)
        for (std::size_t ny1 : sizes) {
            std::vector<double> X0(nx0), X1(nx1), Y0(ny0), Y1(ny1), phi(nx0 * nx1);
            fillUniform(X0, rng, 0.0, 1.0);
            fillUniform(X1, rng, 0.0, 1.0);
            fillUniform(Y0, rng, 0.0, 1.0);
            fillUniform(Y1, rng, 0.0, 1.0);
            fillUniform(phi, rng, -1.0, 1.0);
            Grid2D grid{nx0, nx1, ny0, ny1};

            std::size_t nout = ny0 * ny1;
            std::vector<double> cpu(nout), naive(nout), sep(nout);
            quadraticCTransformCPU2D      (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), cpu.data(),   grid);
            quadraticCTransform2D         (X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), naive.data(), grid);
            quadraticCTransform2DSeparable(X0.data(), X1.data(), Y0.data(), Y1.data(), phi.data(), sep.data(),   grid);

            const char* ctx = "";  // keep messages short; combo is deterministic per seed
            ASSERT_TRUE(allFinite(naive)) << "seed=" << seed;
            ASSERT_TRUE(allFinite(sep))   << "seed=" << seed;
            EXPECT_LT(maxAbsErr(cpu, naive), 1e-12) << "seed=" << seed;
            EXPECT_LT(maxAbsErr(cpu, sep),   1e-12) << "seed=" << seed;
            (void)ctx;
        }
    }
}
