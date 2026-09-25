#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

#include "ctransform.hpp"
#include "helpers_3d.hpp"

using t3d::Axes3;
using t3d::CTransform3DFn;
using t3d::flat3;
using t3d::maxAbsErr;
using t3d::randomAxes;
using t3d::uniform;
using Vec = std::vector<double>;

namespace {

// checkSpike: a test whose every output value is known exactly, used to catch indexing bugs.
//
// phi is 0 everywhere except at one source point x* = (X0[s0], X1[s1], X2[s2]), where it
// is 100. The c-transform takes the minimum of 0.5*|x - y|^2 - phi(x) over all source
// points x:
//   - at x*, the value is 0.5*|x* - y|^2 - 100, which is negative (the largest distance
//     term on this grid is about 15.6, far below 100);
//   - at every other x, the value is 0.5*|x - y|^2 - 0, which is >= 0.
// So x* wins for every target y, and every output entry must equal 0.5*|x* - y|^2 - 100.
//
// Why this catches indexing bugs: if the implementation reads phi or writes out with a
// wrong stride or with two axes swapped, it finds the spike at a different point, or
// writes the answers to the wrong places, and the values no longer match.
//
// Why these grids:
//   - the axes have different sizes (source 2, 3, 4; target 5, 6, 7), so swapping two
//     axes always changes the result. On a cube grid many swaps would go unnoticed;
//   - all coordinates are multiples of 0.5, so every value is computed exactly in both
//     float and double, which is why the test can use exact equality (EXPECT_EQ).
template <typename T>
std::vector<T> checkSpike(CTransform3DFn<T> f, std::size_t s0, std::size_t s1, std::size_t s2) {
  const Axes3<T> X{{0.0, 1.0}, {0.0, 1.0, 2.0}, {0.0, 1.0, 2.0, 3.0}};
  const Axes3<T> Y{{-1.0, -0.5, 0.5, 1.5, 2.5         },
                   {-1.0, 0.0, 0.5, 1.0, 2.0, 3.0     },
                   {-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0}};
  std::vector<T> phi(X.size(), T(0));
  phi[flat3(s0, s1, s2, X.a1.size(), X.a2.size())] = T(100);

  const std::vector<T> out = t3d::run(f, X, Y, phi);

  for (std::size_t j0 = 0; j0 < Y.a0.size(); ++j0)
  for (std::size_t j1 = 0; j1 < Y.a1.size(); ++j1)
  for (std::size_t j2 = 0; j2 < Y.a2.size(); ++j2) {
      const T d0 = X.a0[s0] - Y.a0[j0];
      const T d1 = X.a1[s1] - Y.a1[j1];
      const T d2 = X.a2[s2] - Y.a2[j2];
      const T expected = T(0.5) * (d0 * d0 + d1 * d1 + d2 * d2) - T(100);
      EXPECT_EQ(out[flat3(j0, j1, j2, Y.a1.size(), Y.a2.size())], expected)
          << "spike=(" << s0 << "," << s1 << "," << s2 << ")"
          << " out(" << j0 << "," << j1 << "," << j2 << ")";
  }
  return out;
}
  
} // namespace

class Quadratic3D : public ::testing::TestWithParam<CTransform3DFn<double>> {
protected:
    Vec run(const Axes3<double>& X, const Axes3<double>& Y, const Vec& phi) const {
        return t3d::run(GetParam(), X, Y, phi);
    }
};

// ---- Closed-form values ----

TEST_P(Quadratic3D, TrivialZero) {
    const Vec out = run({{0.0}, {0.0}, {0.0}}, {{0.0}, {0.0}, {0.0}}, {0.0});
    EXPECT_NEAR(out[0], 0.0, 1e-12);
}

TEST_P(Quadratic3D, SingleSource) {
    // x = (1, 2, -2), y = (0, 0, 0), phi = 0.5 -> 0.5*(1 + 4 + 4) - 0.5 = 4
    const Vec out = run({{1.0}, {2.0}, {-2.0}}, {{0.0}, {0.0}, {0.0}}, {0.5});
    EXPECT_NEAR(out[0], 4.0, 1e-12);
}

TEST_P(Quadratic3D, ZeroPhiSeparability) {
    // phi = 0 -> per-axis minima add: 0.125 + 0.125 + 0.5 = 0.75
    const Axes3<double> X{{-0.5, 0.5}, {0.0, 1.0}, {1.0, 3.0}};
    const Axes3<double> Y{{0.0}, {0.5}, {2.0}};
    const Vec out = run(X, Y, Vec(X.size(), 0.0));
    EXPECT_NEAR(out[0], 0.75, 1e-12);
}

TEST_P(Quadratic3D, CubeNonZeroPhi) {
    // Every source point is at squared distance 3 * 0.25 from y = (0.5, 0.5, 0.5),
    // so phi^c = 0.375 - max(phi) = 0.375 - 0.35 = 0.025
    const Axes3<double> X{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
    const Axes3<double> Y{{0.5}, {0.5}, {0.5}};
    const Vec phi = {0.1, 0.3, 0.0, 0.2, -0.1, 0.25, 0.35, 0.05};
    const Vec out = run(X, Y, phi);
    EXPECT_NEAR(out[0], 0.025, 1e-12);
}

TEST_P(Quadratic3D, SpikeLayoutProbe) {
    const std::size_t mid = flat3(1, 2, 3, 6, 7);   // output point y = (-0.5, 0.5, 1)

    const Vec a = checkSpike<double>(GetParam(), 1, 2, 0);   // phi flat index 20
    EXPECT_EQ(a.front(), -93.0);
    EXPECT_EQ(a.back(), -90.375);
    EXPECT_EQ(a[mid], -97.25);

    const Vec b = checkSpike<double>(GetParam(), 0, 0, 3);   // phi flat index 3
    EXPECT_EQ(b.front(), -91.0);
    EXPECT_EQ(b.back(), -91.875);
    EXPECT_EQ(b[mid], -97.75);
}

// ---- Oracles from the trusted 1D/2D references ----

TEST_P(Quadratic3D, DimensionReduction) {
    // A singleton axis X = {a}, Y = {b} adds the constant 0.5*(a - b)^2, so the 3D transform
    // equals the 2D reference on the other two axes plus that constant. Singleton axes do not
    // change row-major memory, so one phi buffer of shape (4, 5) serves all three placements.
    std::mt19937_64 rng(7);
    const Vec P = uniform(4, rng, 0.0, 1.0), Q = uniform(5, rng, 0.0, 1.0);   // source axes
    const Vec R = uniform(3, rng, 0.0, 1.0), S = uniform(6, rng, 0.0, 1.0);   // target axes
    const Vec phi = uniform(P.size() * Q.size(), rng, -1.0, 1.0);
    const double a = 0.25, b = 1.0;                                            // adds 0.28125

    Vec ref(R.size() * S.size());
    quadraticCTransformCPU2D(P.data(), Q.data(), R.data(), S.data(), phi.data(), ref.data(),
                             Grid2D{P.size(), Q.size(), R.size(), S.size()});
    for (double& v : ref) v += 0.5 * (a - b) * (a - b);

    for (std::size_t pos = 0; pos < 3; ++pos) {
        std::vector<Vec> xs{P, Q}, ys{R, S};
        xs.insert(xs.begin() + pos, Vec{a});
        ys.insert(ys.begin() + pos, Vec{b});
        const Vec out = run({xs[0], xs[1], xs[2]}, {ys[0], ys[1], ys[2]}, phi);
        EXPECT_LT(maxAbsErr(out, ref), 1e-12) << "singleton axis " << pos;
    }
}

TEST_P(Quadratic3D, AdditivePhi) {
    // phi(x) = f0(x0) + f1(x1) + f2(x2)  =>  phi^c(y) = f0^c(y0) + f1^c(y1) + f2^c(y2)
    std::mt19937_64 rng(11);
    const Axes3<double> X = randomAxes(4, 5, 6, rng);
    const Axes3<double> Y = randomAxes(3, 7, 2, rng);
    const Vec f0 = uniform(X.a0.size(), rng, -1.0, 1.0);
    const Vec f1 = uniform(X.a1.size(), rng, -1.0, 1.0);
    const Vec f2 = uniform(X.a2.size(), rng, -1.0, 1.0);

    Vec phi(X.size());
    for (std::size_t i0 = 0; i0 < X.a0.size(); ++i0)
        for (std::size_t i1 = 0; i1 < X.a1.size(); ++i1)
            for (std::size_t i2 = 0; i2 < X.a2.size(); ++i2)
                phi[flat3(i0, i1, i2, X.a1.size(), X.a2.size())] = f0[i0] + f1[i1] + f2[i2];

    Vec g0(Y.a0.size()), g1(Y.a1.size()), g2(Y.a2.size());
    quadraticCTransformCPU1D(X.a0.data(), Y.a0.data(), f0.data(), g0.data(), Grid1D{X.a0.size(), Y.a0.size()});
    quadraticCTransformCPU1D(X.a1.data(), Y.a1.data(), f1.data(), g1.data(), Grid1D{X.a1.size(), Y.a1.size()});
    quadraticCTransformCPU1D(X.a2.data(), Y.a2.data(), f2.data(), g2.data(), Grid1D{X.a2.size(), Y.a2.size()});

    Vec expected(Y.size());
    for (std::size_t j0 = 0; j0 < Y.a0.size(); ++j0)
        for (std::size_t j1 = 0; j1 < Y.a1.size(); ++j1)
            for (std::size_t j2 = 0; j2 < Y.a2.size(); ++j2)
                expected[flat3(j0, j1, j2, Y.a1.size(), Y.a2.size())] = g0[j0] + g1[j1] + g2[j2];

    EXPECT_LT(maxAbsErr(run(X, Y, phi), expected), 1e-12);
}

// ---- Invariants (no reference needed) ----

TEST_P(Quadratic3D, ConstantPhiShift) {
    // (phi + k)^c = phi^c - k
    std::mt19937_64 rng(3);
    const Axes3<double> X = randomAxes(3, 4, 5, rng);
    const Axes3<double> Y = randomAxes(4, 2, 3, rng);
    const Vec phi = uniform(X.size(), rng, -1.0, 1.0);
    const double k = 3.7;

    Vec shifted = phi;
    for (double& v : shifted) v += k;

    Vec expected = run(X, Y, phi);
    for (double& v : expected) v -= k;

    EXPECT_LT(maxAbsErr(run(X, Y, shifted), expected), 1e-12);
}

TEST_P(Quadratic3D, Monotone) {
    // psi >= phi pointwise  =>  psi^c <= phi^c pointwise
    std::mt19937_64 rng(5);
    const Axes3<double> X = randomAxes(3, 4, 5, rng);
    const Axes3<double> Y = randomAxes(4, 2, 3, rng);
    const Vec phi = uniform(X.size(), rng, -1.0, 1.0);
    const Vec bump = uniform(X.size(), rng, 0.0, 0.5);

    Vec psi(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) psi[i] = phi[i] + bump[i];

    const Vec phiC = run(X, Y, phi), psiC = run(X, Y, psi);
    for (std::size_t j = 0; j < Y.size(); ++j)
        EXPECT_LE(psiC[j], phiC[j]) << "target " << j;
}

TEST_P(Quadratic3D, CInequalityAndAttainment) {
    // phi^c(y) + phi(x) <= c(x, y) for every pair, with equality for at least one x per y.
    // Pins down both the min convention and the sign of phi.
    std::mt19937_64 rng(13);
    const Axes3<double> X = randomAxes(3, 4, 5, rng);
    const Axes3<double> Y = randomAxes(4, 3, 2, rng);
    const Vec phi = uniform(X.size(), rng, -1.0, 1.0);
    const Vec psi = run(X, Y, phi);

    for (std::size_t j0 = 0; j0 < Y.a0.size(); ++j0)
        for (std::size_t j1 = 0; j1 < Y.a1.size(); ++j1)
            for (std::size_t j2 = 0; j2 < Y.a2.size(); ++j2) {
                const double psiY = psi[flat3(j0, j1, j2, Y.a1.size(), Y.a2.size())];
                double minGap = INFINITY;
                for (std::size_t i0 = 0; i0 < X.a0.size(); ++i0)
                    for (std::size_t i1 = 0; i1 < X.a1.size(); ++i1)
                        for (std::size_t i2 = 0; i2 < X.a2.size(); ++i2) {
                            const double d0 = X.a0[i0] - Y.a0[j0];
                            const double d1 = X.a1[i1] - Y.a1[j1];
                            const double d2 = X.a2[i2] - Y.a2[j2];
                            const double c = 0.5 * (d0 * d0 + d1 * d1 + d2 * d2);
                            const double gap = c - phi[flat3(i0, i1, i2, X.a1.size(), X.a2.size())] - psiY;
                            EXPECT_GE(gap, -1e-12);
                            minGap = std::min(minGap, gap);
                        }
                EXPECT_NEAR(minGap, 0.0, 1e-12) << "target (" << j0 << "," << j1 << "," << j2 << ")";
            }
}

TEST_P(Quadratic3D, TripleTransformIdempotent) {
    // Tbar = the same function with X and Y swapped (c is symmetric).
    // phi^cc = Tbar(T phi) >= phi, and T(Tbar(T phi)) = T phi.
    std::mt19937_64 rng(17);
    const Axes3<double> X = randomAxes(5, 6, 7, rng);
    const Axes3<double> Y = randomAxes(4, 8, 3, rng);
    const Vec phi = uniform(X.size(), rng, -1.0, 1.0);

    const Vec phiC   = run(X, Y, phi);
    const Vec phiCC  = run(Y, X, phiC);
    const Vec phiCCC = run(X, Y, phiCC);

    for (std::size_t i = 0; i < X.size(); ++i)
        EXPECT_GE(phiCC[i], phi[i] - 1e-12) << "source " << i;
    EXPECT_LT(maxAbsErr(phiCCC, phiC), 1e-12);
}

TEST_P(Quadratic3D, AxisPermutation) {
    // Cycling the axes (0, 1, 2) -> (1, 2, 0) of X, Y and phi cycles the output the same way:
    //   phiP[i1, i2, i0] = phi[i0, i1, i2]   =>   outP[j1, j2, j0] = out[j0, j1, j2]
    std::mt19937_64 rng(19);
    const Axes3<double> X = randomAxes(3, 4, 5, rng);
    const Axes3<double> Y = randomAxes(6, 2, 4, rng);
    const Vec phi = uniform(X.size(), rng, -1.0, 1.0);

    const Axes3<double> XP{X.a1, X.a2, X.a0};
    const Axes3<double> YP{Y.a1, Y.a2, Y.a0};
    Vec phiP(X.size());
    for (std::size_t i0 = 0; i0 < X.a0.size(); ++i0)
        for (std::size_t i1 = 0; i1 < X.a1.size(); ++i1)
            for (std::size_t i2 = 0; i2 < X.a2.size(); ++i2)
                phiP[flat3(i1, i2, i0, X.a2.size(), X.a0.size())] =
                    phi[flat3(i0, i1, i2, X.a1.size(), X.a2.size())];

    const Vec out = run(X, Y, phi), outP = run(XP, YP, phiP);
    for (std::size_t j0 = 0; j0 < Y.a0.size(); ++j0)
        for (std::size_t j1 = 0; j1 < Y.a1.size(); ++j1)
            for (std::size_t j2 = 0; j2 < Y.a2.size(); ++j2)
                EXPECT_NEAR(outP[flat3(j1, j2, j0, Y.a2.size(), Y.a0.size())],
                            out[flat3(j0, j1, j2, Y.a1.size(), Y.a2.size())], 1e-12)
                    << "target (" << j0 << "," << j1 << "," << j2 << ")";
}

// ---- Empty axes (docs/engineering/api.md, "Empty-axis semantics (3D)") ----

TEST_P(Quadratic3D, EmptySourceAxis) {
    // nx1 = 0: min over an empty set -> every output entry is +inf; phi is never read.
    const Axes3<double> X{{0.0, 1.0}, {}, {0.5}};
    const Axes3<double> Y{{0.0}, {0.25, 0.75}, {1.0}};
    const Vec out = run(X, Y, Vec{});
    ASSERT_EQ(out.size(), 2u);
    for (double v : out) EXPECT_TRUE(std::isinf(v) && v > 0) << v;
}

TEST_P(Quadratic3D, EmptyTargetAxis) {
    // ny1 = 0: no output points -> no throw, nothing written.
    const Vec X0{0.0, 1.0}, X1{0.5}, X2{0.5}, Y0{0.0}, Y1{}, Y2{1.0};
    const Vec phi(2, 0.0);
    double sentinel[1] = {42.0};
    EXPECT_NO_THROW(GetParam()(X0.data(), X1.data(), X2.data(),
                               Y0.data(), Y1.data(), Y2.data(),
                               phi.data(), sentinel, Grid3D{2, 1, 1, 1, 0, 1}));
    EXPECT_EQ(sentinel[0], 42.0);
}

// ---- Float: the dyadic spike fixture is exact in float too ----

TEST(Quadratic3DFloat, SpikeLayoutProbeCPU) {
    checkSpike<float>(&quadraticCTransformCPU3D<float>, 1, 2, 0);
    checkSpike<float>(&quadraticCTransformCPU3D<float>, 0, 0, 3);
}

// ---- Implementations under test ----

INSTANTIATE_TEST_SUITE_P(CPU, Quadratic3D,
                         ::testing::Values(&quadraticCTransformCPU3D<double>));
// Next stage, once implemented:
// INSTANTIATE_TEST_SUITE_P(Naive, Quadratic3D,
//                          ::testing::Values(&quadraticCTransform3D<double>));

// ---- Naive GPU kernel vs CPU reference ----
// The Quadratic3D suite above already runs every closed-form test on the naive kernel.
// These tests add what that suite cannot: grids large or oddly shaped enough to exercise the
// launch configuration (partial blocks, very long axes, more than 65535 blocks).

namespace {

// Runs the CPU reference and the naive GPU kernel on the same inputs and returns the largest
// absolute difference between the two outputs.
double naiveVsCpu(const Axes3<double>& X, const Axes3<double>& Y, const Vec& phi) {
    const Vec cpu = t3d::run(&quadraticCTransformCPU3D<double>, X, Y, phi);
    const Vec gpu = t3d::run(&quadraticCTransform3D<double>, X, Y, phi);
    return maxAbsErr(cpu, gpu);
}

}  // namespace

TEST(NaiveVsCPU3D, OddSizesMultiBlock) {
    // 11 * 13 * 37 = 5291 outputs = 20 full blocks of 256 threads plus one partial block of 171.
    // No axis size is a multiple of 8 or 32, so any rounding mistake in the block count or a
    // missing "t < N" bounds check shows up here.
    std::mt19937_64 rng(23);
    const Axes3<double> X = randomAxes(5, 7, 9, rng);
    const Axes3<double> Y = randomAxes(11, 13, 37, rng);
    const Vec phi = uniform(X.size(), rng, -1.0, 1.0);
    EXPECT_LT(naiveVsCpu(X, Y, phi), 1e-12);
}

TEST(NaiveVsCPU3D, LongAxisInEachPosition) {
    // One target axis with 70000 points, the other two with 1. A launch that puts an output
    // axis on gridDim.y or gridDim.z (maximum 65535 blocks each) fails to launch for at least
    // one of the three positions; a flattened launch over gridDim.x handles all three.
    std::mt19937_64 rng(29);
    const Axes3<double> X = randomAxes(2, 3, 2, rng);
    const Vec phi = uniform(X.size(), rng, -1.0, 1.0);
    const Vec longAxis = uniform(70000, rng, 0.0, 1.0);
    const Vec one = {0.5};

    const Axes3<double> targets[] = {
        {longAxis, one, one},   // ny = (70000, 1, 1)
        {one, longAxis, one},   // ny = (1, 70000, 1)
        {one, one, longAxis},   // ny = (1, 1, 70000)
    };
    for (std::size_t pos = 0; pos < 3; ++pos)
        EXPECT_LT(naiveVsCpu(X, targets[pos], phi), 1e-12) << "long target axis " << pos;
}

TEST(NaiveVsCPU3D, MoreThan65535Blocks) {
    // 260^3 = 17,576,000 outputs need 68,657 blocks of 256 threads, more than the 65535 that
    // gridDim.y or gridDim.z allow. A single source point keeps the CPU side cheap.
    std::mt19937_64 rng(31);
    const Axes3<double> X = randomAxes(1, 1, 1, rng);
    const Axes3<double> Y = randomAxes(260, 260, 260, rng);
    const Vec phi = {0.3};
    EXPECT_LT(naiveVsCpu(X, Y, phi), 1e-12);
}

// ---- Float ----

TEST(Quadratic3DFloat, SpikeLayoutProbeNaive) {
    // Exact equality still holds on the GPU: the spike values are exact in float, so it makes
    // no difference that nvcc may fuse a multiply and an add into one FMA instruction.
    checkSpike<float>(&quadraticCTransform3D<float>, 1, 2, 0);
    checkSpike<float>(&quadraticCTransform3D<float>, 0, 0, 3);
}

TEST(Quadratic3DFloat, NaiveRandomVsDoubleCpu) {
    // Float kernel on float-rounded inputs vs the double CPU reference on the same (rounded)
    // values. Outputs here are at most a few units in size, and float carries about 7 digits,
    // so 1e-5 leaves a wide margin while still catching a real bug.
    std::mt19937_64 rng(37);
    const Axes3<double> Xd = randomAxes(6, 5, 7, rng);
    const Axes3<double> Yd = randomAxes(9, 4, 8, rng);
    const Vec phid = uniform(Xd.size(), rng, -1.0, 1.0);

    const auto toFloat = [](const Vec& v) { return std::vector<float>(v.begin(), v.end()); };
    const auto toDouble = [](const std::vector<float>& v) { return Vec(v.begin(), v.end()); };
    const Axes3<float> Xf{toFloat(Xd.a0), toFloat(Xd.a1), toFloat(Xd.a2)};
    const Axes3<float> Yf{toFloat(Yd.a0), toFloat(Yd.a1), toFloat(Yd.a2)};
    const std::vector<float> phif = toFloat(phid);

    // Double reference computed from the float-rounded inputs, so only the kernel's own
    // float arithmetic is being measured.
    const Axes3<double> Xr{toDouble(Xf.a0), toDouble(Xf.a1), toDouble(Xf.a2)};
    const Axes3<double> Yr{toDouble(Yf.a0), toDouble(Yf.a1), toDouble(Yf.a2)};
    const Vec ref = t3d::run(&quadraticCTransformCPU3D<double>, Xr, Yr, toDouble(phif));

    const Vec gpu = toDouble(t3d::run(&quadraticCTransform3D<float>, Xf, Yf, phif));
    EXPECT_LT(maxAbsErr(gpu, ref), 1e-5);
}

INSTANTIATE_TEST_SUITE_P(Naive, Quadratic3D,
                         ::testing::Values(&quadraticCTransform3D<double>));
