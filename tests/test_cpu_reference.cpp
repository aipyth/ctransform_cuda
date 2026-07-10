#include <gtest/gtest.h>
#include "ctransform.hpp"

TEST(CPUReference1D, TrivialZero) {
  double X[] = { 0.0 }, Y[] = { 0.0 }, phi[] = {0.0}, out[1];
  quadraticCTransformCPU(X, Y, phi, out, Grid1D(1, 1));
  EXPECT_NEAR(out[0], 0.0, 1e-12);
}

TEST(CPUReference1D, SinglePair) {
  // phi^c (0.5) = min(0.5 * (0 - 0.5)^2 - 0, 0.5 * (1 - 0.5)^2 - 0) = 0.125
  double X[] = {0.0, 1.0}, Y[] = {0.5}, phi[] = {0.0, 0.0}, out[1];
  quadraticCTransformCPU(X, Y, phi, out, Grid1D(2, 1));
  EXPECT_NEAR(out[0], 0.125, 1e-12);
}

TEST(CPUReference1D, ConstantPhiShift) {
  // (phi + k)^c = phi^c - k
  double X[] = {0.0, 1.0}, Y[] = {0.0};
  Grid1D grid(2, 1);
  double out_base[1], out_shifted[1];

  double phi0[] = {0.0, 0.0};
  quadraticCTransformCPU(X, Y, phi0, out_base, grid);

  double k = 3.7;
  double phik[] = {k, k};
  quadraticCTransformCPU(X, Y, phik, out_shifted, grid);

  EXPECT_NEAR(out_shifted[0], out_base[0] - k, 1e-12);
}

TEST(CPUReference1D, SymmetricGrid) {
  // X = {-1, 0, 1}, Y = {-0.5, 0.5}, phi = 0
  // phi^c(-0.5) = min(0.5*0.25, 0.5*0.25, 0.5*2.25) = 0.125
  // phi^c( 0.5) = min(0.5*2.25, 0.5*0.25, 0.5*0.25) = 0.125
  double X[] = {-1.0, 0.0, 1.0};
  double Y[] = {-0.5, 0.5};
  double phi[] = {0.0, 0.0, 0.0};
  double out[2];
  quadraticCTransformCPU(X, Y, phi, out, Grid1D(3, 2));
  EXPECT_NEAR(out[0], 0.125, 1e-12);
  EXPECT_NEAR(out[1], 0.125, 1e-12);
}

TEST(CPUReference1D, PhiCancellation) {
  // X = {0, 2}, Y = {1}, phi = {0.5, 0.5}
  // phi^c(1) = min(0.5*1^2 - 0.5, 0.5*1^2 - 0.5) = 0.0
  double X[] = {0.0, 2.0}, Y[] = {1.0}, phi[] = {0.5, 0.5}, out[1];
  quadraticCTransformCPU(X, Y, phi, out, Grid1D(2, 1));
  EXPECT_NEAR(out[0], 0.0, 1e-12);
}

// ---- 2D CPU reference tests ----

TEST(CPUReference2D, TrivialZero) {
    double X0[] = {0.0}, X1[] = {0.0};
    double Y0[] = {0.0}, Y1[] = {0.0};
    double phi[] = {0.0}, out[1];
    quadraticCTransformCPU(X0, X1, Y0, Y1, phi, out, Grid2D{1,1,1,1});
    EXPECT_NEAR(out[0], 0.0, 1e-12);
}

TEST(CPUReference2D, SingleSource) {
    // X = {(0,0)}, Y = {(1,0)}, phi = {0}
    // phi^c(1,0) = 0.5*(1^2 + 0^2) = 0.5
    double X0[] = {0.0}, X1[] = {0.0};
    double Y0[] = {1.0}, Y1[] = {0.0};
    double phi[] = {0.0}, out[1];
    quadraticCTransformCPU(X0, X1, Y0, Y1, phi, out, Grid2D{1,1,1,1});
    EXPECT_NEAR(out[0], 0.5, 1e-12);
}

TEST(CPUReference2D, ConstantPhiShift) {
    // (phi + k)^c = phi^c - k
    double X0[] = {0.0, 1.0}, X1[] = {0.0, 1.0};
    double Y0[] = {0.5}, Y1[] = {0.5};
    Grid2D grid{2, 2, 1, 1};
    double out_base[1], out_shifted[1];

    double phi0[] = {0.0, 0.0, 0.0, 0.0};
    quadraticCTransformCPU(X0, X1, Y0, Y1, phi0, out_base, grid);

    double k = 2.5;
    double phik[] = {k, k, k, k};
    quadraticCTransformCPU(X0, X1, Y0, Y1, phik, out_shifted, grid);

    EXPECT_NEAR(out_shifted[0], out_base[0] - k, 1e-12);
}

TEST(CPUReference2D, Separability) {
    // With phi=0, the 2D c-transform splits along each axis:
    // phi^c(y0, y1) = min_{ix0} 0.5*(X0[ix0]-y0)^2 + min_{ix1} 0.5*(X1[ix1]-y1)^2
    // X0={-0.5,0.5}, X1={0,1}, Y0={0}, Y1={0.5}
    // dim-0: min(0.5*0.25, 0.5*0.25) = 0.125
    // dim-1: min(0.5*0.25, 0.5*0.25) = 0.125
    // expected: 0.25
    double X0[] = {-0.5, 0.5}, X1[] = {0.0, 1.0};
    double Y0[] = {0.0},       Y1[] = {0.5};
    double phi[] = {0.0, 0.0, 0.0, 0.0}, out[1];
    quadraticCTransformCPU(X0, X1, Y0, Y1, phi, out, Grid2D{2,2,1,1});
    EXPECT_NEAR(out[0], 0.25, 1e-12);
}

TEST(CPUReference2D, KnownNonZeroPhi) {
    // X0={0,1}, X1={0,1}, Y0={0.5}, Y1={0.5}, all distances = 0.5 in each dim
    // phi^c(0.5,0.5) = min over 4 source pts of (0.25 - phi[ix0,ix1])
    //                = 0.25 - max(phi)
    // phi = {0.1, 0.3, 0.0, 0.2} -> max = 0.3 -> expected = -0.05
    double X0[] = {0.0, 1.0}, X1[] = {0.0, 1.0};
    double Y0[] = {0.5},      Y1[] = {0.5};
    double phi[] = {0.1, 0.3, 0.0, 0.2}, out[1];
    quadraticCTransformCPU(X0, X1, Y0, Y1, phi, out, Grid2D{2,2,1,1});
    EXPECT_NEAR(out[0], -0.05, 1e-12);
}
