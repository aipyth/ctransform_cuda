#include <gtest/gtest.h>

#include "ctransform.hpp"

TEST(CTransform1D, TrivialZero) {
    double X[] = {0.0},
        Y[] = {0.0},
        phi[] = {0.0},
        out[1];
    Grid1D grid(1, 1);
    quadraticCTransform1D(X, Y, phi, out, grid);
    EXPECT_NEAR(out[0], 0.0, 1e-12);
}

TEST(CTransform1D, SinglePair) {
    double X[] = {0.0, 1.0}, Y[] = {0.5}, phi[] = {0.0, 0.0}, out[1];
    quadraticCTransform1D(X, Y, phi, out, Grid1D(2, 1));
    EXPECT_NEAR(out[0], 0.125, 1e-12);  // min(0.5*0.25, 0.5*0.25) = 0.125
}
