#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "ctransform.hpp"

namespace t3d {
  
template <typename T>
using CTransform3DFn = void (*)(const T*, const T*, const T*,
                                const T*, const T*, const T*,
                                const T*, T*, Grid3D);

template <typename T>
struct Axes3 {
  std::vector<T> a0, a1, a2;
  std::size_t size() const { return a0.size() * a1.size() * a2.size(); }
};

inline std::size_t flat3(std::size_t i0, std::size_t i1, std::size_t i2,
                         std::size_t n1, std::size_t n2) {
  return (i0 * n1 + i1) * n2 + i2;
}

template <typename T>
std::vector<T> run(CTransform3DFn<T> f, const Axes3<T>& X, const Axes3<T>& Y,
                   const std::vector<T>& phi) {
  std::vector<T> out(Y.size());
  const Grid3D grid{X.a0.size(), X.a1.size(), X.a2.size(),
                    Y.a0.size(), Y.a1.size(), Y.a2.size()};
  f(X.a0.data(), X.a1.data(), X.a2.data(),
    Y.a0.data(), Y.a1.data(), Y.a2.data(),
    phi.data(), out.data(), grid);
  return out;
}

template <typename T>
T maxAbsErr(const std::vector<T>& a, const std::vector<T>& b) {
  T err = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    err = std::max(err, std::abs(a[i] - b[i]));
  }
  return err;
}

inline std::vector<double> uniform(std::size_t n,
    std::mt19937_64& rng, double lo, double hi) {
  std::uniform_real_distribution<double> dist(lo, hi);
  std::vector<double> v(n);
  for (double& x : v) x = dist(rng);
  return v;
}

inline Axes3<double> randomAxes(std::size_t n0, std::size_t n1, std::size_t n2,
                                std::mt19937_64& rng) {
  return {uniform(n0, rng, 0.0, 1.0), uniform(n1, rng, 0.0, 1.0),
    uniform(n2, rng, 0.0, 1.0)};
}

} // namespace t3d
