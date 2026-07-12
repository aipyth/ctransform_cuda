# Test strategy

## Overview

Tests are organized in four tiers: tiny deterministic, CPU reference comparison, randomized, and stress/benchmark. Tests live in `tests/` as GoogleTest suites built as CMake targets linked against the c-transform kernels.

| Tier | Status | File(s) |
|---|---|---|
| 1. Tiny deterministic | Implemented | `test_ctransform_1d.cpp`, `test_cpu_reference.cpp` |
| 2. CPU reference comparison | Implemented | `test_cuda_vs_cpu.cpp` |
| 3. Randomized | Not yet implemented | — |
| 4. Stress/large-scale | Not yet implemented | — |

`test_perf.cpp` builds as a separate binary (`ctransform_perf`, not a CTest target); see [Performance benchmarks](#performance-benchmarks) below.

---

## Tier 1: Tiny deterministic tests (implemented)

These tests have closed-form answers that can be verified by hand. They catch regression in the kernel formula itself. Implemented in `test_ctransform_1d.cpp` and `test_cpu_reference.cpp`.

### 1D tests

**Trivial zero**: X={0}, Y={0}, Phi={0}  
→ psi = min{ ½(0−0)² − 0 } = 0

**Single pair, zero phi**: X={0, 1}, Y={0.5}, Phi={0, 0}  
→ psi = min{ ½·0.25 − 0, ½·0.25 − 0 } = **0.125**

**Non-zero phi**: X={0, 1}, Y={0.5}, Phi={0.1, 0.0}  
→ psi = min{ ½·0.25 − 0.1, ½·0.25 − 0.0 } = min{0.025, 0.125} = **0.025**

**Existing source test** (from `ctransform_naive.cu`):  
X={−0.5, 0.0, 0.5}, Y={−0.2, 0.0, 0.2, 0.5}, Phi={0.2, 0.0, 0.3}  
→ compute expected values by hand and assert exact double match

**Constant-phi shift**: X={a, b}, Phi={c, c}  
→ psi(y) = min{ ½(a−y)², ½(b−y)² } − c; the argmin switches at the midpoint (a+b)/2

### 2D tests

**Trivial zero**: Xaxis0={0}, Xaxis1={0}, Yaxis0={0}, Yaxis1={0}, Phi={0}  
→ psi = 0

**Single source, zero phi**: Xaxis0={1}, Xaxis1={2}, Yaxis0={0}, Yaxis1={0}, Phi={0}  
→ psi = ½((1−0)² + (2−0)²) = ½(1+4) = **2.5**

**2×2 source, zero phi**: Xaxis0={0,1}, Xaxis1={0,1}, Yaxis0={0.5}, Yaxis1={0.5}, Phi={0,0,0,0}  
→ psi = min{ ½(0.25+0.25), ½(0.25+0.25), ½(0.25+0.25), ½(0.25+0.25) } = **0.25**

**Existing source test** (from `ctransform2d_naive.cu`):  
Xaxis0={−0.5,0.5}, Xaxis1={0.0,1.0,2.0}, Yaxis0={−0.5,0.5}, Yaxis1={−0.5,0.5}, Phi (2×3) as in source  
→ compute expected values and assert

---

## Tier 2: CPU reference comparisons (implemented)

Implemented in `test_cuda_vs_cpu.cpp`. A pure C++ reference function (no CUDA, no parallelism) computes the c-transform by nested loop:

```cpp
// 1D reference
void ctransform_cpu_1d(const double* X, const double* Y, const double* Phi,
                       double* out, int nx, int ny) {
    for (int iy = 0; iy < ny; ++iy) {
        double result = INFINITY;
        for (int ix = 0; ix < nx; ++ix) {
            double d = X[ix] - Y[iy];
            result = std::min(result, 0.5 * d * d - Phi[ix]);
        }
        out[iy] = result;
    }
}
```

Run both the GPU kernel and the CPU reference on identical inputs. Assert:

```
max_{i} |psi_GPU[i] − psi_CPU[i]| < 1e-12   (double)
```

This tolerance is derived from double machine epsilon (~2.2e-16) times the expected magnitude of the result (~1). For inputs with ‖psi‖ ≫ 1 the tolerance should scale accordingly.

Run the CPU reference comparison for all Tier-1 inputs and all Tier-3 (randomized) inputs.

---

## Tier 3: Randomized tests (not yet implemented)

**Parameters**: seed-fixed `std::mt19937_64`; repeat for seeds {0, 1, 2, …, 9}

| Parameter | Range |
|---|---|
| X, Y coordinates | Uniform in [0, 1]^d |
| Phi values | Uniform in [−1, 1] |
| N (1D source) | {10, 100, 500} |
| M (1D target) | {10, 100, 500} |
| nx0, nx1 (2D source) | {4, 16, 32} |
| ny0, ny1 (2D target) | {4, 16, 32} |

For each (seed, N, M) combination:
1. Generate random inputs on host
2. Run CPU reference → psi_cpu
3. Run GPU kernel → psi_gpu
4. Assert `max |psi_gpu[i] − psi_cpu[i]| < 1e-12`
5. Assert no NaN, no Inf in psi_gpu

---

## Tier 4: Stress tests (not yet implemented)

**Purpose**: verify correctness at large scale and check for CUDA memory errors.

| Test | Config | Assertion |
|---|---|---|
| 1D large | N=M=5000 (25M pairs) | Same CPU/GPU within 1e-12; no CUDA error |
| 1D very large | N=M=10000 (100M pairs) | No CUDA error; GPU result finite |
| 2D medium | nx0=nx1=50, ny0=ny1=50 (6.25M pairs) | Same CPU/GPU within 1e-12 |
| 2D large | nx0=nx1=128, ny0=ny1=128 | No CUDA error; GPU result finite |

For the "no CUDA error" tests without CPU comparison, verify:
- All output values are finite (`std::isfinite`)
- No `cudaGetLastError` after kernel launch
- `cudaDeviceSynchronize` returns `cudaSuccess`

---

## Edge cases

| Case | Expected behavior |
|---|---|
| N=1 (single source point) | psi(y) = ½(x₀−y)² − φ₀ for all y |
| M=1 (single target point) | One output value; compare to CPU |
| X=Y (same coordinates as target) | Valid; diagonal terms are ½·0² − φ(x) = −φ(x) for x=y |
| Phi all zeros | psi(y) = min_x ½‖x−y‖² |
| Phi with large values (1e6) | Requires double; check no overflow in subtraction |
| Non-unit domain, e.g. X,Y ∈ [0,100] | double required; float may lose precision |

---

## Precision tolerances

| Precision | Tolerance vs CPU reference |
|---|---|
| double (T=double) | `max abs error < 1e-12` |
| float (T=float, future) | `max abs error < 1e-4` vs double reference, or `< 1e-5` vs float CPU |

The double tolerance is conservative; in practice errors are near machine epsilon (~2e-16) for unit-domain inputs. The tolerance allows for one ULP difference in the sequential scan accumulation.

---

## Performance benchmarks

Benchmarks are informational and do not block test pass/fail.

| Measurement | Method |
|---|---|
| GPU kernel time | `cudaEventRecord` before and after kernel; `cudaEventElapsedTime` |
| CPU reference time | `std::chrono::high_resolution_clock` |
| Reported metric | Wall-clock kernel time in ms; effective GFLOP/s (2·N·M ops / time) |

Target: GPU kernel faster than single-threaded CPU for N·M ≥ 10⁶.

Benchmark configurations:

| Config | N·M |
|---|---|
| 1D N=M=1000 | 10⁶ |
| 1D N=M=3162 | 10⁷ |
| 1D N=M=10000 | 10⁸ |
| 2D 32×32 source, 32×32 target | ~10⁶ |
| 2D 64×64 source, 64×64 target | ~1.7×10⁷ |
| 2D 128×128 source, 128×128 target | ~2.7×10⁸ |

Report speedup ratio (CPU time / GPU kernel time) for each. Note that transfer time is excluded from GPU time since in a production pipeline the data would already reside on device.
