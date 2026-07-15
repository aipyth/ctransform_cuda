# Test strategy

## Overview

Tests are organized in four tiers: tiny deterministic, CPU reference comparison, randomized, and stress/benchmark. Tests live in `tests/` as GoogleTest suites built as CMake targets linked against the c-transform kernels.

| Tier | Status | File(s) |
|---|---|---|
| 1. Tiny deterministic | Implemented | `test_ctransform_1d.cpp`, `test_cpu_reference.cpp` |
| 2. CPU reference comparison | Implemented | `test_cuda_vs_cpu.cpp` |
| 3. Randomized | Implemented | `test_randomized.cpp` |
| 4. Stress/large-scale | Implemented | `test_stress.cpp` |

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

### 2D separable tests (implemented in `test_cuda_vs_cpu.cpp`)

See [`separable_kernel_spec.md`](separable_kernel_spec.md) for the kernel design these
tests exercise. Two tests assert a literal closed-form value (not just agreement with
another implementation — a bug shared between the separable and naive/CPU kernels
wouldn't be caught by cross-comparison alone), reusing fixtures already hand-verified in
`test_cpu_reference.cpp`:

**`Separable2D.KnownZeroPhi`** — same fixture as `CPUReference2D.Separability`:
Xaxis0={−0.5,0.5}, Xaxis1={0,1}, Yaxis0={0}, Yaxis1={0.5}, Phi=0  
→ psi = **0.25** (per-axis split: dim-0 min 0.125, dim-1 min 0.125)

**`Separable2D.KnownNonZeroPhi`** — same fixture as `CPUReference2D.KnownNonZeroPhi`:
Xaxis0={0,1}, Xaxis1={0,1}, Yaxis0={0.5}, Yaxis1={0.5}, Phi={0.1,0.3,0.0,0.2}  
→ psi = **−0.05** (= 0.25 − max(phi))

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

### 2D separable comparisons (implemented)

Two comparisons, both addable to `tests/test_cuda_vs_cpu.cpp` reusing its existing
`maxAbsErr` helper — no new test file or CMake target needed.

**`CudaSeparableVsCPU2D`**: separable GPU output vs. `quadraticCTransformCPU2D` (the
existing joint-loop CPU reference). Reuse the same three fixtures as `CudaVsCPU2D`:
`ZeroPhi`, `NonZeroPhi`, `NonSquareGrid`. `NonSquareGrid` matters specifically here: it's
the case that would catch a pass-1 launch-grid sizing bug (`nx0` vs. `ny0` — see
`separable_kernel_spec.md`), since it's the only fixture where `nx0 ≠ ny0`.

**`CudaSeparableVsNaive2D`**: separable GPU output vs. the existing naive
`quadraticCTransform2D` GPU kernel, same fixtures. This is the comparison called for in
`todo.md`'s Testing section — the two GPU implementations should agree even though
neither is the "reference."

```cpp
TEST(CudaSeparableVsCPU2D, NonSquareGrid) {
    // same fixture as CudaVsCPU2D.NonSquareGrid
    ...
    quadraticCTransformCPU2D(...);
    quadraticCTransform2DSeparable(...);
    EXPECT_LT(maxAbsErr(cpu, separable), 1e-12);
}

TEST(CudaSeparableVsNaive2D, NonSquareGrid) {
    ...
    quadraticCTransform2D(...);            // naive GPU
    quadraticCTransform2DSeparable(...);   // separable GPU
    EXPECT_LT(maxAbsErr(naive, separable), 1e-12);
}
```

**Tolerance note**: per the "Numerical note" in
[`separable_ctransform.md`](../math/separable_ctransform.md), the separable path's
nested min-of-min accumulates `0.5*d0^2 + g(x0,y1)` in a different grouping than the
naive kernel's single joint `0.5*(d0^2+d1^2) - Phi`, so bit-identical output isn't
guaranteed — only equal up to rounding. **Outcome**: on the current tiny fixtures the
standard `1e-12` bound holds for both the vs-CPU and vs-naive comparisons (double
precision), so no loosening was needed. The margin has only been checked on small
unit-domain grids; revisit if larger or non-normalized inputs are added (Tier 3/4).

---

## Tier 3: Randomized tests (implemented)

Implemented in `test_randomized.cpp`. The 2D suite checks **both** the naive and the
separable GPU kernels against the CPU reference in the same loop. `1e-12` holds across all
seeds and shapes.

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

## Tier 4: Stress tests (implemented)

Implemented in `test_stress.cpp` (2D cases exercise naive + separable). **Note**: the
"no CUDA error" cases currently assert finiteness only (`std::isfinite`) — the
`cudaGetLastError`/`cudaDeviceSynchronize` checks below need CUDA headers in the test TU,
which the g++-compiled `.cpp` tests deliberately avoid, so they are not yet asserted.

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
