# Numerical stability

## Initialization with INFINITY

Each kernel thread initializes its minimum accumulator to `INFINITY` (the IEEE 754 positive infinity for type `T`):

```cpp
T result = INFINITY;
for (int ix = 0; ix < nx; ++ix) {
    result = fmin(result, cost - phi);
}
out[iy] = result;
```

**Safe case**: N ≥ 1 and all costs and φ values are finite → result is finite.

**Risk case**: N = 0 (empty source domain) → output is `INFINITY`. The kernel does not guard against this; callers must ensure N ≥ 1.

**Risk case**: φ(x) = +∞ for all x → all candidates are −∞ and result may be −∞. Avoid storing +∞ in the input potential.

---

## Cancellation with large inputs

The key subtraction is:

```
c(x, y) − φ(x) = ½‖x−y‖² − φ(x)
```

If both terms are large and nearly equal, the subtraction loses relative precision. Example:

- x=1e4, y=0, φ(x)=5e7 → c = 5e7, φ = 5e7; difference is near-zero but computed from two 8-digit (in float) numbers
- In double: 15 digits available, ~7 are lost → result has ~8 significant digits; acceptable
- In float: 7 digits available, ~7 are lost → result has ~0 significant digits; catastrophic

**Recommendation**: normalize inputs so that ‖x−y‖ ≤ 1 and |φ(x)| ≤ 1. On a unit domain [0,1]^d this holds by construction. For unnormalized inputs, double is required and float is unsafe.

---

## float32 vs float64

| Property | float32 | float64 |
|---|---|---|
| Mantissa bits | 24 | 53 |
| Significant digits | ~7 | ~15 |
| Max safe ‖x−y‖ | ≈ 3e3 | ≈ 1e8 |
| Cost at unit spacing | 0.5 (exact) | 0.5 (exact) |
| Recommended for | small normalized grids | general use |

The kernel template `T = float` compiles and runs but is not numerically validated. Until float-specific tests are added, treat float32 output as approximate.

---

## `fmin()` on device

The inner loop uses `fmin(result, val)` rather than `std::min` or the ternary operator. This is intentional:

- `fmin()` is an IEEE 754-2008 device intrinsic available in CUDA device code for both float and double
- If one argument is NaN, `fmin` returns the other (NaN-propagation is asymmetric but well-defined)
- `std::min` may not compile in all CUDA device code contexts and has different NaN semantics
- The ternary `result < val ? result : val` propagates NaN from `val` but not from `result`; consistent behavior with `fmin` requires explicit NaN checks

**Invariant**: if both inputs are finite, `fmin` is exact. If either is NaN, output is the other value (which may or may not be desired).

---

## Determinism

The current implementation assigns one output point per thread and performs a **sequential scan** over all source points within that thread:

```
for ix in range(N):
    result = fmin(result, cost(X[ix], Y[iy]) - Phi[ix])
```

This is fully deterministic: no atomics, no parallel reduction, no warp shuffles. Identical inputs on the same GPU produce bit-identical outputs.

If a parallel reduction is added in the future (e.g., shared-memory tree reduction), it must use a fixed reduction order or accept non-determinism for extreme inputs where `fmin` on nearly equal values could produce different results depending on thread scheduling.

---

## Shared memory and bandwidth

The current implementation reads source coordinates (`X` or `Xaxis0`, `Xaxis1`) and `Phi` from global memory in a sequential inner loop. For large N:

- Each thread reads N floats/doubles from `Phi` (not coalesced across threads, but streaming)
- No shared memory tiling is used
- L2 cache may absorb repeated reads of the same `Xaxis0[ix0]` row across threads in the same block

**For correctness**: this is safe. Numerical result is identical to a CPU loop.  
**For performance**: at large N, the kernel is memory-bandwidth-bound. Shared memory tiling of source coordinates could improve throughput but is not yet implemented.

---

## Summary table

| Issue | Current behavior | Risk level | Mitigation |
|---|---|---|---|
| N=0 empty source | Output = INFINITY | Medium | Caller must ensure N≥1 |
| Large unnormalized inputs, float32 | Catastrophic cancellation | High | Use double or normalize to [0,1]^d |
| Large normalized inputs, double | ~8 digits lost | Low | Acceptable for unit-domain inputs |
| NaN in φ | Propagates via fmin | Medium | Validate inputs before calling |
| Non-determinism | Not present | None | Sequential per-thread scan |
| Overflow in ½‖x−y‖² | Possible if ‖x−y‖ > sqrt(DBL_MAX/0.5) ≈ 1.3e154 | Negligible | Physical inputs never reach this |
