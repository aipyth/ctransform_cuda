# Numerical stability

## Initialization with INFINITY

Each kernel thread initializes its minimum accumulator to `INFINITY` (the IEEE 754 positive infinity for type `T`):

```cpp
T result = INFINITY;
for (int ix = 0; ix < nx; ++ix) {
    result = min(result, cost - phi);
}
out[iy] = result;
```

**Safe case**: $N \geq 1$ and all costs and $\varphi$ values are finite → result is finite.

**Risk case**: $N = 0$ (empty source domain) → output is `INFINITY`. The kernel does not guard against this; callers must ensure $N \geq 1$.

**Risk case**: $\varphi(x) = +\infty$ for all $x$ → all candidates are $-\infty$ and result may be $-\infty$. Avoid storing $+\infty$ in the input potential.

---

## Cancellation with large inputs

The key subtraction is:

$$
c(x, y) - \varphi(x) = \tfrac{1}{2}\|x-y\|^2 - \varphi(x)
$$

If both terms are large and nearly equal, the subtraction loses relative precision. Example:

- $x = 10^4, y = 0, \varphi(x) = 5 \times 10^7$ → $c = 5 \times 10^7$, $\varphi = 5 \times 10^7$; difference is near-zero but computed from two 8-digit (in float) numbers
- In double: 15 digits available, ~7 are lost → result has ~8 significant digits; acceptable
- In float: 7 digits available, ~7 are lost → result has ~0 significant digits; catastrophic

**Recommendation**: normalize inputs so that $\|x-y\| \leq 1$ and $|\varphi(x)| \leq 1$. On a unit domain $[0,1]^d$ this holds by construction. For unnormalized inputs, double is required and float is unsafe.

---

## float32 vs float64

| Property | float32 | float64 |
|---|---|---|
| Mantissa bits | 24 | 53 |
| Significant digits | ~7 | ~15 |
| Max safe $\|x-y\|$ | $\approx 3 \times 10^3$ | $\approx 10^8$ |
| Cost at unit spacing | 0.5 (exact) | 0.5 (exact) |
| Recommended for | small normalized grids | general use |

The kernel template `T = float` compiles and runs but is not numerically validated. Until float-specific tests are added, treat float32 output as approximate.

---

## `min()` on device

The inner loop uses the CUDA device intrinsic `min(result, val)` rather than `std::fmin` or the ternary operator. This is intentional:

- In C++20, `std::fmin` is declared `constexpr __host__` and cannot be called from `__global__` functions
- The CUDA `min()` intrinsic maps to the same hardware instruction as `fmin`/`fminf` (IEEE 754-2008 `minNum` semantics) for both float and double, so the numerical behavior is identical
- If one argument is NaN, `min()` returns the other (NaN-propagation is asymmetric but well-defined)
- The ternary `result < val ? result : val` propagates NaN from `val` but not from `result`; consistent behavior with `min()` requires explicit NaN checks

**Invariant**: if both inputs are finite, `min()` is exact. If either is NaN, output is the other value (which may or may not be desired).

---

## Determinism

The current implementation assigns one output point per thread and performs a **sequential scan** over all source points within that thread:

```
for ix in range(N):
    result = min(result, cost(X[ix], Y[iy]) - Phi[ix])
```

This is fully deterministic: no atomics, no parallel reduction, no warp shuffles. Identical inputs on the same GPU produce bit-identical outputs.

If a parallel reduction is added in the future (e.g., shared-memory tree reduction), it must use a fixed reduction order or accept non-determinism for extreme inputs where `min()` on nearly equal values could produce different results depending on thread scheduling.

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
| $N=0$ empty source | Output = INFINITY | Medium | Caller must ensure $N \geq 1$ |
| Large unnormalized inputs, float32 | Catastrophic cancellation | High | Use double or normalize to $[0,1]^d$ |
| Large normalized inputs, double | ~8 digits lost | Low | Acceptable for unit-domain inputs |
| NaN in $\varphi$ | Propagates via `min()` | Medium | Validate inputs before calling |
| Non-determinism | Not present | None | Sequential per-thread scan |
| Overflow in $\tfrac{1}{2}\|x-y\|^2$ | Possible if $\|x-y\| > \sqrt{\text{DBL\_MAX}/0.5} \approx 1.3 \times 10^{154}$ | Negligible | Physical inputs never reach this |
