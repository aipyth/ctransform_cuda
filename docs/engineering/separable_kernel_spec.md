# Separable 2D c-transform — design notes

Design and implementation notes for the separable 2D c-transform kernels. For *why* the
decomposition is exact, see the derivation in
[`separable_ctransform.md`](../math/separable_ctransform.md). For the public function
signatures and the library's layer structure (host wrapper vs. device-pointer launch
layer), see [`api.md`](api.md). This document covers the *kernel-level algorithm*: the two
passes, the sign convention, the index mappings, and the correctness/performance gotchas
to watch when implementing.

The separable transform is exposed through the standard two-layer API
(`quadraticCTransform2DSeparable` and `quadraticCTransform2DSeparable_launch`); the two
kernels below sit underneath the launch layer.

---

## Sign convention

`separable_ctransform.md` defines the pass-1 intermediate implicitly via
`min_{x1}{c1(x1,y1) − φ(x0,x1)} =: −m(x0,y1)`, i.e. `m` is the *negation* of what pass 1
naturally computes. Implementing that literally would require either negating into a
second buffer or subtracting a negative in pass 2. Neither is necessary — define the
intermediate buffer as exactly what pass 1's inner loop produces, call it `g`:

$$
g(x_0, y_1) := \min_{x_1} \Big\{ \tfrac12 (x_1 - y_1)^2 - \varphi(x_0, x_1) \Big\}
\qquad \text{(pass 1 output, shape } (n_{x_0}, n_{y_1})\text{)}
$$

$$
\varphi^c(y_0, y_1) = \min_{x_0} \Big\{ \tfrac12 (x_0 - y_0)^2 + g(x_0, y_1) \Big\}
\qquad \text{(pass 2, reads } g \text{, writes final output)}
$$

Pass 2 **adds** `g` — it does not subtract a negated copy. Using `g = −m` directly removes
a negation from the hot loop and avoids a second buffer.

**Hand-check** (against the "2×2 source, zero phi" fixture in
[`test_strategy.md`](test_strategy.md)): `Xaxis0=Xaxis1={0,1}`, `Yaxis0=Yaxis1={0.5}`,
`Phi` all zero.
Pass 1: `g(0,0) = min(½·0.5², ½·0.5²) = 0.125`; `g(1,0) = 0.125`.
Pass 2: `out(0,0) = min(½·0.5² + 0.125, ½·0.5² + 0.125) = 0.25`.
Matches the known joint-loop answer `0.25`. ✓

---

## Pass 1 — collapse axis 1

Computes `g` as defined above. Structurally this is the existing 1D c-transform
(`quadraticCTransform1DKernel` in `src/ctransform1D_naive.cu`) **batched over `x0`**: each
row `x0` of `Phi` is an independent 1D problem over `Xaxis1`, evaluated at every `Yaxis1`.

- **Thread mapping**: one thread per `(ix0, iy1)` pair; each thread reduces over
  `ix1 ∈ [0, nx1)`, maintaining a running minimum. The fast axis is `iy1` (so the write to
  `g` is coalesced, matching its row-major `(nx0, ny1)` layout — see
  [`memory_layout.md`](memory_layout.md)).
- **Batch axis gotcha**: the slow/batch axis is `ix0`, a **source** axis — there is *no*
  `iy0` in this pass. The launch grid must therefore be sized over `(ny1, nx0)`, i.e. the
  row count is `nx0`, **not** `ny0`. This is the single easiest mistake to make when
  copy-adjusting the naive 2D kernel's launch config, where `ny0` sits in that position.
  A test with `nx0 ≠ ny0` (the `NonSquareGrid` case in `test_strategy.md`) catches it:
  either an illegal-memory-access, or silently truncated/garbage rows of `g` when the grid
  is undersized.

---

## Pass 2 — collapse axis 0

Computes the final output `φ^c(y0, y1)` from `g`.

- **Thread mapping**: one thread per output point `(iy0, iy1)` — the standard 2D output
  shape `(ny0, ny1)`, same as the naive kernel. Each thread reduces over `ix0 ∈ [0, nx0)`,
  reading `g[ix0, iy1]` and **adding** `½(Xaxis0[ix0] − Yaxis0[iy0])²` to it (the `+` from
  the sign convention above), maintaining a running minimum.
- Fast axis `iy1` again gives a coalesced write to `out`.

---

## CUDA review checklist (per [`.claude/rules/cuda_cpp.md`](../../.claude/rules/cuda_cpp.md))

| Check | Notes |
|---|---|
| Bounds checks | Each thread must guard its indices before touching memory (`ix0/iy0` and `iy1` against their axis sizes) |
| Coalesced writes | Both `g[ix0*ny1+iy1]` and `out[iy0*ny1+iy1]` have `iy1` as the fast index → coalesced for `ny1 ≥ 16` |
| Shared memory | None needed; consistent with the existing kernels (tiling is a future optimization, see [`cuda_cpp_architecture.md`](cuda_cpp_architecture.md)) |
| Atomics | None; each output element (and each `g` element) is written by exactly one thread |
| Determinism | Sequential per-thread scan in both passes → fully deterministic, same argument as [`numerical_stability.md`](../math/numerical_stability.md) makes for the naive kernels |
| Scratch (`g`) sizing | Must be `nx0 * ny1` — **not** `ny0*ny1` (that's `out`'s size) or `nx0*nx1` (that's `Phi`'s size). The three coincide for square grids, so only a non-square test catches a mix-up |
| Pass ordering | Pass 2 reads what pass 1 wrote; ordering is guaranteed by issuing both on the same stream (no explicit sync needed *between* passes) |
| Launch / sync / allocation contract | Owned by the launch layer, not the kernels — see the layer description in [`api.md`](api.md): the launch layer does no alloc/copy/sync |
| Error checking | Check `cudaGetLastError()` after each kernel launch |

---

## Numerical stability

No new stability concerns beyond those documented in
[`numerical_stability.md`](../math/numerical_stability.md) for the naive kernels — both
passes use the same `½d² − φ` candidate structure and the `min()` device intrinsic. The
one difference worth restating: pass 2's accumulation order (`½d0² + g(x0,y1)`, where `g`
already folds in a `min` over `x1`) is not the same floating-point operation sequence as
the naive kernel's single joint `½(d0² + d1²) − Phi`, so the two GPU implementations may
differ at the ULP level even though both are mathematically exact minimizations of the
same value. See the "Numerical note" in `separable_ctransform.md` and the tolerance
discussion in [`test_strategy.md`](test_strategy.md).
