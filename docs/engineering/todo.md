# Project TODO

Ideas and planned work worth trying, roughly ordered by the priority rules in
[`python_extension.md`](../../.claude/rules/python_extension.md): correct reference
behavior and numerical coverage come before the API surface and bindings.

This list tracks intent, not commitments. Check items off as they land; add a line
to the relevant doc (architecture, api, test_strategy) when an item is completed so
this file doesn't drift into being the only record of a decision.

---

## Algorithms

- [x] **Separable 2D c-transform kernel.** Exploit additive separability of the cost
      across axes: two sequential 1D passes (collapse axis 1, then axis 0) via an
      intermediate buffer, instead of the joint double loop. Reduces complexity from
      $O(n_{x_0}n_{x_1}n_{y_0}n_{y_1})$ to $O(n_{x_0}n_{x_1}n_{y_1} + n_{x_0}n_{y_0}n_{y_1})$.
      Implemented in `src/ctransform2D_separable.cu` (two kernels + `_launch` device layer
      + host wrapper); design notes in [`separable_kernel_spec.md`](separable_kernel_spec.md).
- [ ] **GPU parabola lower-envelope (Felzenszwalt–Huttenlocher generalized distance
      transform).** $O(n)$ per line instead of $O(n^2)$, for the squared-Euclidean
      cost specifically. Well-established on CPU; GPU viability is unproven — the
      per-line algorithm is a sequential stack scan, so parallelism must come from
      running one thread (or one block) per line rather than within a line. Worth a
      standalone spike before committing to it as the default path.

## Dimensionality

- [ ] **3D kernels** (naive first, matching the existing 1D/2D pattern; separable
      pass extends naturally to 3 sequential axis collapses once the 2D separable
      kernel is validated). Needs a `Grid3D` struct and a 3-axis extension of the
      axis-separated representation in [`memory_layout.md`](memory_layout.md).

## API / naming

- [x] **Rename host wrappers** `quadraticCTransform`/`quadraticCTransformCPU`
      (previously overloaded by signature) to `quadraticCTransform1D` /
      `quadraticCTransform2D` (and CPU equivalents), matching the kernel names and
      the planned Python names (`ctransform_1d`/`ctransform_2d`).
- [ ] **Argmin/index output — two separate open questions, both need resolving before
      implementing.** $x^*(y) = \arg\min_x\{c(x,y)-\varphi(x)\}$ is the discrete
      **proximal point** of $-\varphi$ (extended by $+\infty$ off $X$) — see
      [`moreau_proximal.md`](../math/moreau_proximal.md) for the derivation.
      1. **Non-uniqueness / tie-breaking.** The extended $-\varphi$ is essentially never
         convex (indicator of a finite set + arbitrary values), so unlike the classical
         proximal operator, $x^*(y)$ is not guaranteed single-valued — and ties are
         systematic, not a measure-zero edge case, on regular/lattice inputs (any $y$
         equidistant from two $x\in X$ with equal $\varphi$). An `int* argmin_out` needs
         an explicit, documented tie-breaking rule (e.g. lowest index); today's
         value-only kernel has no need for one since tied $x$'s give the same output
         value regardless of which the `min()` reduction effectively picks.
      2. **Match to Matt Jacobs' back-and-forth method** (not the Auction algorithm),
         the target use case. Jacobs recovers the transport map via pushforward
         interpolation / Jacobian pullback rather than reading off $x^*(y)$ directly
         from the c-transform's inner minimization. Whether the kernel's argmin
         coincides with the map Jacobs' interpolation step produces (same map, or
         merely an equally valid but different-in-general selection when $x^*(y)$ is
         non-unique) is unverified — check against the paper's construction before
         assuming the two are interchangeable.
- [ ] **float32 validation.** `T=float` compiles but is not numerically validated
      (see [`numerical_stability.md`](../math/numerical_stability.md)); needs its own
      tolerance-scaled test tier before being called supported.

## Bindings & integration

- [ ] **Python wrapper** per [`python_extention_plan.md`](python_extention_plan.md) —
      pybind11 or nanobind, `float64` first, after Tier 1–4 tests pass.
- [ ] **Device-pointer-only launch layer.** Split each host wrapper into (a) a
      device-pointer/stream-only launch function with no allocation, copy, or
      implicit sync, and (b) today's alloc/copy/sync convenience wrapper calling (a).
      Needed for any GPU-resident iterative solver that calls the c-transform every
      iteration without a host round-trip. **Done for 2D separable**
      (`quadraticCTransform2DSeparable_launch`); still to do for `quadraticCTransform1D`
      and `quadraticCTransform2D` (naive). The `_launch` declaration is exposed in
      `ctransform.hpp` behind a g++-safe forward declaration of `cudaStream_t`; apply the
      same header pattern when adding the 1D/2D launch functions. Signatures and launch
      config drafted in [`api.md`](api.md#gpu-launch-layer--1d-and-2d-naive-spec-not-yet-implemented);
      also flags a pre-existing `int`/`std::size_t` index-width mismatch in
      `quadraticCTransform1DKernel` worth fixing in the same pass.
- [ ] **JAX custom-call / FFI integration.** For an exact (non-entropic) GPU-resident
      solver driven by `lax.while_loop`, register the device-pointer launch function
      as an XLA custom call so JAX never touches host memory mid-iteration. Depends
      on the device-pointer-only layer above, not on the pybind11/NumPy binding.
      **The FFI shim itself lives in the consuming solver repo, not here** — see
      [`jax_ffi_integration.md`](jax_ffi_integration.md) for the repo boundary, the
      constraints this places on the launch layer (static shapes, XLA-visible scratch,
      async/non-throwing error reporting), and a test-coverage gap it surfaces (no
      existing test calls `_launch` directly on pre-staged device buffers for the
      naive kernels, once they exist).

## Testing

- [x] **Tier 3 (randomized) and Tier 4 (stress/large-scale) suites** — implemented
      per [`test_strategy.md`](test_strategy.md) (`test_randomized.cpp`,
      `test_stress.cpp`); both cover naive and separable 2D kernels against the CPU
      reference.
- [x] Tolerance check for the separable kernel specifically: its reduction order
      differs from the naive joint loop (nested min vs. joint min), so add a
      separable-vs-naive comparison in addition to separable-vs-CPU-reference.
      Done — `CudaSeparableVsNaive2D` and `CudaSeparableVsCPU2D` (+ two literal-value
      `Separable2D` cases); `1e-12` holds on current fixtures.
