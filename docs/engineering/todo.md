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
      standalone spike before committing to it as the default path. Note: the consuming
      solver's existing pure-JAX Legendre transform (`legendre_along_axis_*`) does not
      scale to larger grids — the motivating reason this CUDA kernel exists — so a working
      FH path here is a strict improvement, not a lateral move.

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

      This also blocks autodiff through the JAX FFI target (see
      [`jax_ffi_integration.md`](jax_ffi_integration.md)): by the envelope theorem the
      gradient of the c-transform value w.r.t. $\varphi$ flows only through $x^*(y)$, so
      a custom VJP rule needs a resolved, documented argmin before it can be written.
- [ ] **float32 validation.** `T=float` compiles but is not numerically validated
      (see [`numerical_stability.md`](../math/numerical_stability.md)); needs its own
      tolerance-scaled test tier before being called supported.

## Bindings & integration

- [ ] **Python wrapper — now the active next task**, per
      [`python_extention_plan.md`](python_extention_plan.md): all four prerequisites
      (C++ reference, CUDA-vs-reference, Tier 1–4 tests, stable launch-layer API) are
      met. Two entry points, in build order: (1) NumPy/pybind11-or-nanobind binding,
      `float64` first; (2) JAX FFI target wrapping the `_launch` functions directly, per
      [`jax_ffi_integration.md`](jax_ffi_integration.md) — this is the one the actual
      downstream consumer (a JAX back-and-forth solver) needs, since it runs with zero
      host round-trip inside `lax.while_loop`. Both now live in *this* repo's `python/`
      package (revised from an earlier plan that put the FFI shim in the consuming
      repo) so the solver repo needs no C++ of its own.
- [x] **Device-pointer-only launch layer.** Split each host wrapper into (a) a
      device-pointer/stream-only launch function with no allocation, copy, or
      implicit sync, and (b) an alloc/copy/sync convenience wrapper calling (a).
      Needed for any GPU-resident iterative solver that calls the c-transform every
      iteration without a host round-trip. Done for all three kernels:
      `quadraticCTransform2DSeparable_launch` (separable), and now
      `quadraticCTransform1D_launch` / `quadraticCTransform2D_launch` (naive) — see
      [`api.md`](api.md#gpu-launch-layer--1d-and-2d-naive). Also fixed in the same pass:
      the `int`/`std::size_t` index-width mismatch in both naive kernels' thread-index
      computation (now casts to `std::size_t` before the `blockDim * blockIdx` multiply,
      not just on the final assignment).
## Build & packaging

- [ ] **Two `.so` copies after an editable install.** `cmake --build` writes
      `_ctransform_ffi...so` into `python/ctransform_cuda/` (via `LIBRARY_OUTPUT_DIRECTORY`),
      while `pip install -e .` writes a second copy into site-packages. `pytest` imports the
      first (`conftest.py` inserts `build/` and `python/` onto `sys.path`); downstream repos
      import the second. So a `cmake --build` alone silently leaves consumers on stale
      compiled code — currently harmless only because both copies were built from the same
      source. Accepted deliberately for now; documented in
      [`development.md`](development.md#5-the-two-so-copies). Two candidate fixes:
      (a) set `editable.rebuild = true` under `[tool.scikit-build]`, which recompiles on
      import and collapses the two loops into one, at the cost of a staleness check on every
      import and compiler output appearing mid-script; or (b) drop the
      `LIBRARY_OUTPUT_DIRECTORY` property and let `pip install -e .` be the only path that
      puts the `.so` into the package, which then makes `conftest.py`'s source-tree
      `sys.path` insert dead weight. (b) is the cleaner end state; (a) is the smaller change.
- [ ] **Stray comma in `project(ctransform, LANGUAGES CXX CUDA)`** (`CMakeLists.txt:3`) makes
      the project name literally `ctransform,`. Harmless today — nothing consumes
      `PROJECT_NAME` — but it would leak into a `find_package`/config-package export.

## Testing

- [x] **Tier 3 (randomized) and Tier 4 (stress/large-scale) suites** — implemented
      per [`test_strategy.md`](test_strategy.md) (`test_randomized.cpp`,
      `test_stress.cpp`); both cover naive and separable 2D kernels against the CPU
      reference.
- [ ] **Analytic Hamilton–Jacobi tests.** [`hopf_lax_hj.md`](../math/hopf_lax_hj.md#analytic-test-cases)
      gives four closed-form solutions of $\partial_t u + \tfrac12\|\nabla u\|^2=0$ that the
      existing kernel must reproduce, with no kernel changes required — quadratic data
      (smooth; isolates grid error and gives a convergence-rate probe), $u_0=|x|$ (the
      Huber function; nonsmooth data, tests the kink at $|y|=t$), $u_0=-|x|$ (systematic
      two-point ties at $y=0$), and constant $\varphi$ (squared distance transform;
      future cross-check against Felzenszwalb–Huttenlocher). Stronger than the current
      CPU-reference comparisons, which validate GPU-vs-CPU agreement but not agreement
      with the true continuum answer. The $u_0=-|x|$ case is also the sharpest available
      forcing function for the argmin tie-breaking question above: its *value* is
      unambiguous, its *argmin* is not.
- [x] Tolerance check for the separable kernel specifically: its reduction order
      differs from the naive joint loop (nested min vs. joint min), so add a
      separable-vs-naive comparison in addition to separable-vs-CPU-reference.
      Done — `CudaSeparableVsNaive2D` and `CudaSeparableVsCPU2D` (+ two literal-value
      `Separable2D` cases); `1e-12` holds on current fixtures.
