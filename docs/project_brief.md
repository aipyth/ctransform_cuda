# Project brief

## What this project is

`ctransform_cuda` implements the **discrete c-transform** operator in CUDA C++. The c-transform is the core computational primitive of optimal transport duality: given a potential φ defined on a finite source grid X and a finite target grid Y, it produces the conjugate potential φ^c on Y.

This project is a **standalone kernel library** for that single operation. It is not an optimal transport solver.

**Equivalent framing.** For the squared-Euclidean cost this project implements, the
c-transform is *exactly* the grid-sampled **Moreau envelope** of $-\varphi$ (extended by
$+\infty$ off $X$), and its (currently unexposed) argmin is the corresponding
**proximal point**. So the same kernel is a primitive not only for OT solvers (Auction,
back-and-forth, semi-discrete methods) but for any proximal-gradient, ADMM, or
operator-splitting method that needs a Moreau-envelope/prox evaluation over a finite
point set. See [`docs/math/moreau_proximal.md`](math/moreau_proximal.md) for the exact
identity and its caveats (non-uniqueness of the argmin, grid-discretization error when
approximating a continuous objective).

**Second equivalent framing.** The same operator is the **Hopf–Lax solution operator**
for the Hamilton–Jacobi equation $\partial_t u + \tfrac12\|\nabla u\|^2 = 0$: one call
advances initial data $u_0=-\varphi$ to time $t$, exactly in time (Hopf–Lax is a
representation formula, not a time discretization), with the Moreau scale $\lambda$
playing the role of $t$. See [`docs/math/hopf_lax_hj.md`](math/hopf_lax_hj.md) for the
dictionary, four analytic test cases with closed-form solutions, and — importantly — the
boundaries of the correspondence: obstacles that constrain *paths* (rather than merely
forbidding states) make the effective cost geodesic and therefore non-separable, which
removes both of this library's sub-quadratic acceleration routes.

---

## Formulation in scope

The project computes the **fully discrete c-transform**:

```
φ^c(y) = min_{x ∈ X} { c(x, y) − φ(x) }
```

where:
- X = { x₁, …, x_N } ⊂ ℝ^d is a finite source point cloud
- Y = { y₁, …, y_M } ⊂ ℝ^d is a finite target point cloud
- φ : X → ℝ is the input potential (one value per source point)
- φ^c : Y → ℝ is the output potential (one value per target point)

Both 1D (d=1) and 2D (d=2) are implemented.

---

## Supported cost functions

| Name | Formula | Status |
|---|---|---|
| Squared Euclidean | c(x,y) = ½‖x−y‖² | Implemented |
| L1, Lp, other | — | Out of scope |

The factor ½ is part of the definition and matches the kernel implementations exactly.

---

## Out of scope

The following are explicitly **not** part of this project:

- Full optimal transport plan computation (no transport map, no coupling matrix)
- Sinkhorn or entropic regularization
- Semi-discrete formulations (one continuous, one discrete measure)
- Continuous formulations
- Dual ascent or iterative c-conjugate computation
- W₂ Wasserstein distance computation
- Cost functions other than squared Euclidean
- Hamilton–Jacobi / HJB / mean-field-game **solvers**. The kernel is the Hopf–Lax
  solution operator for one such equation (see
  [`docs/math/hopf_lax_hj.md`](math/hopf_lax_hj.md)), and that identity is documented
  here because it yields analytic tests and clarifies the roadmap — but time-stepping
  loops, boundary/obstacle handling, coupled forward–backward systems, and
  position-dependent Hamiltonians belong to a consuming solver, not to this library.

**No longer out of scope:** Python bindings, previously listed here as future work, are
implemented — a NumPy/pybind11 module and a JAX FFI target, both under `python/`. See
[`docs/engineering/python_extention_plan.md`](engineering/python_extention_plan.md) and
[`docs/engineering/jax_ffi_integration.md`](engineering/jax_ffi_integration.md).

---

## Expected precision

| Precision | Status | Notes |
|---|---|---|
| `double` (float64) | Default and tested | ~15 significant decimal digits |
| `float` (float32) | Compilable via template T | Not yet numerically validated |

The CUDA kernels are templated on `T`. All current instantiations and tests use `T = double`.

---

## Correctness criteria

A result is considered correct when:

1. **CPU reference match**: GPU output agrees with a sequential CPU reference loop to within **1e-12** in absolute error (double precision).
2. **Analytical solutions**: small hand-constructed cases with known closed-form answers pass exactly (to within floating-point rounding).
3. **No CUDA errors**: no `cudaError_t` failures, no NaN or Inf in output for well-posed inputs (finite X, Y, φ).
4. **Determinism**: identical inputs produce bit-identical output on the same hardware and CUDA version (guaranteed by the current sequential per-thread scan).

---

## Performance goals

| Metric | Target |
|---|---|
| GPU vs CPU crossover | GPU faster than single-threaded CPU for N·M ≥ ~10⁶ |
| Algorithm complexity | O(N·M) naive brute force (no acceleration structure yet) |
| Sub-quadratic algorithms | Out of scope for the initial version |

Performance is measured as GPU kernel time (excluding host↔device transfers) versus a sequential CPU reference loop. Transfer overhead is acknowledged but not optimized.
