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
- Python bindings (planned future work; see [`docs/engineering/python_extention_plan.md`](engineering/python_extention_plan.md))
- Cost functions other than squared Euclidean

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
