# c-transform overview

## Mathematical definition

The **c-transform** (also called the c-conjugate or Kantorovich conjugate) of a function φ with respect to a cost function c is:

```
φ^c(y) = min_{x ∈ X} { c(x, y) − φ(x) }
```

- **Input**: potential φ : X → ℝ (one scalar per source point)
- **Output**: potential φ^c : Y → ℝ (one scalar per target point)
- **Cost**: c : X × Y → ℝ, here always c(x,y) = ½‖x−y‖²

The minimum is taken pointwise for each y ∈ Y over all source points x ∈ X.

---

## Discrete formulation

This project uses the **fully discrete** formulation: both X and Y are finite point sets.

```
X = { x₁, …, x_N } ⊂ ℝ^d
Y = { y₁, …, y_M } ⊂ ℝ^d
φ ∈ ℝ^N  (one value per source point)
φ^c ∈ ℝ^M  (one value per target point)
```

No probability measures or weights are needed; the transform operates on the supports and potentials directly.

---

## 1D case

**Source domain**: flat array X[N] of real coordinates  
**Target domain**: flat array Y[M] of real coordinates  
**Cost**: c(xᵢ, yⱼ) = ½ (xᵢ − yⱼ)²

For each j ∈ {0,…,M−1}:
```
φ^c[j] = min_{i=0}^{N-1} { ½ (X[i] − Y[j])² − φ[i] }
```

Storage: X[N], Y[M], Phi[N], out[M] — all flat 1D arrays.

---

## 2D case

**Source domain**: axis-separated grid Xaxis0[nx0] × Xaxis1[nx1], representing N = nx0·nx1 points  
**Target domain**: axis-separated grid Yaxis0[ny0] × Yaxis1[ny1], representing M = ny0·ny1 points  
**Cost**: c(x, y) = ½ ((x₀ − y₀)² + (x₁ − y₁)²)

For each (j0, j1):
```
φ^c[j0, j1] = min_{i0, i1} { ½((Xaxis0[i0] − Yaxis0[j0])² + (Xaxis1[i1] − Yaxis1[j1])²) − Phi[i0·nx1 + i1] }
```

Storage: axes stored separately; Phi and out stored row-major (slow index first).  
See `docs/engineering/memory_layout.md` for full indexing details.

---

## Computational complexity

| Configuration | Complexity |
|---|---|
| 1D, N source, M target | O(N·M) |
| 2D, N=nx0·nx1 source, M=ny0·ny1 target | O(nx0·nx1·ny0·ny1) |

The current implementation is a **naive linear scan**: each output point independently scans all source points. There is no KD-tree, no auction algorithm, no truncation. This is intentional for a correctness baseline.

---

## Key properties of φ^c

**Constant shift**: if φ is replaced by φ + k (constant), then φ^c is replaced by φ^c − k.  
**c-concavity**: φ^c is c-concave, meaning (φ^c)^c ≥ φ pointwise, with equality when φ is itself c-concave.  
**Envelope structure**: φ^c(y) is the pointwise minimum of M affine (in a generalized sense) functions of y; its graph is the lower envelope of the family { c(xᵢ, ·) − φᵢ }.

---

## Numerical hazards

| Hazard | Description |
|---|---|
| Infinity initialization | Each thread starts min-accumulator at `INFINITY`; if N=0 the output is `INFINITY` |
| Cancellation | When c(x,y) ≈ φ(x) with both large, subtraction loses precision |
| `fmin()` on device | Correct CUDA device function for double; use instead of `std::min` in kernel code |
| Float32 precision | Template T=float is valid but introduces ~7-digit precision; not yet validated |
| Non-associativity | Sequential scan is deterministic; a parallel reduction would lose bit-exactness |

Detailed analysis is in `docs/math/numerical_stability.md`.
