# Cost functions

## Squared Euclidean cost

The only cost function currently implemented is the **squared Euclidean cost with factor ½**:

```
c(x, y) = ½ ‖x − y‖²
```

In 1D:
```
c(x, y) = ½ (x − y)²
```

In 2D:
```
c(x, y) = ½ ((x₀ − y₀)² + (x₁ − y₁)²)
```

The factor ½ is part of the definition and matches the kernel implementations exactly. Do not omit it when verifying results analytically.

---

## How the kernels compute it

**1D kernel** (`quadraticCTransform1D`):
```
T cost = T(0.5) * (xi - yj) * (xi - yj)
```

**2D kernel** (`quadraticCTransform2D`):
```
T d0 = xi0 - yj0;
T d1 = xi1 - yj1;
T cost = T(0.5) * (d0 * d0 + d1 * d1);
```

Both accumulate in type `T` (template parameter). Current instantiation: `T = double`.

---

## Why this cost

The squared Euclidean cost has several favorable mathematical properties:

- **Convexity in x**: for fixed y, c(x,y) is convex and strongly convex in x
- **Brenier's theorem**: under this cost, the optimal transport map (when it exists) is the gradient of a convex function; this connects c-concave potentials to convex functions
- **Decomposability**: in 2D with axis-separated grids, the cost splits as ½d₀² + ½d₁², which the kernel exploits by loading y₀ and y₁ into registers independently
- **Smoothness**: c is C^∞, which simplifies analytical gradient checks

---

## Precision notes

| Scenario | Risk |
|---|---|
| Grid spacing ≈ 1e-3 | c ≈ 5e-7; representable in both float and double |
| Grid spacing ≈ 1 | c ≈ 0.5; no precision issue |
| ‖x−y‖ ≈ 1e4 | c ≈ 5e7; if φ(x) ≈ c(x,y), subtraction loses ~8 digits in double, all digits in float |
| φ near c(x,y) by construction | Catastrophic cancellation; normalize inputs to [0,1]^d to avoid |

The template `T` determines accumulation precision. For inputs normalized to [0,1], `float` is generally safe; for unnormalized or large-scale inputs, `double` is required.

---

## Future cost functions (not yet in scope)

The following cost functions are mathematically interesting for c-transform research but are **not implemented** and are outside the current project scope:

| Cost | Formula | Notes |
|---|---|---|
| L1 (city-block) | c(x,y) = ‖x−y‖₁ | Non-differentiable; different min structure |
| Lp | c(x,y) = ‖x−y‖_p^p / p | Generalizes squared Euclidean (p=2) |
| Separable convex | c(x,y) = Σ f(xᵢ−yᵢ) | Decomposable; exploitable in axis-separated grids |
| Ground metric | c(x,y) = d(x,y)^2 on Riemannian manifold | Requires geometry-specific implementation |

Adding a new cost requires a new kernel variant; the existing template parameter `T` only controls precision, not cost structure.
