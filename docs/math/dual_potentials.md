# Dual potentials

## Context: Kantorovich duality

In the Kantorovich formulation of optimal transport, the primal problem minimizes transport cost over couplings. Its dual introduces two potential functions:

```
φ : X → ℝ   (source potential)
ψ : Y → ℝ   (target potential)
```

The dual objective is:

```
max_{φ,ψ} { ∫ φ dμ + ∫ ψ dν  subject to  φ(x) + ψ(y) ≤ c(x,y)  ∀x,y }
```

The c-transform connects φ and ψ at the dual optimum: **ψ = φ^c**.

---

## c-conjugate pair

A pair (φ, ψ) is called a **c-conjugate pair** when:

```
ψ = φ^c   and   φ = ψ^c
```

At the dual optimum of the Kantorovich problem, the potentials form a c-conjugate pair. This means:

```
ψ(y) = min_{x ∈ X} { c(x,y) − φ(x) }
φ(x) = min_{y ∈ Y} { c(x,y) − ψ(y) }
```

The two equalities hold simultaneously only at optimality.

---

## c-concavity

A function φ is called **c-concave** if there exists ψ such that φ = ψ^c, i.e. φ can be written as a c-transform of some function.

Equivalently, φ is c-concave if and only if:

```
φ = (φ^c)^c
```

The inequality (φ^c)^c ≥ φ always holds; equality characterizes c-concavity.

The c-transform of any function is always c-concave: φ^c is c-concave regardless of whether φ itself is.

---

## Inf-convolution interpretation

For fixed y, the function x ↦ c(x,y) − φ(x) is a shifted version of the cost. The c-transform:

```
φ^c(y) = inf_{x ∈ X} { c(x,y) − φ(x) }
```

is the **inf-convolution** of c(·,y) with −φ. Geometrically:

- The graph of −φ^c is the **lower envelope** of the family of functions { x ↦ c(x,y) − φ(x) } as x varies
- For the squared Euclidean cost, each function x ↦ ½‖x−y‖² − φ(x) is a paraboloid translated by y; the lower envelope is the Legendre–Fenchel transform up to a sign and quadratic shift

---

## What this project computes

This project computes **one application of the c-transform**: given φ and the grids X and Y, it returns φ^c. It does **not**:

- Iterate to find a c-conjugate pair
- Solve the full Kantorovich dual problem
- Compute Wasserstein distances
- Perform Sinkhorn iterations or entropic regularization
- Compute the optimal transport plan or map

The single c-transform step is the inner loop of many OT algorithms (Auction, APDAGD, semi-discrete OT solvers). This library provides that primitive efficiently on GPU.

---

## Relation to convex analysis (squared Euclidean cost)

For c(x,y) = ½‖x−y‖², the c-transform relates to the classical **Legendre–Fenchel transform** (convex conjugate). Specifically, if we define:

```
f(x) = ½‖x‖² − φ(x)
```

then:

```
φ^c(y) = ½‖y‖² − f*(y)
```

where f*(y) = sup_x { ⟨x,y⟩ − f(x) } is the Legendre transform of f. This connection means c-concave functions correspond to convex functions via this substitution, and the dual potentials at OT optimality are gradients of convex functions (Brenier's theorem).
