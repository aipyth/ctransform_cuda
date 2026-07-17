# Dual potentials

## Context: Kantorovich duality

In the Kantorovich formulation of optimal transport, the primal problem minimizes transport cost over couplings. Its dual introduces two potential functions:

$$
\varphi : X \to \mathbb{R} \quad \text{(source potential)}, \qquad \psi : Y \to \mathbb{R} \quad \text{(target potential)}
$$

The dual objective is:

$$
\max_{\varphi, \psi} \left\{ \int \varphi \, d\mu + \int \psi \, d\nu \; : \; \varphi(x) + \psi(y) \leq c(x,y) \;\; \forall x, y \right\}
$$

The c-transform connects $\varphi$ and $\psi$ at the dual optimum: $\psi = \varphi^c$.

---

## c-conjugate pair

A pair $(\varphi, \psi)$ is called a **c-conjugate pair** when:

$$
\psi = \varphi^c \quad \text{and} \quad \varphi = \psi^c
$$

At the dual optimum of the Kantorovich problem, the potentials form a c-conjugate pair. This means:

$$
\psi(y) = \min_{x \in X} \{ c(x,y) - \varphi(x) \}, \qquad \varphi(x) = \min_{y \in Y} \{ c(x,y) - \psi(y) \}
$$

The two equalities hold simultaneously only at optimality.

---

## c-concavity

A function $\varphi$ is called **c-concave** if there exists $\psi$ such that $\varphi = \psi^c$, i.e. $\varphi$ can be written as a c-transform of some function.

Equivalently, $\varphi$ is c-concave if and only if:

$$
\varphi = (\varphi^c)^c
$$

The inequality $(\varphi^c)^c \geq \varphi$ always holds; equality characterizes c-concavity.

The c-transform of any function is always c-concave: $\varphi^c$ is c-concave regardless of whether $\varphi$ itself is.

---

## Inf-convolution interpretation

For fixed $y$, the function $x \mapsto c(x,y) - \varphi(x)$ is a shifted version of the cost. The c-transform:

$$
\varphi^c(y) = \inf_{x \in X} \{ c(x,y) - \varphi(x) \}
$$

is the **inf-convolution** of $c(\cdot,y)$ with $-\varphi$. Geometrically:

- The graph of $-\varphi^c$ is the **lower envelope** of the family of functions $\{ x \mapsto c(x,y) - \varphi(x) \}$ as $x$ varies
- For the squared Euclidean cost, each function $x \mapsto \tfrac{1}{2}\|x-y\|^2 - \varphi(x)$ is a paraboloid translated by $y$; the lower envelope is the Legendre–Fenchel transform up to a sign and quadratic shift

---

## What this project computes

This project computes **one application of the c-transform**: given $\varphi$ and the grids $X$ and $Y$, it returns $\varphi^c$. It does **not**:

- Iterate to find a c-conjugate pair
- Solve the full Kantorovich dual problem
- Compute Wasserstein distances
- Perform Sinkhorn iterations or entropic regularization
- Compute the optimal transport plan or map

The single c-transform step is the inner loop of many OT algorithms (Auction, APDAGD, semi-discrete OT solvers). This library provides that primitive efficiently on GPU.

---

## Relation to convex analysis (squared Euclidean cost)

For $c(x,y) = \tfrac{1}{2}\|x-y\|^2$, the c-transform relates to the classical **Legendre–Fenchel transform** (convex conjugate). Specifically, if we define:

$$
f(x) = \tfrac{1}{2}\|x\|^2 - \varphi(x)
$$

then:

$$
\varphi^c(y) = \tfrac{1}{2}\|y\|^2 - f^*(y)
$$

where $f^*(y) = \sup_x \{ \langle x,y \rangle - f(x) \}$ is the Legendre transform of $f$. This connection means c-concave functions correspond to convex functions via this substitution, and the dual potentials at OT optimality are gradients of convex functions (Brenier's theorem).

---

## Moreau envelope reading

A second, more direct convex-analysis identity: $\varphi^c$ is exactly the Moreau envelope of $-\varphi$ (extended by $+\infty$ off $X$), and its argmin is the corresponding proximal point. This is the identity underlying the still-open argmin/index-output item in [`todo.md`](../engineering/todo.md). See [`moreau_proximal.md`](moreau_proximal.md) for the derivation, the non-uniqueness caveat (this extended function is essentially never convex, so unlike the classical case the proximal point is not guaranteed single-valued), and what this means for readers arriving from proximal/operator-splitting algorithms rather than OT.
