# c-transform overview

## Mathematical definition

The **c-transform** (also called the c-conjugate or Kantorovich conjugate) of a function $\varphi$ with respect to a cost function $c$ is:

$$
\varphi^c(y) = \min_{x \in X} \left\{ c(x, y) - \varphi(x) \right\}
$$

- **Input**: potential $\varphi : X \to \mathbb{R}$ (one scalar per source point)
- **Output**: potential $\varphi^c : Y \to \mathbb{R}$ (one scalar per target point)
- **Cost**: $c : X \times Y \to \mathbb{R}$, here always $c(x,y) = \tfrac{1}{2}\|x-y\|^2$

The minimum is taken pointwise for each $y \in Y$ over all source points $x \in X$.

---

## Discrete formulation

This project uses the **fully discrete** formulation: both $X$ and $Y$ are finite point sets.

$$
\begin{aligned}
X &= \{x_1, \dots, x_N\} \subset \mathbb{R}^d \\
Y &= \{y_1, \dots, y_M\} \subset \mathbb{R}^d \\
\varphi &\in \mathbb{R}^N \quad \text{(one value per source point)} \\
\varphi^c &\in \mathbb{R}^M \quad \text{(one value per target point)}
\end{aligned}
$$

No probability measures or weights are needed; the transform operates on the supports and potentials directly.

---

## 1D case

**Source domain**: flat array $X[N]$ of real coordinates
**Target domain**: flat array $Y[M]$ of real coordinates
**Cost**: $c(x_i, y_j) = \tfrac{1}{2}(x_i - y_j)^2$

For each $j \in \{0, \dots, M-1\}$:

$$
\varphi^c[j] = \min_{i=0}^{N-1} \left\{ \tfrac{1}{2}(X[i] - Y[j])^2 - \varphi[i] \right\}
$$

Storage: `X[N]`, `Y[M]`, `Phi[N]`, `out[M]` — all flat 1D arrays.

---

## 2D case

**Source domain**: axis-separated grid $X_{\text{axis0}}[n_{x_0}] \times X_{\text{axis1}}[n_{x_1}]$, representing $N = n_{x_0} n_{x_1}$ points
**Target domain**: axis-separated grid $Y_{\text{axis0}}[n_{y_0}] \times Y_{\text{axis1}}[n_{y_1}]$, representing $M = n_{y_0} n_{y_1}$ points
**Cost**: $c(x, y) = \tfrac{1}{2}\left((x_0 - y_0)^2 + (x_1 - y_1)^2\right)$

For each $(j_0, j_1)$:

$$
\varphi^c[j_0, j_1] = \min_{i_0, i_1} \left\{ \tfrac{1}{2}\Big((X_{\text{axis0}}[i_0] - Y_{\text{axis0}}[j_0])^2 + (X_{\text{axis1}}[i_1] - Y_{\text{axis1}}[j_1])^2\Big) - \text{Phi}[i_0 \cdot n_{x_1} + i_1] \right\}
$$

Storage: axes stored separately; `Phi` and `out` stored row-major (slow index first).
See [`docs/engineering/memory_layout.md`](../engineering/memory_layout.md) for full indexing details.

---

## Computational complexity

| Configuration | Complexity |
|---|---|
| 1D, $N$ source, $M$ target | $O(NM)$ |
| 2D, $N = n_{x_0} n_{x_1}$ source, $M = n_{y_0} n_{y_1}$ target | $O(n_{x_0} n_{x_1} n_{y_0} n_{y_1})$ |

The current implementation is a **naive linear scan**: each output point independently scans all source points. There is no KD-tree, no auction algorithm, no truncation. This is intentional for a correctness baseline.

---

## Key properties of $\varphi^c$

**Constant shift**: if $\varphi$ is replaced by $\varphi + k$ (constant), then $\varphi^c$ is replaced by $\varphi^c - k$.
**c-concavity**: $\varphi^c$ is c-concave, meaning $(\varphi^c)^c \geq \varphi$ pointwise, with equality when $\varphi$ is itself c-concave.
**Envelope structure**: $\varphi^c(y)$ is the pointwise minimum of $M$ affine (in a generalized sense) functions of $y$; its graph is the lower envelope of the family $\{c(x_i, \cdot) - \varphi_i\}$.

---

## Numerical hazards

| Hazard | Description |
|---|---|
| Infinity initialization | Each thread starts min-accumulator at `INFINITY`; if $N=0$ the output is `INFINITY` |
| Cancellation | When $c(x,y) \approx \varphi(x)$ with both large, subtraction loses precision |
| `min()` on device | CUDA intrinsic used instead of `std::fmin` (illegal in `__global__` code under C++20); maps to the same IEEE 754-2008 `minNum` hardware instruction |
| Float32 precision | Template `T=float` is valid but introduces ~7-digit precision; not yet validated |
| Non-associativity | Sequential scan is deterministic; a parallel reduction would lose bit-exactness |

Detailed analysis is in [`docs/math/numerical_stability.md`](numerical_stability.md).
