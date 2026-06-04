# ctransform_cuda

A standalone CUDA C++ library that computes the **discrete c-transform** operator — the core computational primitive of optimal transport duality.

## What it does

Given a potential $\varphi : X \to \mathbb{R}$ defined on a finite source point cloud $X \subset \mathbb{R}^d$ and a finite target point cloud $Y \subset \mathbb{R}^d$, the library produces the c-conjugate potential $\varphi^c : Y \to \mathbb{R}$:

$$\varphi^c(y) = \min_{x \in X} \bigl\{ c(x, y) - \varphi(x) \bigr\}$$

with the squared Euclidean cost:

$$c(x, y) = \tfrac{1}{2} \|x - y\|^2$$

The minimum is taken pointwise for each $y \in Y$ over all source points $x \in X$. Both 1D and 2D point clouds are supported.

This is **not** an optimal transport solver. It computes a single c-transform evaluation with no transport plan, no coupling matrix, and no iterative dual ascent.

## Formulation

The library operates on the **fully discrete** formulation:

$$X = \{x_1, \dots, x_N\} \subset \mathbb{R}^d, \quad Y = \{y_1, \dots, y_M\} \subset \mathbb{R}^d$$

$$\varphi \in \mathbb{R}^N \;\longrightarrow\; \varphi^c \in \mathbb{R}^M$$

No probability measures or weights are required; the transform operates on supports and potentials directly.

**1D** — for each $j \in \{0, \dots, M-1\}$:

$$\varphi^c[j] = \min_{i=0}^{N-1} \left\{ \tfrac{1}{2}(X[i] - Y[j])^2 - \varphi[i] \right\}$$

**2D** — source grid $n_{x_0} \times n_{x_1}$, target grid $n_{y_0} \times n_{y_1}$, for each $(j_0, j_1)$:

$$\varphi^c[j_0, j_1] = \min_{i_0, i_1} \left\{ \tfrac{1}{2}\bigl((X_0[i_0] - Y_0[j_0])^2 + (X_1[i_1] - Y_1[j_1])^2\bigr) - \varphi[i_0 \cdot n_{x_1} + i_1] \right\}$$

In 2D, grids are represented by their axis coordinates separately; potentials and outputs are stored row-major.

| Dimension | Complexity |
|---|---|
| 1D | $O(N \cdot M)$ |
| 2D | $O(n_{x_0} \cdot n_{x_1} \cdot n_{y_0} \cdot n_{y_1})$ |

## Documentation

Detailed documentation is in `docs/`:

- `docs/project_brief.md` — scope, correctness criteria, performance targets
- `docs/math/ctransform_overview.md` — mathematical definition and key properties
- `docs/math/cost_functions.md` — cost function derivations and precision analysis
- `docs/math/dual_potentials.md` — duality and c-concavity
- `docs/math/numerical_stability.md` — numerical hazards and mitigation
- `docs/engineering/cuda_cpp_architecture.md` — kernel design and module layout
- `docs/engineering/memory_layout.md` — indexing conventions
- `docs/engineering/test_strategy.md` — testing approach

## License

See [LICENSE](LICENSE).
