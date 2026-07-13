# Separable c-transform

## Motivation

The 2D kernel already stores $X$ and $Y$ as axis-separated tensor-product grids
(see [`memory_layout.md`](../engineering/memory_layout.md)), but the current kernel
still evaluates the transform as one joint minimization over all $(i_0, i_1)$ pairs,
i.e. $O(n_{x_0} n_{x_1} n_{y_0} n_{y_1})$. The cost function itself has more structure
than that loop uses: it is **additively separable across axes**, and that structure
lets the joint minimization be replaced by a sequence of cheaper 1D minimizations.

---

## The separability property

A cost is additively separable across axes if it decomposes as a sum of per-axis
costs:

$$
c(x, y) = \sum_{k=0}^{d-1} c_k(x_k, y_k)
$$

The squared Euclidean cost used in this project satisfies this with
$c_k(x_k, y_k) = \tfrac{1}{2}(x_k - y_k)^2$. Note this property is **not specific to
the quadratic cost**: any $L_p$ cost $c(x,y) = \|x-y\|_p^p / p = \sum_k |x_k-y_k|^p / p$
is separable in the same sense (see [`cost_functions.md`](cost_functions.md)). What
follows depends only on separability of $c$, not on which per-axis cost is used, and
places no requirement on $\varphi$ itself — $\varphi$ need not be separable.

---

## Derivation (2D case)

$$
\varphi^c(y_0, y_1) = \min_{x_0, x_1} \big\{ c_0(x_0, y_0) + c_1(x_1, y_1) - \varphi(x_0, x_1) \big\}
$$

Since $c_0(x_0, y_0)$ does not depend on $x_1$, it can be pulled out of the inner
minimization over $x_1$:

$$
\varphi^c(y_0, y_1) = \min_{x_0} \Big\{ c_0(x_0, y_0) + \underbrace{\min_{x_1} \big\{ c_1(x_1, y_1) - \varphi(x_0, x_1) \big\}}_{=: -m(x_0, y_1)} \Big\}
$$

Both the inner and outer minimizations are themselves 1D c-transforms:

- **Pass 1** (collapse axis 1): for each fixed row $x_0$, compute
  $m(x_0, \cdot)$ as the 1D c-transform of $\varphi(x_0, \cdot)$ (as a source
  potential over $X_{\text{axis1}}$) evaluated at every $y_1 \in Y_{\text{axis1}}$.
  This produces an intermediate array $m$ of shape $(n_{x_0}, n_{y_1})$.
- **Pass 2** (collapse axis 0): for each fixed column $y_1$, compute
  $\varphi^c(\cdot, y_1)$ as the 1D c-transform of $-m(\cdot, y_1)$ (as a source
  potential over $X_{\text{axis0}}$) evaluated at every $y_0 \in Y_{\text{axis0}}$.

This equality is exact — it follows from distributing $+$ over $\min$, not from any
approximation. The order of axis collapse is arbitrary (axis 1 then axis 0, or vice
versa); for non-square grids, collapsing in an order chosen to minimize the
intermediate array size is a free optimization.

---

## Complexity

| Algorithm | Complexity |
|---|---|
| Naive joint loop (current 2D kernel) | $O(n_{x_0} n_{x_1} n_{y_0} n_{y_1})$ |
| Separable (two 1D passes) | $O(n_{x_0} n_{x_1} n_{y_1} + n_{x_0} n_{y_0} n_{y_1})$ |

For a symmetric grid with all axis sizes equal to $n$: $O(2n^3)$ vs. $O(n^4)$ — an
$O(n)$ speedup that grows with grid size, at the cost of one intermediate buffer of
size $n_{x_0} \times n_{y_1}$ and two sequential kernel launches instead of one.

In $d$ dimensions the same argument applies inductively: $d$ sequential passes, each
collapsing one axis via the same 1D primitive, replace the single $O(N \cdot M)$
joint loop ($N = \prod n_{x_k}$, $M = \prod n_{y_k}$) with $d$ passes each costing at
most $O(N \cdot M / \min_k n_{y_k})$ — see [`ctransform_overview.md`](ctransform_overview.md#computational-complexity)
for the corresponding naive-case complexity table this improves on.

---

## Numerical note

The separable algorithm computes a mathematically identical result to the joint
loop, but via a different reduction order (nested min-of-min rather than a single
joint min), so it is **not guaranteed bit-identical** in floating point — only equal
up to rounding. Existing tests compare GPU output to a CPU reference at `1e-12`
absolute tolerance (see [`test_strategy.md`](../engineering/test_strategy.md)); the
separable kernel should be checked against both the CPU reference *and* the existing
naive GPU kernel, since the two GPU implementations may now legitimately disagree at
the ULP level even though both are correct.
