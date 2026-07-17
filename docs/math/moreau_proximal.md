# Moreau envelope and proximal operator

This page reframes the existing quadratic-cost c-transform in convex-analysis terms.
**No kernel changes are implied by this page** — the identity below holds for the code
exactly as it is today; only the still-open argmin/index output (see
[`todo.md`](../engineering/todo.md)) is new work, not the identity itself.

---

## The exact identity

For the squared-Euclidean cost this project uses ($c(x,y)=\tfrac12\|x-y\|^2$, fixed per
[`cost_functions.md`](cost_functions.md)), define

$$
g_{\varphi,X}(x) = \begin{cases} -\varphi(x) & x \in X \\ +\infty & x \notin X \end{cases}
$$

— the standard convex-analysis extension of a function defined only on a finite set.
Then

$$
\varphi^c(y) = \min_{x\in X}\{\tfrac12\|x-y\|^2 - \varphi(x)\}
             = \inf_{x\in\mathbb R^d}\{\tfrac12\|x-y\|^2 + g_{\varphi,X}(x)\}
             = M_{g_{\varphi,X}}(y)
$$

the **Moreau envelope** of $g_{\varphi,X}$ at $\lambda=1$. This is exact, not an
approximation: the $+\infty$ term outside $X$ guarantees the continuum infimum and the
grid minimum agree.

A general scale $\lambda$ ($c(x,y)=\tfrac{1}{2\lambda}\|x-y\|^2$) needs no new kernel:
rescale $X\to X/\sqrt\lambda$, $Y\to Y/\sqrt\lambda$ before the call, since
$\tfrac1{2\lambda}\|x-y\|^2 = \tfrac12\|x/\sqrt\lambda - y/\sqrt\lambda\|^2$ and
$\varphi$ is unaffected by relabeling its domain.

This is a different (though related) classical object from the Legendre–Fenchel
connection already in [`dual_potentials.md`](dual_potentials.md#relation-to-convex-analysis-squared-euclidean-cost);
the two are tied together by Moreau's decomposition
$M_f(y) + M_{f^*}(y) = \tfrac12\|y\|^2$ for convex $f$, which does not directly apply
here since $g_{\varphi,X}$ is not convex (next section) — noted for completeness, not
used below.

---

## Why the classical smoothness/uniqueness guarantees do not transfer

Standard Moreau-envelope theory ($g$ proper, convex, lower semicontinuous) gives:
$M_g$ is everywhere finite and $C^{1,1}$ (differentiable with Lipschitz gradient), and
$\operatorname{prox}_g(y) = \arg\min_x\{g(x)+\tfrac12\|x-y\|^2\}$ is single-valued and
firmly nonexpansive.

$g_{\varphi,X}$ is the indicator of a finite point set plus arbitrary values — this is
**never convex** unless $|X|=1$. So:

- $\varphi^c$ is still finite and well-defined everywhere (a min over a finite set
  always exists), and it's still the lower envelope of finitely many paraboloids —
  exactly the "envelope structure" already documented in
  [`ctransform_overview.md`](ctransform_overview.md#key-properties-of-varphic). But it
  is only piecewise smooth: it has kinks exactly where two or more $x\in X$ tie for the
  minimizer.
- The argmin — the discrete proximal point — is **not guaranteed unique**. Ties are not
  a measure-zero edge case here the way they are for generic continuous convex $g$: on
  regular/lattice inputs (a common case for this library, e.g. tensor-product grids),
  ties occur systematically wherever $y$ sits on a Voronoi-type boundary between two
  source points, e.g. any $y$ equidistant from two $x\in X$ with equal $\varphi$.

---

## What this means for the argmin/index-output TODO

`todo.md`'s open item ("argmin/index output") is asking, in this language, to expose
$\operatorname{prox}_{g_{\varphi,X}}(y) = \arg\min_{x\in X}\{c(x,y)-\varphi(x)\}$. Two
separate questions, both still open:

1. **Tie-breaking convention.** Because uniqueness isn't guaranteed (previous section),
   an `argmin_out` needs an explicit, documented rule for ties (e.g. lowest index among
   minimizers) — today's value-only kernel has no need for one, since tied $x$'s produce
   the same output value regardless of which is "selected" by the `min()` reduction.
2. **Match to Jacobs' back-and-forth transport map** (unchanged from the existing
   `todo.md` note): Jacobs recovers the transport map via pushforward
   interpolation/Jacobian pullback, not by reading off this argmin directly. Whether the
   two coincide is unverified and cost-function/measure-specific — check against the
   paper's construction before assuming they're interchangeable.

These are independent: (1) is a generic proximal-operator well-posedness question that
would need answering even outside the OT context; (2) is specific to one downstream
consumer's algorithm.

---

## Scoping note for proximal-algorithm readers

If you're arriving at this library from ADMM / proximal-gradient / operator-splitting
rather than OT: this kernel computes the **exact** Moreau envelope / prox of a function
given *only on the finite set $X$ you pass in*. If your actual objective is a continuous
function (an $\ell_1$ penalty, a smooth regularizer, …) and $X$ is a discretization grid
of its domain, this gives the exact prox of the *discrete restriction*, not the exact
continuous prox — grid resolution controls the approximation error, same as evaluating
any continuous function from samples. Other cost functions (needed for proxes of norms
other than the quadratic, e.g. $\ell_1$ for soft-thresholding) are listed as future,
unimplemented work in [`cost_functions.md`](cost_functions.md#future-cost-functions-not-yet-in-scope).
