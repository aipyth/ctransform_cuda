# Hopf–Lax and Hamilton–Jacobi equations

This page records an exact identity: **the quadratic-cost c-transform this library
computes is the Hopf–Lax solution operator for a first-order Hamilton–Jacobi equation.**

As with [`moreau_proximal.md`](moreau_proximal.md), **no kernel changes are implied**.
The identity holds for the code exactly as it is today. What the page adds is (a) a
family of analytic test cases with closed-form solutions, and (b) an explicit statement
of where the correspondence stops — which matters because the natural extensions
(obstacles, state constraints, position-dependent costs) break the assumptions that
make this library's *fast* kernels fast.

---

## The correspondence

Consider the Hamilton–Jacobi Cauchy problem

$$
\partial_t u + H(\nabla u) = 0 \quad \text{on } \mathbb{R}^d\times(0,\infty),
\qquad u(\cdot,0) = u_0 .
$$

For $H$ convex and superlinear, with Lagrangian $L = H^*$, the **Hopf–Lax formula**
gives the unique viscosity solution:

$$
u(y,t) = \min_{x\in\mathbb{R}^d}\Big\{ u_0(x) + t\,L\!\left(\tfrac{y-x}{t}\right) \Big\}.
$$

Specialize to $H(p)=\tfrac12\|p\|^2$, so $L(v)=\tfrac12\|v\|^2$ and
$t\,L(\tfrac{y-x}{t}) = \tfrac{\|x-y\|^2}{2t}$:

$$
\boxed{\;u(y,t) = \min_{x}\Big\{ u_0(x) + \frac{\|x-y\|^2}{2t} \Big\}\;}
$$

Now compare the c-transform this library computes
($c(x,y)=\tfrac12\|x-y\|^2$, per [`cost_functions.md`](cost_functions.md)):

$$
\varphi^c(y) = \min_{x\in X}\Big\{\tfrac12\|x-y\|^2 - \varphi(x)\Big\}.
$$

These are the same expression under the dictionary:

| c-transform | Hamilton–Jacobi |
|---|---|
| potential $\varphi$ | initial data $u_0 = -\varphi$ |
| output $\varphi^c$ | solution $u(\cdot,t)$ |
| Moreau scale $\lambda$ (see [`moreau_proximal.md`](moreau_proximal.md)) | **time** $t$ |
| argmin $x^*(y)$ | foot of the optimal characteristic reaching $y$ at time $t$ |
| cost $c$ | $t\,L(\cdot/t)$ for the Lagrangian $L$ |

The third row is the one worth internalizing: the scale parameter $\lambda$ documented in
`moreau_proximal.md` as a convex-analysis rescaling **is physical time**. The rescaling
recipe there ($X\to X/\sqrt\lambda$, $Y\to Y/\sqrt\lambda$, $\varphi$ unchanged) is
therefore also the recipe for advancing the HJ equation to an arbitrary time $t$, with
no new kernel.

### Calling convention for HJ use

To obtain $u(\cdot,t)$ from initial data $u_0$ sampled on a grid $X$, evaluated on $Y$:

1. pass $\varphi = -u_0|_X$;
2. pass rescaled grids $X/\sqrt t$ and $Y/\sqrt t$;
3. the returned array is $u(\cdot,t)$ on $Y$, in the same index order.

---

## What is exact and what is not

**Exact in time.** Hopf–Lax is a representation formula, not a time discretization.
There is no CFL condition, no time-step error, and no accumulation of temporal error in
a single call — a property that finite-difference HJ solvers do not have and that is the
main structural advantage of building a solver on this primitive.

**Approximate in space.** The kernel minimizes over the finite set $X$, not over
$\mathbb{R}^d$. In the language of `moreau_proximal.md`, it computes the *exact*
Hopf–Lax solution for initial data extended by $+\infty$ off the grid:

$$
u_0^{\text{grid}}(x) = \begin{cases} -\varphi(x) & x\in X\\ +\infty & x\notin X\end{cases}
$$

which is not the same function as a smooth $u_0$ one intended to sample. The resulting
error is a spatial discretization error governed by grid resolution — second order in the
grid spacing near a nondegenerate interior minimizer (the objective is locally quadratic
there), degrading where the minimizer is at a kink or on a domain boundary.

**Semigroup caveat — worth testing before relying on it.** The continuum solution
operator satisfies the semigroup property
$u(\cdot,t+s) = \text{HopfLax}_s\big[u(\cdot,t)\big]$, so iterating this kernel is a
legitimate way to time-step. But the *grid-restricted* operator does **not** inherit this
exactly: each application re-restricts to $X$, so $N$ steps of size $t/N$ and one step of
size $t$ generally disagree. Whether that discrepancy accumulates or is damped over many
steps is an open question for any iterative solver built on this library, and should be
measured rather than assumed. (One step of the full $t$ is exact-in-time and therefore
preferable whenever the problem permits it.)

---

## Analytic test cases

Closed-form solutions of $\partial_t u + \tfrac12\|\nabla u\|^2 = 0$, usable as
Tier-1 analytic checks in the sense of
[`project_brief.md`](../project_brief.md#correctness-criteria). Each is stated as
$(u_0, u(\cdot,t))$; convert to kernel inputs via the calling convention above.

### 1. Quadratic data — smooth, tests discretization error alone

$$
u_0(x) = \frac{a\|x\|^2}{2}
\qquad\Longrightarrow\qquad
u(y,t) = \frac{a\|y\|^2}{2(1+at)}, \qquad 1+at>0 .
$$

The minimizer is $x^*(y) = y/(1+at)$, unique and interior. No kinks, no ties, so any
deviation is attributable purely to grid resolution — making this the right first test
and a usable convergence-rate probe (halve the spacing, expect ~4× error reduction).

### 2. Absolute-value data — the Huber function, tests kink handling

$$
u_0(x) = |x|
\qquad\Longrightarrow\qquad
u(y,t) = \begin{cases}
\dfrac{y^2}{2t}, & |y| \le t,\\[2ex]
|y| - \dfrac{t}{2}, & |y| > t .
\end{cases}
$$

This is exactly the Huber function, and the minimizer is the soft-thresholding operator
$x^*(y)=\operatorname{sign}(y)\max(|y|-t,\,0)$ — the standard prox of $t|\cdot|$. It
exercises nonsmooth initial data and the $|y|=t$ transition where the minimizer detaches
from the origin. Note the solution is smooth ($C^1$) even though $u_0$ is not: the
Hopf–Lax operator regularizes.

### 3. Negative absolute value — tests systematic ties

$$
u_0(x) = -|x|
\qquad\Longrightarrow\qquad
u(y,t) = -|y| - \frac{t}{2}
\quad\text{for all } y,\ t>0,
$$

with minimizers $x^*(y) = y + t\operatorname{sign}(y)$ for $y\ne0$, and **two distinct
minimizers $x^*=\pm t$ at $y=0$**. This is the cleanest available exercise of the
argmin non-uniqueness discussed in
[`moreau_proximal.md`](moreau_proximal.md#why-the-classical-smoothnessuniqueness-guarantees-do-not-transfer)
and blocking the argmin/index-output item in
[`todo.md`](../engineering/todo.md): the *value* is unambiguous and any correct kernel
must reproduce it, while the *index* depends entirely on an as-yet-unspecified
tie-breaking rule. A value-only test passes today; an argmin test on this case is
precisely the one that would force the tie-breaking convention to be pinned down.

Note that the solution has a concave kink at $y=0$ that persists for all $t$ — admissible
for a viscosity solution of a convex Hamiltonian, and a good check that nothing in the
pipeline smooths it away.

### 4. Indicator data — the squared distance transform

$$
u_0 = \begin{cases}0 & x\in S\\ +\infty & \text{otherwise}\end{cases}
\qquad\Longrightarrow\qquad
u(y,t) = \frac{d(y,S)^2}{2t}
$$

Not covered by the standard viscosity-solution theory quoted above (the data is not
real-valued), but it is exactly what the grid-restricted kernel computes when $\varphi$
is constant, and it is the **squared Euclidean distance transform** — the object the
Felzenszwalb–Huttenlocher item in [`todo.md`](../engineering/todo.md) would compute in
$O(n)$ per line. Useful as a cross-check between the two algorithms once FH exists.

---

## Where the correspondence stops

The extensions most likely to be wanted next — obstacles, domain boundaries, mean-field
games, control problems — leave the regime where the above holds. Two distinct
boundaries, in increasing order of severity.

### Forbidding states is free; forbidding paths is not

These are routinely conflated and only one is hard.

**(a) State constraints.** "Mass/value may not live in region $O$." Expressible today at
zero cost: set $\varphi = -\infty$ on $O$ (equivalently $u_0=+\infty$ there), which is
already the $+\infty$-extension used throughout `moreau_proximal.md`. No kernel change.

**(b) Path constraints.** "Trajectories may not cross $O$." This is a genuinely different
problem. The Hopf–Lax minimizer implicitly travels the straight segment from $x$ to $y$;
if that segment is blocked, the formula is invalid. The correct cost becomes geodesic
within the admissible domain $\Omega$:

$$
c(x,y) = \frac{d_\Omega(x,y)^2}{2t}.
$$

**Consequence for this library, and it is a significant one:** both of its fast paths
assume the cost is *additively separable across axes*. That hypothesis is what the
separable kernel exploits ([`separable_ctransform.md`](separable_ctransform.md)) and what
Felzenszwalb–Huttenlocher requires. A geodesic distance around an obstacle is **not**
separable. Under path constraints, both accelerations are unavailable and the naive
$O(N\cdot M)$ kernel is the only member of this library that still applies. Computing
$d_\Omega$ itself requires a different algorithm family altogether — fast sweeping for
the eikonal equation (GPU-tractable), fast marching (heap-based, inherently sequential),
or a semi-Lagrangian scheme.

### Position-dependent Hamiltonians defeat the formula entirely

Hopf–Lax requires $H = H(\nabla u)$, with **no explicit $x$-dependence**. Position-varying
running cost $L(x,v)$, state-dependent control sets, and path constraints as in (b) all
produce $H = H(x,\nabla u)$, for which no representation formula exists in general. The
value function must instead be computed by dynamic programming — semi-Lagrangian schemes,
or monotone finite differences.

Worth noting that this library's kernel *is* already a semi-Lagrangian step, differing
from the textbook version only in performing a global search over all of $X$ rather than
over a local stencil. That is the natural bridge into the semi-Lagrangian literature if
this direction is pursued.

Second-order (viscous / stochastic) HJB equations, carrying a
$\tfrac12\operatorname{tr}(\sigma\sigma^\top D^2u)$ term, admit **no** Hopf–Lax
representation at all and require entirely separate machinery. If that becomes the
target, the c-transform ceases to be the central primitive.

---

## Scope note

This page is explanatory: it documents what the existing kernel already is, in a second
mathematical language, and supplies analytic tests exploitable today. **Building an HJ /
HJB / mean-field-game solver is out of scope for this repository**, which remains a
primitive library ([`project_brief.md`](../project_brief.md)). The value of recording the
identity here is that a downstream solver need not rediscover it, and that the
"where it stops" section above states in advance which extensions preserve this library's
performance characteristics and which discard them.

---

## References

- Evans, *Partial Differential Equations*, §3.3 (Hopf–Lax) and Ch. 10 (viscosity
  solutions) — the shortest path to the formula above.
- Bardi & Capuzzo-Dolcetta, *Optimal Control and Viscosity Solutions of
  Hamilton–Jacobi–Bellman Equations* — reference work for the $x$-dependent case.
- Falcone & Ferretti, *Semi-Lagrangian Approximation Schemes for Linear and
  Hamilton–Jacobi Equations* — the most directly implementable treatment of the
  dynamic-programming route.
- Sethian, *Level Set Methods and Fast Marching Methods* — eikonal solvers, needed for
  $d_\Omega$ under path constraints.
