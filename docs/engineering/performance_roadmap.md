# Performance roadmap

This document lists **concrete, implementable ways to make the c-transform kernels in this
repository faster**, ordered by (effort → payoff). It is a companion to
[`todo.md`](todo.md): `todo.md` tracks *what* is planned, this file explains *why each item
is expected to help* and *what exactly to change*.

Scope: the three GPU paths — `quadraticCTransform1DKernel`,
`quadraticCTransform2DKernel` (naive joint loop), and the two separable passes
`separablePass1Kernel` / `separablePass2Kernel` — plus the host layers around them. The
mathematics is unchanged throughout: every item below must reproduce

$$
\varphi^c(y) \;=\; \min_{x \in X} \Big\{ \tfrac12\|x-y\|^2 - \varphi(x) \Big\}
$$

on the discrete tensor-product grids described in
[`memory_layout.md`](memory_layout.md), to within the tolerances in
[`test_strategy.md`](test_strategy.md).

The target workload is the one that actually matters: a GPU-resident iterative solver that
calls the c-transform **once or twice per iteration, thousands of iterations, on
moderate grids** ($n$ per axis in the hundreds to low thousands), with no host round-trip.
That shapes the priorities below — per-call fixed overhead and asymptotic complexity
matter at least as much as peak kernel throughput.

---

## Tier 0 — Defects in the current inner loops (small edits, immediate payoff)

These are not redesigns. They are things the current kernels do that cost time for no
mathematical reason.

### 0.1 The literal `0.5` forces every `T=float` instantiation into double arithmetic

Every kernel (and the CPU reference) writes `0.5 * dx * dx`. `0.5` is a **`double`
literal**, so for `T = float` the C++ usual arithmetic conversions promote `dx` to
`double`, evaluate the whole candidate expression in double, and narrow only on the
assignment back to `T`. Nothing in `--use_fast_math` changes this — it is required
language semantics, not an optimization the compiler is free to skip.

Consequence: the float32 path performs FP64 multiplies and subtracts in its innermost
loop. On datacenter parts that is a 2× penalty; on consumer parts with 1/32 or 1/64 FP64
throughput it means **the float32 path can be slower than the float64 path**, which
silently removes single precision as a performance option.

Fix: write the constant as `T(0.5)` / `static_cast<T>(0.5)` (or fold it away entirely —
see §1.1). Applies to `ctransform1D_naive.cu`, `ctransform2D_naive.cu`,
`ctransform2D_separable.cu` (both passes), and `cpu_reference.cpp`. This should be done
before any float32 benchmarking is believed, and it is a prerequisite for the float32
validation item in [`todo.md`](todo.md).

### 0.2 `d1` is recomputed $n_{x_0}$ times in the 2D naive kernel

In `quadraticCTransform2DKernel` the inner statement `d1 = Xaxis1[ix1] - yi1` sits inside
the `ix0` loop, although it depends only on `ix1`. The kernel therefore issues
$n_{x_0} n_{x_1}$ loads of `Xaxis1` and $n_{x_0} n_{x_1}$ subtract/multiply pairs where
$n_{x_1}$ would do. The per-axis penalty term $\tfrac12 d_1^2$ should be computed once per
`ix1` — staged into shared memory per block, since every thread in the block with the same
`threadIdx.x` needs the same values.

(This kernel is superseded by the separable path for production use; the fix is still worth
having so that naive-vs-separable comparisons measure the algorithm and not this.)

### 0.3 Block shape: `blockDim.x = 16` splits every warp across two rows

All three 2D kernels launch `dim3 threads(16, 16)`. A warp is 32 lanes, so each warp spans
`threadIdx.x = 0..15` for **two adjacent `threadIdx.y` values** — i.e. two disjoint
16-element runs of the fast (contiguous) index, in two different rows of the output. The
stores to `out` / `dScratchG` are therefore two half-width segments per warp rather than
one contiguous 32-element run.

Fix: make `blockDim.x` a multiple of 32 (`(32, 8)` or `(64, 4)` at the same 256-thread
occupancy) so a warp covers a contiguous run of the fast index. Combine with a check that
the row stride (`ny1`) keeps rows aligned; padding the scratch/output row stride to a
multiple of 32 elements is a cheap way to guarantee it.

### 0.4 64-bit index arithmetic in the innermost loop

Loop counters are `std::size_t`, and `Grid2D` members are `std::size_t`, so the address
computation `ix0 * grid.nx1 + ix1` is a 64-bit multiply-add per iteration, and the loop
counter itself is a 64-bit increment/compare. On a loop whose useful work is one FMA and
one `min`, that is a substantial fraction of the issued instructions.

Fix: keep `std::size_t` in the public API and for the *base* pointer offset computed once
per thread, but use 32-bit (`int` / `unsigned`) counters and offsets **inside** the loop,
guarded by a documented precondition that each individual axis length fits in 32 bits (a
per-axis limit of $2^{31}$ is not a real restriction here; the total array size can still
exceed it because the base offset stays 64-bit).

### 0.5 Host-layer overheads

- `quadraticCTransform*` (the convenience wrappers) do `cudaMalloc` × 4–6, host↔device
  copies, a kernel launch, and `cudaDeviceSynchronize` per call. They are one-shot
  entry points and must never appear in a solver loop; the `_launch` layer exists exactly
  for that (see [`api.md`](api.md#gpu-launch-layer--2d-separable)). Worth stating
  explicitly in `api.md` as a documented contract, not just an implicit one.
- The wrappers call `cudaDeviceSynchronize()` where `cudaStreamSynchronize(stream)` is
  meant. The former serializes against *all* device work, including unrelated streams —
  a real cost as soon as anything else shares the GPU (e.g. a JAX process).
- `CUDA_CHECK` prints to `stderr` and **continues**. A failed launch or allocation
  therefore produces wrong results and a plausible-looking timing rather than an error.
  Benchmark numbers are not trustworthy until this aborts or propagates.
- Two kernel launches per separable c-transform × thousands of iterations: at a few
  microseconds of launch latency each, this is a floor on iteration time that shows up as
  soon as the grid is small enough for kernels to run in tens of microseconds. **CUDA graph
  capture** of the two-pass sequence (or of the solver's whole inner step) removes most of
  it, and requires no kernel changes — only that the launch layer keeps taking an explicit
  stream and pre-allocated scratch, which it already does.

---

## Tier 1 — Rewriting the inner loop (moderate effort, large constant-factor payoff)

### 1.1 Legendre form: one FMA per candidate instead of sub–mul–mul–sub

Expand the quadratic cost and split by which variable each term depends on:

$$
\tfrac12 (x-y)^2 - \varphi(x) \;=\; \underbrace{\tfrac12 x^2 - \varphi(x)}_{=:\,a(x)} \;-\; x\,y \;+\; \tfrac12 y^2
$$

The $\tfrac12 y^2$ term is constant over the minimization, so

$$
\varphi^c(y) \;=\; \tfrac12 y^2 \;+\; \min_{x\in X}\big\{ a(x) - x\,y \big\}
\;=\; \tfrac12 y^2 \;-\; a^{*}(y),
$$

with $a^{*}$ the discrete **Legendre–Fenchel transform** of $a$ (see
[`dual_potentials.md`](../math/dual_potentials.md) and
[`moreau_proximal.md`](../math/moreau_proximal.md) for the surrounding theory).

Two consequences, both practical:

1. **Cheaper inner loop.** Precompute $a(x) = \tfrac12 x^2 - \varphi(x)$ once (an
   $O(n_x)$ elementwise kernel, or fused into whatever produces $\varphi$). The inner
   loop body becomes a single fused multiply-add plus a `min`: `best = min(best, fma(-x[ix], y, a[ix]))`.
   That is roughly a 2–3× reduction in issued arithmetic per candidate, and it removes one
   of the two streaming loads if $a$ replaces $\varphi$ in memory rather than sitting
   beside it. In 2D, $a(x_0,x_1) = \tfrac12(x_0^2 + x_1^2) - \varphi(x_0,x_1)$ and the
   split is per-axis, so the separable passes get the same treatment independently.
2. **It is the gateway to the $O(n)$ algorithms** in Tier 3 — the lower-envelope and
   monotone-minima methods are naturally stated on $\min_x\{a(x) - xy\}$, not on the
   $(x-y)^2$ form.

**Numerical caveat, must be documented alongside it:** the Legendre form subtracts two
potentially large, nearly equal quantities ($a(x)$ and $xy$ both grow like the square of
the coordinate magnitude), where the $(x-y)^2$ form differences the coordinates *first*
and squares a small number. For grids far from the origin the Legendre form loses
significant digits — exactly the cancellation failure mode described in
[`numerical_stability.md`](../math/numerical_stability.md#cancellation-with-large-inputs).
Mitigation is cheap and should be mandatory if this form is adopted: **center the grids**
(subtract a common offset $\bar{x}$ from both $X$ and $Y$ axes before the transform; the
transform is equivariant under a common translation, so the result needs no correction
beyond the bookkeeping of the shift). With centered coordinates the two forms are
comparable in accuracy. Without centering, the Legendre form should not be used in
float32.

### 1.2 Thread coarsening / register blocking over the $y$ index

Every kernel currently computes **one output per thread** and streams the entire $x$-axis
(and, in pass 1, an entire row of $\varphi$) from memory for it. Arithmetic intensity is
therefore ~1 min-plus operation per loaded element, which puts all three kernels firmly on
the memory-bandwidth side of the roofline — the brute-force $O(n^3)$ / $O(n^4)$ work is
*not* what limits them at moderate $n$; the redundant re-reading of the same $\varphi$ and
$x$ values by every $y$-thread is.

Fix: have each thread compute $R$ consecutive outputs (say $R = 4$ or $8$), holding $R$
running minima in registers. Each loaded `a[ix]` / `x[ix]` value is then reused $R$ times,
cutting global traffic for those arrays by $R$ and raising arithmetic intensity to $R$ FMAs
per load. Secondary benefit: $R$ independent `min` accumulators break the single serial
dependency chain on `best`, giving the scheduler instruction-level parallelism it currently
does not have.

For `separablePass1Kernel` this is the single highest-value non-algorithmic change: the
row `dPhi[ix0, :]` is currently re-read once per 16-wide block of $i_{y_1}$, i.e.
$\lceil n_{y_1}/16 \rceil$ times in total (L2 absorbs some of it, but the request count is
real). Coarsening by $R$ divides that count by $R$.

**But note the direction:** coarsening *divides* the total thread count by $R$. That is the
right trade only when the device is already saturated. When it is not — and at
solver-relevant grid sizes it often is not, see §1.5 — coarsening makes the occupancy
problem worse, and the opposite move (splitting each output's reduction across several
threads) is what is called for. The two are complementary knobs on the same tile, not
alternatives; §1.5 states how to choose.

### 1.3 Shared-memory staging of the $x$-axis and the $\varphi$ tile

Complementary to §1.2 and standard: stage a tile of `Xaxis*` (and, in pass 1, the
corresponding tile of `dPhi`) into shared memory, `__syncthreads()`, then loop the tile
from shared. This does not reduce traffic between blocks, but it collapses many redundant
L1 requests within a block into one, and it makes the coarsened loop in §1.2 register- and
issue-efficient rather than load-issue-bound. Tile widths should be a multiple of 32; with
the axis-separated layout there are no bank-conflict hazards for the axis arrays (uniform
broadcast reads) and none for a $\varphi$ tile indexed by the fast axis.

### 1.4 `#pragma unroll` on the reduction loop

With a compile-time-unknown trip count the compiler will not aggressively unroll a loop
whose body ends in a dependent `min`. An explicit `#pragma unroll 4` (or 8), combined with
§1.2's multiple accumulators, is worth measuring; it is a one-line change with a real
effect on issue efficiency. To remove the unknown-trip-count problem at its root rather than
work around it, see §1.6.

### 1.5 Parallelism from the reduction axis: warp- and block-cooperative minima

Every kernel in the repository maps **one thread to one output element** and runs the entire
minimization over the collapsed axis *sequentially inside that thread*. Nothing is staged in
shared memory and no warp-level reduction primitive is used anywhere. Two distinct costs
follow, and neither is addressed by §1.2–§1.4:

1. **The reduction axis contributes no parallelism at all.** Total threads launched equals
   the number of outputs: $n_{x_0} n_{y_1}$ for `separablePass1Kernel`, $n_{y_0} n_{y_1}$
   for `separablePass2Kernel`, $n_y$ for the 1D kernel. The problem's *available*
   parallelism is $\text{outputs} \times n_{\text{reduce}}$ — larger by a factor of the
   reduction length. At the grid sizes an iterative solver actually uses ($n$ in the
   hundreds), the launched thread count lands below the $\sim\!10^5$ mark at which the
   device stops being latency-bound (§2.1), while the unused factor sitting in the reduction
   axis is exactly what would push it over.
2. **The inner loop is a serial dependent chain.** Each iteration's `min` consumes the
   previous value of `best`. The critical path per thread is therefore
   $n_{\text{reduce}}$ back-to-back `min` latencies with no instruction-level parallelism to
   fill them — and with too few resident warps (cost 1) to hide them by multithreading
   either. The two costs compound.

**The fix.** Split each output's reduction across $S$ threads: thread $s$ accumulates a
partial minimum over a strided slice of the collapsed axis, then the $S$ partials are
combined by a tree reduction. Thread count is multiplied by $S$, the serial chain per thread
is divided by $S$, and the combine adds $\log_2 S$ steps. Two shapes, chosen by $S$:

- **Warp-cooperative ($S \le 32$).** One warp per output; lane $l$ takes
  $i_{x_1} = l,\, l{+}32,\, l{+}64, \dots$; combine the 32 lane partials with
  `__shfl_down_sync` in five steps. No shared memory, no `__syncthreads`, no scratch. This
  is the smallest change that tests the hypothesis and is the right first spike.
- **Block-tiled ($S > 32$).** Arrange the block as $(\text{outputs}) \times
  (\text{reduction partitions})$ — e.g. 32 outputs along the fast index by 8 partitions —
  stage the tile of the axis array and of $\varphi$ into shared memory, have each thread
  reduce its partition out of shared, then combine the per-output partials through shared
  memory (or through `__shfl` if the partition index is made the lane index). This form
  delivers §1.3's staging and §1.5's parallelism in one kernel, and it is where §1.2's $R$
  and this section's $S$ meet.

**Choosing $R$ and $S$.** They pull in opposite directions on thread count: §1.2's
coarsening gives $\text{outputs}/R$ threads, this section's splitting gives
$\text{outputs}\cdot S$, and the general tiled kernel launches $\text{outputs}\cdot S/R$.
$R > 1$ is right when the device is saturated and the limiter is redundant global traffic
(large $n$); $S > 1$ is right when it is not and the limiter is occupancy plus chain latency
(moderate $n$). Neither should be a compile-time constant baked into the kernel — pick both
at launch from the problem size (§2.4).

**What each pass actually needs** — the two separable passes are not symmetric, and the
difference is visible in their indexing:

- **Pass 1** indexes `dPhi[ix0 * nx1 + ix1]` and `dXaxis1[ix1]` with `ix1` stepping
  uniformly across the whole block. Consecutive threads share `ix0` and read the *same*
  element at each step, so these are **broadcast** reads: traffic is already low and the
  limiter is issue rate and dependent-chain latency, not bandwidth. That is precisely what
  §1.5 attacks and what §1.2 alone does not. A secondary benefit: once the reduction is
  split, the $\varphi$-row access becomes a coalesced run across the partition threads
  rather than a broadcast.
- **Pass 2** indexes `dScratchG[ix0 * ny1 + iy1]`, which *is* coalesced across
  `threadIdx.x`, and re-reads each column once per block-row of $i_{y_0}$. That redundancy
  is real traffic and is what §1.2's coarsening removes. Pass 2 therefore wants both knobs;
  pass 1 wants mainly this one.

**Validation — cheaper than it looks.** Unlike a sum, a `min` reduction over non-NaN values
is *exactly* order-independent in floating point: it selects an input rather than rounding a
computed value, so no reassociation error exists to accumulate. A tree-reduced kernel should
therefore agree with the sequential one **bit for bit**, not merely to tolerance. The
exceptions are NaN propagation (CUDA's `min` on doubles follows `fmin`, which returns the
non-NaN operand, so a NaN candidate makes order matter) and $\pm 0$ ties. Assert the
bit-exactness in the separable-vs-naive comparison rather than assuming it, and make sure
every partial accumulator — not just the first — is initialized to `INFINITY`.

### 1.6 Shape specialization: make the reduction length a compile-time constant

The trip count of every reduction loop is a runtime value (`grid.nx1`, `grid.nx0`, members
of `Grid1D`/`Grid2D`). The compiler consequently cannot unroll by a known factor, cannot
eliminate the loop-bound test, and cannot prove the absence of a tail. §1.4's
`#pragma unroll 4` recovers part of this; the remainder needs the length known at compile
time.

Direct approach: template the kernel on the reduction length — and, once §1.2/§1.5 land, on
$R$ and $S$ as well — instantiate a bounded set of lengths (powers of two from 64 to 1024
covers the solver-relevant range), and dispatch on an exact match in the **`_launch`
layer**, falling back to the existing runtime-length kernel for everything else. Putting the
dispatch in `_launch` means the host convenience wrappers and the FFI handlers inherit it
without changes of their own. The cost is compile time and binary size, both bounded by the
size of the instantiation set and both easy to cap.

Adjacent, near-free: mark the specialized kernels with `__launch_bounds__` so the register
allocator sizes the budget for the block shape actually launched rather than for the worst
case it must otherwise assume.

---

## Tier 2 — Occupancy and per-call overhead at solver-relevant sizes

### 2.1 Batching

At the grid sizes an iterative solver actually uses, a single c-transform may not fill the
GPU: pass 1 launches $n_{x_0} \times n_{y_1}$ threads, and once that drops below roughly
$10^5$ the device is latency-bound, not throughput-bound. Two structural fixes:

- **Batch dimension in the API.** Accept a leading batch index $b$ over independent
  $\varphi$'s (shared grids, stacked potentials) and fold it into the grid launch. Costs
  nothing when $B = 1$ and turns a latency-bound launch into a throughput-bound one when
  the caller has several transforms to do.
- **Concurrent streams.** A back-and-forth-style solver computes a c-transform for each of
  two potentials per iteration. If they are independent within the iteration they should
  be issued on two streams (or as one batched call), not serialized on the default stream.
  Note the current convenience wrappers hard-code `stream = 0`; the `_launch` layer does
  not, which is the right split.

**The concrete path, end to end.** Today there is no batch anywhere in the stack, and the
Python layer papers over that with a fallback that costs exactly what the batching was
meant to save. Three changes, in dependency order:

1. **Launch layer.** Add a leading batch extent $B$ — either a field on `Grid1D`/`Grid2D` or
   a separate parameter on the `*_launch` functions — under the convention that the
   coordinate axes are *shared* across the batch and only the potential, the output, and the
   scratch carry it: $\varphi : (B, n_{x_0}, n_{x_1})$, out $: (B, n_{y_0}, n_{y_1})$,
   scratch $: (B, n_{x_0}, n_{y_1})$. Map $B$ onto `gridDim.z` with `blockDim.z = 1` and
   offset those three base pointers by `blockIdx.z`. The inner loop is untouched, the axis
   loads stay shared across the batch, and $B = 1$ compiles to the same work as today.
2. **FFI handlers** (`python/src/ffi_handlers.cpp`). Read $B$ from `phi.dimensions()[0]`
   when $\varphi$ arrives with the extra rank, size the `ScratchAllocator` request as
   $B \cdot n_{x_0} n_{y_1} \cdot \texttt{sizeof(T)}$ instead of $n_{x_0} n_{y_1}
   \cdot \texttt{sizeof(T)}$, and pass $B$ through to `_launch`. One custom call and one
   scratch allocation for the whole batch, rather than $B$ of each.
3. **Python wrapper** (`python/ctransform_cuda/jax.py`). All three entry points currently
   pass `vmap_method="sequential"`, which lowers a `jax.vmap` to `lax.map` — $B$ separate
   custom calls and $B$ separate kernel launches, serialized on the stream. Once (1) and (2)
   exist, switch to `vmap_method="broadcast_all"` (the coordinate axes broadcast, $\varphi$
   batched) so a `vmap` becomes a single call into a single batched kernel. The output
   `ShapeDtypeStruct` must then be derived from the batched $\varphi$ shape, not from the
   `Yaxis*` lengths alone.

Why this is the highest-value item in Tier 2 for the intended consumer: a
back-and-forth-style solver transforms two independent potentials per iteration. Under the
current wrapper that is two serialized custom calls — four kernel launches per iteration on
the separable path, each launching a grid too small to fill the device. Batched, it is one
call, two launches, and one grid of twice the size. See
[`jax_ffi_integration.md`](jax_ffi_integration.md) for the FFI-side constraints this must
respect.

### 2.2 Fusing the two separable passes

Pass 2 needs a full column of the intermediate $m(\cdot, y_1)$ across all $i_{x_0}$, so the
passes cannot be fused by a simple block-local rewrite — they need a device-wide barrier.
Two options, in increasing order of ambition: a cooperative-groups grid sync inside one
kernel (removes one launch and keeps the intermediate hot in L2, but constrains grid size
to what fits resident); or leaving them split and letting CUDA graphs (§0.5) absorb the
launch cost. **Try graphs first** — it is a host-side change with no kernel risk.

### 2.3 Keeping the intermediate in L2

The intermediate $m$ has shape $(n_{x_0}, n_{y_1})$. At $n = 512$ and float64 that is 2 MB
— comparable to or smaller than L2 on modern parts. Choosing the **collapse order** to
minimize the intermediate (already noted as a free optimization in
[`separable_ctransform.md`](../math/separable_ctransform.md)) is worth doing not only for
the flop count but because it decides whether the intermediate round-trips to DRAM between
the two passes. Where the API allows it, an L2 persistence hint on the scratch buffer is
a cheap experiment.

### 2.4 Choose the launch configuration from the problem size

Every launch site hardcodes `dim3 threads(16, 16)` (or its 1D equivalent) regardless of $n$,
of which pass is running, and of how many outputs there are. With §0.3 (warp splitting),
§1.2 ($R$ outputs per thread) and §1.5 ($S$ reduction partitions per output) in play, the
block shape stops being a detail and becomes the decision that selects between two different
kernels' worth of behavior.

Put a small, *documented* heuristic in the `_launch` layer:

- compute the total output count for the pass about to be launched (they differ: pass 1 has
  $n_{x_0} n_{y_1}$, pass 2 has $n_{y_0} n_{y_1}$, so a single policy for both is wrong);
- below the saturation threshold, take $S > 1$ and $R = 1$ — buy occupancy and shorten the
  dependent chain;
- above it, take $R > 1$ and $S = 1$ — buy arithmetic intensity and cut redundant traffic;
- in every case keep `blockDim.x` a multiple of 32 so a warp covers a contiguous run of the
  fast index (§0.3).

`cudaOccupancyMaxPotentialBlockSize` gives a defensible starting point for the *total* block
size; how to split that between $R$, $S$, and the output dimension has to be measured, and
the crossover is device-specific. Record the resulting policy in this document, so the
dispatch is a stated design decision rather than folklore embedded in a launch site — the
same treatment §3.1's small-$n$/large-$n$ dispatch needs.

---

## Tier 3 — Algorithmic: $O(n)$ per line instead of $O(n^2)$

Everything above is constant factors on a brute-force minimization. The asymptotics are
where the large-$n$ gap lives: the separable path is $O(n^3)$ for a $n\times n$ problem,
and the exact answer is obtainable in $O(n^2)$ — i.e. **linear work per 1D line**. At
$n = 1024$ that is three orders of magnitude of arithmetic, and no amount of tiling closes
it. Both methods below are exact (not approximations) and both rest on the Legendre form
of §1.1.

### 3.1 Lower envelope of parabolas (Felzenszwalb–Huttenlocher)

$\min_x\{a(x) - xy\}$ over a **sorted** $x$-grid is the evaluation of the lower convex
envelope of the points $(x_i, a_i)$ — equivalently, a lower envelope of parabolas in the
$(x-y)^2$ form. The classical algorithm builds the envelope with a single left-to-right
scan maintaining a stack of currently-visible pieces and their breakpoints (each site is
pushed once and popped at most once → $O(n_x)$), then walks the sorted $y$-grid across the
breakpoints in one merge pass ($O(n_y)$). It requires only that both axes are sorted, not
that they are uniform, and it is exact.

GPU shape of the algorithm — this is the part that needs care:

- Parallelism comes from **lines, not within a line**: pass 1 has $n_{x_0}$ independent
  rows, pass 2 has $n_{y_1}$ independent columns. One thread per line is the simple
  mapping.
- Each thread needs a stack of up to $n_x$ sites and $n_x{+}1$ breakpoints. Placing those
  in local memory gives uncoalesced per-thread arrays; placing them in shared memory caps
  block size hard. The fix that makes this viable is a **transposed (thread-strided)
  scratch layout**: `stack[k * num_lines + line]`, so that at scan step $k$ the whole warp
  touches one contiguous run. Budget the scratch explicitly — it is $O(n_x \times
  \text{lines})$, i.e. the same order as the intermediate buffer, not free.
- Expect divergence: the pop loop's trip count differs per line. It is bounded in
  aggregate (amortized $O(1)$ per site) but locally uneven, so warps will wait on their
  worst line.
- Crossover matters. At small $n$ the brute-force loop — pure FMA+min, no branches, perfect
  coalescing — beats a branchy scan. **Dispatch on $n$**, with the crossover measured, not
  assumed. Keeping the brute-force kernel as the small-$n$ path is a feature, not debt.

Validation for this path is already specified: the $\varphi \equiv \text{const}$ case in
[`hopf_lax_hj.md`](../math/hopf_lax_hj.md#analytic-test-cases) reduces exactly to a squared
distance transform, and the four analytic Hamilton–Jacobi solutions there give a
closed-form check that does not go through the existing brute-force kernels at all.

### 3.2 Monotone (Monge) structure and divide-and-conquer minima

An alternative with a better parallel shape. The candidate matrix

$$
M[i_y][i_x] \;=\; c(x_{i_x}, y_{i_y}) - \varphi(x_{i_x})
$$

has mixed partial $\partial^2 c / \partial x \partial y = -1 < 0$ for the quadratic cost:
$c$ is strictly submodular in $(x,y)$, so $M$ satisfies the inverse-Monge condition and its
**row minimizers are nondecreasing in $i_y$** whenever both axes are sorted. (This holds
for any cost of the form $c = h(x-y)$ with $h$ convex, so it survives the $L_p$
generalization discussed in [`cost_functions.md`](../math/cost_functions.md); it does not
depend on $\varphi$.)

That monotonicity licenses divide-and-conquer over rows: solve the middle $y$, then recurse
on the two halves with the $x$-search range restricted to $[\,\cdot, i_x^*]$ and
$[i_x^*, \cdot\,]$. Total work $O((n_x + n_y)\log n_y)$, and — the point — **depth
$O(\log n_y)$ with every subproblem at a level independent**, so the parallelism is inside
a single line rather than only across lines. That matters exactly when the number of lines
is small (a 1D transform, or a 2D pass on a skewed grid), which is where §3.1's
one-thread-per-line mapping starves. SMAWK reduces this to $O(n_x + n_y)$ but is more
sequential and probably not worth it on a GPU; the $\log$ factor is cheap.

This method also yields the **argmin for free**, which is the blocked prerequisite for the
transport map and for a custom VJP through the JAX FFI (see the argmin item in
[`todo.md`](todo.md)) — with the caveat recorded there that ties must be broken by an
explicit documented rule, since monotone selection and lowest-index selection are not the
same rule.

### 3.3 What to expect

| Path | Work for $n\times n$ 2D | Parallel shape |
|---|---|---|
| Naive joint loop | $O(n^4)$ | one thread per output |
| Separable, brute-force lines | $O(n^3)$ | one thread per output, per pass |
| Separable + lower envelope (§3.1) | $O(n^2)$ | one thread per line |
| Separable + monotone D&C (§3.2) | $O(n^2 \log n)$ | $O(\log n)$ depth, wide at each level |

The two $O(n^2)$-class rows are the only entries that change the asymptotic picture; Tiers
0–2 move the constant in the $O(n^3)$ row, which is the right investment at small $n$ and
the wrong one at large $n$. A production library should ship both and dispatch.

---

## Tier 4 — Precision as a performance lever

Once §0.1 is fixed, `T = float` halves every byte moved (these kernels are bandwidth-bound
before Tier 1 lands) and multiplies arithmetic throughput by 2× to 64× depending on the
part. The blocker is validation, not implementation: float32 is currently compiled but not
numerically validated, and the $10^{-12}$ absolute tolerances in
[`test_strategy.md`](test_strategy.md) are meaningless for it. What is needed:

- a tolerance tier scaled to `float` epsilon and to the *magnitude of the cost*, not an
  absolute constant (the candidate values scale like the square of the domain diameter);
- grid centering (§1.1) as a documented precondition, which bounds the magnitudes being
  differenced;
- the analytic Hamilton–Jacobi cases as the reference, since a float32-vs-float64 GPU
  comparison confounds precision error with reduction-order error.

A mixed strategy is also open: run the envelope/monotone *structure* decisions in float32
(they are comparisons, and a wrong decision near a tie costs accuracy proportional to the
tie gap, i.e. nothing) while accumulating the returned *value* in float64. Worth
considering only after the single-precision path is validated on its own.

---

## Measurement discipline

The benchmark configurations in [`test_strategy.md`](test_strategy.md#performance-benchmarks)
report wall-clock kernel time and effective GFLOP/s. For the items above, GFLOP/s is the
wrong headline number — it rewards the $O(n^4)$ kernel for doing work the $O(n^2)$ one
skips. Recommended additions:

- **Time per output element** at fixed $n$, so algorithms with different work can be
  compared on the same axis.
- **Achieved bandwidth as a fraction of peak** for the brute-force kernels — this is what
  tells you whether Tier 1 has anything left to give, or whether the kernel is already at
  the memory roof and only Tier 3 helps.
- **End-to-end iteration time in the calling solver**, not just kernel time. Tier 0.5's
  launch/sync overheads and Tier 2.1's occupancy problems are invisible in a kernel-only
  measurement and can dominate at solver-relevant grid sizes.
- **The isolated steady-state call, reported separately from the end-to-end number.** Time a
  single transform on a fixed stream after discarding warm-up iterations, and report it
  alongside — not instead of — the end-to-end figure. The two answer different questions:
  the isolated number is what Tiers 1 and 3 move, the end-to-end number additionally
  contains per-call fixed cost (Tier 0.5, Tier 2.1) and any one-time setup on the calling
  side. A single blended number lets a fixed-cost regression hide behind a kernel
  improvement, or the reverse.
- Fixed random seeds and a fixed device clock state (`nvidia-smi --lock-gpu-clocks`) if
  comparisons across commits are meant to be trusted.
