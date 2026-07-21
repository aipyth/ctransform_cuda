# JAX / XLA FFI integration

Status: **implemented** for `float64`, all three kernels. This describes how this repo's
`python/` package registers the device-pointer launch layer as an XLA custom call, so JAX
code (e.g. Jacobs' back-and-forth method driven by `lax.while_loop`) can call the
c-transform with no host round-trip per iteration. It builds on the
[device-pointer-only launch layer](api.md#gpu-launch-layer--1d-and-2d-naive) and is one of
two Python entry points this repo exposes — see
[`python_extention_plan.md`](python_extention_plan.md) for how it relates to the
NumPy/pybind11 binding, and [`development.md`](development.md) for how to build it.

| Registered target | Python wrapper | Underlying launch function |
|---|---|---|
| `ctransform_1d_f64` | `ctransform_cuda.jax.ctransform_1d` | `quadraticCTransform1D_launch` |
| `ctransform_2d_f64` | `ctransform_cuda.jax.ctransform_2d` | `quadraticCTransform2D_launch` |
| `ctransform_2d_separable_f64` | `ctransform_cuda.jax.ctransform_2d_separable` | `quadraticCTransform2DSeparable_launch` |

Handlers live in `python/src/ffi_handlers.cpp`; registration and the traceable Python
wrappers in `python/ctransform_cuda/jax.py`. Which of the two 2D targets should become the
default is a benchmarking question, not yet settled — see [Open questions](#open-questions).

**Revision note**: an earlier version of this doc placed the FFI shim in the *consuming*
solver repo, reasoning that XLA glue isn't part of the c-transform's mathematics. That
was reconsidered once it was clear the consuming repo is pure Python (JAX, possibly
PyTorch) with no intention of writing or maintaining C++ — asking it to own an `extern
"C"` handler and a build step for it would mean either writing CUDA build tooling twice
or (worse) hand-copying a header-sensitive handler across repos. Owning the shim here
means the solver repo does a plain `pip install` and gets a ready-to-call JAX primitive;
nothing about the FFI ABI leaks across the repo boundary.

---

## What this repo's FFI shim consists of

JAX's current FFI mechanism (`jax.ffi`, the successor to the older raw XLA custom-call
ABI) has three parts on the C++ side and two on the Python side, all living in this
repo's `python/` package alongside the pybind11/nanobind module:

**C++ handler** — a non-templated function taking typed `ffi::Buffer<F64>` arguments
(input grids, phi) and an `ffi::ResultBuffer<F64>` output, reading the CUDA stream via
`ffi::PlatformStream<cudaStream_t>` from the execution context, and calling
`quadraticCTransform{1D,2D}[Separable]_launch` with `.typed_data()` pointers pulled from
those buffers. `XLA_FFI_DEFINE_HANDLER_SYMBOL` generates the ABI-facing symbol and its
argument-decoding glue. One handler per `(dim, variant, dtype)` combination — XLA does not
dispatch on dtype the way a C++ template does, so `double` and `float` would be two
distinct registered targets (e.g. `ctransform_1d_f64`, `ctransform_1d_f32`), both calling
the same templated `_launch` function under the hood. Only the `f64` targets exist today.

A handler parameter is one of three kinds, and the distinction drives everything below:

| Kind | Bound with | Nature |
|---|---|---|
| argument | `.Arg<ffi::Buffer<F64>>()` | input buffer — data, shape known at trace time |
| result | `.Ret<ffi::Buffer<F64>>()` | output buffer — data, address assigned by XLA before the call |
| context | `.Ctx<T>()` | **not data** — a per-call service from the execution environment |

Arguments and results participate in compilation: they appear in the HLO, get static
buffer assignments, and are subject to aliasing and donation analysis. Contexts exist only
at runtime and are decoded from an opaque `XLA_FFI_ExecutionContext*` via a
`CtxDecoding<T>` specialization. Both context types this repo uses —
`ffi::PlatformStream<cudaStream_t>` and `ffi::ScratchAllocator` — are thin C++ views over
function pointers in the stable `XLA_FFI_Api` C ABI.

> ⚠️ **The `Bind()` chain is positional.** The order of `.Ctx()`/`.Arg()`/`.Ret()` calls
> must match the handler's parameter order exactly; each call appends to a type list
> (`Binding<stage, Ts..., Tag>`). Interleaving a `.Ctx()` after `.Arg()`s is legal as long
> as the handler agrees. Nothing checks the correspondence against the *Python* call site,
> so a permuted argument list compiles cleanly and yields silently wrong numbers. The
> mitigation in the test suite is to use four distinct axis lengths, which turns most
> permutations into a shape error rather than a wrong value.

**Registration** (Python, at import time of this package) — `jax.ffi.register_ffi_target(name,
capsule, platform="CUDA")` binds the string name to the compiled handler symbol. This is
a process-global registration; the consuming repo never sees it, it just happens the
moment the consuming repo does `import ctransform_cuda`.

**Call site** (Python, exposed by this package) — a thin Python function, e.g.
`ctransform_cuda.jax.ctransform_1d(X, Y, phi)`, wrapping `jax.ffi.ffi_call(name,
result_shape_dtypes, ...)` into an ordinary JAX-traceable function with fixed output
shape/dtype. The consuming repo imports this function and calls it like any other JAX
op — including from inside a `lax.while_loop` body — without ever touching `jax.ffi`
directly.

---

## Relationship to the NumPy/pybind11 binding

Both entry points wrap the same underlying kernels but at different layers. They were
built in this order, and the rationale is retained because it explains the layering:

1. **NumPy binding first** (per [`python_extention_plan.md`](python_extention_plan.md)):
   wraps the host convenience wrappers (`quadraticCTransform1D`, etc.), does the
   alloc/copy/sync internally. Lower-stakes to build — it's a fairly mechanical
   pybind11/nanobind exercise with no XLA involved — and it's the thing the Tier-1–4
   Python test suite (mirroring the C++ tests) validates against first. Good enough for
   ad hoc scripts, notebooks, or a non-JAX (e.g. plain PyTorch, via a `.numpy()`/`.cpu()`
   round trip) caller.
2. **JAX FFI target second**, once the NumPy binding proves the CMake→Python packaging
   and template-instantiation plumbing works. Wraps the `_launch` functions directly,
   zero host round-trip, callable inside `jit`/`lax.while_loop`. This is the one the
   back-and-forth solver actually needs for performance — the NumPy binding's per-call
   host↔device copy would defeat the purpose of an iterative on-device solver.

**PyTorch note**: JAX and PyTorch can share GPU memory via DLPack
(`jax.dlpack.from_dlpack`/`to_dlpack`), so a PyTorch caller can reach the JAX FFI path by
converting a `torch.Tensor` to a `jax.Array` first — this composes for free once the JAX
FFI target above exists no new binding needed. If PyTorch ever needs to call the CUDA
kernels *without* going through JAX at all (e.g. inside a `torch.autograd.Function` with
a custom CUDA kernel op), that's a third, separate binding mechanism
(`torch.utils.cpp_extension`, PyTorch's own C++/CUDA extension API) — not scoped here,
and only worth adding if "maybe pytorch" firms up into an actual requirement independent
of JAX.

---

## Constraints this imposes on the launch layer

- **Static shapes.** `lax.while_loop` requires the carry (and therefore every buffer
  shape) to be fixed across iterations — a JAX/XLA constraint, not a c-transform one, but
  it means `Grid1D`/`Grid2D` must be compile-time-fixed for a given traced program.
  Growing/shrinking grids inside the loop isn't representable; the solver must pad or
  re-trace instead. No action needed in the launch layer itself — it already takes
  `grid` by value with no dynamic resizing.
- **No implicit host sync.** This is exactly what `_launch` already buys. An XLA custom
  call that internally synchronized would serialize the device timeline against nothing
  useful and defeat the reason for using FFI in a `while_loop` at all.
- **Scratch buffers must be XLA-visible.** The separable path's `dScratchG` cannot be a
  `cudaMalloc` hidden inside the handler. Two independent reasons: under
  `lax.while_loop`/`jit`, XLA needs to own (and reuse) all device memory for the traced
  computation; and `cudaMalloc` is a *synchronizing* call, so it would stall the stream
  every iteration and defeat the entire point of the non-synchronizing `_launch` layer.
  Only matters for the separable kernel; the naive kernels need no scratch. See
  [Scratch memory](#scratch-memory-the-separable-path) for how this is resolved.
- **Error reporting is asynchronous and non-throwing.** FFI handlers return an
  `XLA_FFI_Error*`, not a C++ exception, and a CUDA kernel launch error may not surface
  until a later sync anyway (stream-ordered execution). This repo's existing
  `CUDA_CHECK`-after-launch pattern (`cudaGetLastError`, which only reports
  *launch-configuration* errors, not in-kernel faults) is the right foundation, but the
  handler is responsible for translating that into the FFI error ABI — the
  macro/exception style doesn't cross the boundary as-is.
- **`vmap` requires an explicit `vmap_method`.** Like autodiff, JAX's batching is opaque
  to a raw custom call: with the default `vmap_method=None`, `jax.vmap` over an `ffi_call`
  raises. The Python wrappers therefore pass `vmap_method="sequential"`, which lowers a
  batch to `lax.map` — *B* sequential kernel launches on the same stream, no host sync.
  This is what lets the consuming solver write `jax.vmap(c_transform)` over its stack of
  measures. `"sequential"` batches over *all* arguments and handles the mixed
  batched/unbatched case (the solver batches only `phi`, closing over the shared coordinate
  axes) by broadcasting internally. If the *B* launches ever dominate, the alternative is
  native batching inside the handler — read *B* from `phi.dimensions()[0]` and issue *B*
  `_launch` calls, keeping it a single async FFI call; deferred until benchmarks demand it.
- **No automatic differentiation through the custom call**, by default. XLA custom calls
  are opaque to JAX's autodiff unless a custom VJP/JVP rule is also registered — a
  separate, nontrivial piece of work. **Confirmed not needed** for the current consuming
  solver (a regularized-barycenter back-and-forth iteration): it computes its ascent
  direction analytically from a Poisson–Neumann solve on the pushforward residual and only
  ever evaluates the c-transform's *value*, never `jax.grad` through it. Would become a
  hard requirement only if the solver is later wrapped in `jax.grad` (e.g. bilevel
  optimization, or learning `weights`/`regularization_strength`) — at which point the
  argmin/tie-breaking question in [`todo.md`](todo.md) is the blocker.

---

## Scratch memory: the separable path

The separable algorithm needs an intermediate buffer `dScratchG` of `nx0 * ny1` elements
between its two passes (both index it as `ix0 * ny1 + iy1`). An earlier revision of this
doc proposed obtaining it through `input_output_aliases` or an extra declared output.
Neither is needed: XLA FFI provides a purpose-built context type.

Binding `.Ctx<ffi::ScratchAllocator>()` hands the handler an object with a single method,
`Allocate(size_bytes, alignment) -> std::optional<void*>`, drawing from XLA's own device
memory pool. It is RAII: the destructor frees every allocation when the handler returns,
and there is no `Free` to call. Three things the handler must get right:

- **Size is in bytes**, so `nx0 * ny1 * sizeof(T)`, not an element count.
- **Pass an explicit alignment.** The parameter defaults to `1` ("any address"). A
  misaligned 64-bit global access is not merely slow on CUDA — it faults and kills the
  kernel. XLA's allocator returns generously aligned blocks in practice, so `alignof(T)`
  is correctness by contract rather than by luck.
- **Check the `optional`.** Allocation can fail under memory pressure; this is the only
  runtime failure path in any of the three handlers. Translate it to `ffi::Error::Internal`
  rather than dereferencing.

Chief advantage over the extra-output approach: the scratch never appears in the jaxpr, so
the Python wrapper for the separable target is signature-identical to the naive one — no
dummy result to construct and discard at the call site.

> ⚠️ **Unverified: scratch lifetime vs. asynchronous kernels.** The `ScratchAllocator`
> destructor frees at handler *return*, but the kernels are launched asynchronously on the
> stream and may not have executed yet. Whether this is safe depends on XLA's allocator
> being stream-ordered on this path, which has not been confirmed against XLA's source.
>
> The reason to believe it is safe: this is the same mechanism XLA uses for cuDNN and
> cuBLAS workspaces inside its own GPU custom calls, which are equally asynchronous — were
> it unsafe, every convolution in XLA would be broken.
>
> **`compute-sanitizer` cannot settle this**, and a clean run should not be read as having
> done so. XLA reserves a large device slab at startup and sub-allocates from its own pool,
> so a pool-internal free-and-reuse never reaches the driver-level allocation events the
> sanitizer observes. (Sanitizer runs *are* meaningful for the separable path's bounds
> checking — see below — just not for this question.)
>
> Expected symptom if it is in fact unsafe: intermittent, non-reproducible wrong values
> from the separable target specifically, under sustained load. Resolving it properly means
> either reading XLA's allocator source, or falling back to declaring scratch as a second
> `ffi::ResultBuffer` whose lifetime XLA manages through ordinary buffer assignment.

---

## Testing implication

`tests/test_launch_layer.cu` exercises exactly the code path the FFI handler calls —
`_launch` directly on pre-staged device buffers, no host wrapper involved — so the C++ side
of this contract is covered independently of Python.

The Python-side coverage is `python/tests/test_jax_ffi.py`, eight tests against a NumPy
reference at `atol=1e-12`. Four properties it checks that are specific to the FFI layer
rather than to the mathematics:

- **Eager and under `jit`, separately.** Outside `jit` JAX dispatches eagerly; inside, XLA
  performs buffer assignment and may fuse neighbours. Different paths through the handler.
- **Separable against naive**, not only against the reference. The two kernels reassociate
  the arithmetic — naive computes `(½d₀² + ½d₁²) − φ`, separable computes
  `½d₀² + (½d₁² − φ)`. `min` itself is exact in floating point, so the entire discrepancy
  comes from that regrouping, on the order of an ulp. Larger grids make near-ties more
  likely and so make this test bite harder.
- **`nx0 > ny0` for the separable target.** This makes the scratch buffer (`nx0*ny1`)
  larger than the output (`ny0*ny1`), so a regression in the `dOut`/`dScratchG` argument
  order — both `T*`, hence invisible to the compiler — becomes an out-of-bounds write
  rather than merely wrong values.
- **Explicit output-shape assertions**, since `assert_allclose` broadcasts and would let
  some shape errors pass.

Bounds correctness of the separable path has been verified under
`compute-sanitizer --tool memcheck` (0 errors). That result is meaningful for out-of-bounds
access, which the tool detects deterministically; it says nothing about the scratch
lifetime question above.

---

## Open questions

- **Which 2D target should be the default.** Both `ctransform_2d` (naive) and
  `ctransform_2d_separable` are registered and validated against each other. The separable
  path is asymptotically better by a factor of `nx1·ny0/(nx1+ny0)` — roughly `n/2` on
  square `n×n` grids — but the crossover point through the FFI, where per-call overhead and
  XLA buffer assignment also count, has not been measured. Benchmark at the consuming
  solver's real grid size before collapsing the two names into one.
- **Whether the scratch lifetime is genuinely safe** — see the warning under
  [Scratch memory](#scratch-memory-the-separable-path).
- Whether `float32` is needed for the first JAX-facing version — if so, "float32
  validation" in [`todo.md`](todo.md) becomes a hard prerequisite, since the FFI target
  would otherwise have no validated `f32` path to register. JAX defaults to float32
  unless `jax.config.update("jax_enable_x64", True)` is set, so this will come up
  immediately in practice even if `float64` is the "supported" dtype on the C++ side.
- Whether the back-and-forth solver needs gradients through the c-transform call (see
  the autodiff note above) — determines whether a custom VJP/JVP is in scope.
