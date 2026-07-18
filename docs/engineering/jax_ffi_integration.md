# JAX / XLA FFI integration (design notes)

Status: **design only, nothing implemented.** This describes how this repo's `python/`
package registers the device-pointer launch layer as an XLA custom call, so JAX code
(e.g. Jacobs' back-and-forth method driven by `lax.while_loop`) can call the c-transform
with no host round-trip per iteration. It depends on the
[device-pointer-only launch layer](api.md#gpu-launch-layer--1d-and-2d-naive) (done for
all three kernels) and is one of two Python entry points this repo will expose — see
[`python_extention_plan.md`](python_extention_plan.md) for how it relates to the
NumPy/pybind11 binding.

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

**C++ handler** — a non-templated, `extern "C"`-linkage function matching the XLA FFI
handler signature: takes typed `ffi::Buffer<T>` arguments (input grids, phi, output),
reads the CUDA stream via `ffi::PlatformStream<cudaStream_t>` from the execution
context, and calls into `quadraticCTransform{1D,2D}[Separable]_launch` with
`.data()`/`.typed_data()` pointers pulled from those buffers. One handler per `(dim,
variant, dtype)` combination — XLA does not dispatch on dtype the way a C++ template
does, so `double` and `float` are two distinct registered targets (e.g.
`ctransform_1d_f64`, `ctransform_1d_f32`), both calling the same templated `_launch`
function under the hood.

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

Both entry points wrap the same underlying kernels but at different layers, and are
worth building in this order:

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
  `cudaMalloc` hidden inside the handler — under `lax.while_loop`/`jit`, XLA needs to own
  (and reuse) all device memory for the traced computation. The handler must request
  scratch via `ffi_call(..., input_output_aliases=...)` or an explicit scratch output,
  sized from `Grid2D` at trace time. Only matters for the separable kernel; the naive
  kernels need no scratch.
- **Error reporting is asynchronous and non-throwing.** FFI handlers return an
  `XLA_FFI_Error*`, not a C++ exception, and a CUDA kernel launch error may not surface
  until a later sync anyway (stream-ordered execution). This repo's existing
  `CUDA_CHECK`-after-launch pattern (`cudaGetLastError`, which only reports
  *launch-configuration* errors, not in-kernel faults) is the right foundation, but the
  handler is responsible for translating that into the FFI error ABI — the
  macro/exception style doesn't cross the boundary as-is.
- **No automatic differentiation through the custom call**, by default. XLA custom calls
  are opaque to JAX's autodiff unless a custom VJP/JVP rule is also registered — a
  separate, nontrivial piece of work. Not needed if the solver only evaluates the
  c-transform's *value* inside a fixed-point iteration (no gradient through it), which is
  the back-and-forth use case; would become a hard requirement if the solver repo ever
  wants to backprop through a c-transform call (e.g. bilevel optimization, learning a
  potential). Worth confirming which of these the solver actually needs before deciding
  this is out of scope.

---

## Testing implication

`tests/test_launch_layer.cu` already exercises exactly the code path the FFI handler
calls — `_launch` directly on pre-staged device buffers, no host wrapper involved — so
the C++ side of this contract is covered. Once the Python package exists, the Python test
suite ([`python_extention_plan.md`](python_extention_plan.md#python-test-strategy))
should include the FFI path (`ctransform_cuda.jax.ctransform_1d`, called inside a small
`jax.jit`) alongside the NumPy-binding path, comparing both against the same NumPy
reference.

---

## Open questions (not resolved here)

- Whether `float32` is needed for the first JAX-facing version — if so, "float32
  validation" in [`todo.md`](todo.md) becomes a hard prerequisite, since the FFI target
  would otherwise have no validated `f32` path to register. JAX defaults to float32
  unless `jax.config.update("jax_enable_x64", True)` is set, so this will come up
  immediately in practice even if `float64` is the "supported" dtype on the C++ side.
- Whether the back-and-forth solver needs gradients through the c-transform call (see
  the autodiff note above) — determines whether a custom VJP/JVP is in scope.
