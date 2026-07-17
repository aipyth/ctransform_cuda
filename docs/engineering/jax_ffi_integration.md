# JAX / XLA FFI integration (design notes)

Status: **design only, nothing implemented.** This describes how an external repo (a
JAX-based exact optimal-transport solver, e.g. Jacobs' back-and-forth method driven by
`lax.while_loop`) would register this library's device-pointer launch layer as an XLA
custom call. It depends on the [device-pointer-only launch layer](api.md#architecture-layers)
(`todo.md`), not on this repo's own (still unimplemented) NumPy/pybind11 binding — the
two Python integration paths are independent and serve different callers.

---

## Repo boundary

This is the one design decision that shapes everything below: **the FFI shim does not
live in this repo.**

- `ctransform_cuda` owns the mathematics: kernels + the `_launch` layer (device pointers,
  `cudaStream_t`, no allocation/copy/sync). This is exactly the layer
  [`api.md`](api.md#architecture-layers) already designates as "the function registered
  with the Python/JAX FFI custom call."
- The consuming solver repo owns the JAX-specific glue: the `extern "C"` FFI handler,
  `jax.ffi.register_ffi_target` call, and the `ffi_call` wrapper used inside
  `lax.while_loop`.

Rationale: this project's charter is the c-transform operator, not a solver or a Python
package (see `CLAUDE.md`); XLA's FFI ABI is solver-framework-specific glue that has
nothing to do with the c-transform itself. The dependency direction is solver repo →
this repo, same as any other library consumer — it links against
`libctransform_cuda` (or includes `ctransform.hpp` + object files) and writes its own
handler on top.

**What this repo must still provide** for that shim to be writable at all:

1. The `_launch` functions for 1D and naive 2D (todo item above — currently only the 2D
   separable one exists).
2. Explicit template instantiations for `float` and `double` (already the case, per
   [`api.md`](api.md#supported-types)) — XLA FFI targets are registered per concrete
   signature, so the consumer needs a concrete symbol per dtype, not a template.
3. A stable, documented meaning for every buffer and scratch argument (already true —
   see [`api.md`](api.md#gpu-launch-layer--2d-separable) and
   [`memory_layout.md`](memory_layout.md)) since the FFI handler on the other side maps
   XLA buffers onto these parameters positionally with no type-level help from C++.

Nothing here requires a new function *in this repo* beyond finishing the launch-layer
split; this file exists so the shape of that handler is understood before it's written.

---

## What the consuming repo's shim looks like (prose, not code)

JAX's current FFI mechanism (`jax.ffi`, replacing the older raw XLA custom-call ABI) has
three parts on the C++ side and two on the Python side. This project provides none of
them directly, but the shapes matter because they constrain what "stable API" (todo.md
item 3, blocking the Python wrapper) needs to mean for the launch layer specifically.

**C++ handler** (in the *solver* repo): a non-templated, `extern "C"`-linkage function
matching the XLA FFI handler signature — takes typed `ffi::Buffer<T>` arguments (input
grids, phi, output), reads the CUDA stream via `ffi::PlatformStream<cudaStream_t>` from
the execution context, and calls into this repo's `quadraticCTransform{1D,2D}[Separable]_launch`
with `.data()`/`.typed_data()` pointers pulled from those buffers. One handler per
`(dim, variant, dtype)` combination that the solver needs — XLA does not dispatch on
dtype the way a C++ template does, so `double` and `float` are two distinct registered
targets (e.g. `ctransform_1d_f64`, `ctransform_1d_f32`), both calling the same templated
`_launch` function under the hood.

**Registration** (Python, in the solver repo): `jax.ffi.register_ffi_target(name, capsule,
platform="CUDA")` binds the string name to the compiled handler symbol. This is a
process-global registration done once at import time.

**Call site** (Python, in the solver repo): `jax.ffi.ffi_call(name, result_shape_dtypes,
...)` wraps the target into a JAX-traceable primitive with fixed output shape/dtype,
callable like any other JAX op — including from inside a `lax.while_loop` body, since
it's a normal XLA op in the computation graph rather than a Python-level dispatch.

None of this is implementation work for `ctransform_cuda`; it's listed so the constraints
below are traceable back to *why* they exist.

---

## Constraints this imposes on the launch layer

- **Static shapes.** `lax.while_loop` requires the carry (and therefore every buffer
  shape) to be fixed across iterations — this is a JAX/XLA constraint, not a c-transform
  one, but it means `Grid1D`/`Grid2D` must be compile-time-fixed for a given traced
  program. Growing/shrinking grids inside the loop is not representable; the solver must
  pad or re-trace instead. No action needed here — the launch layer already takes `grid`
  by value with no dynamic resizing.
- **No implicit host sync.** This is exactly what `_launch` already buys — the entire
  point of the split. An XLA custom call that internally synchronized would serialize the
  device timeline against nothing useful and defeat the reason for using FFI in a
  `while_loop` at all.
- **Scratch buffers must be XLA-visible.** The separable path's `dScratchG` cannot be a
  `cudaMalloc` hidden inside the handler — under `lax.while_loop`/`jit`, XLA needs to own
  (and reuse) all device memory for the traced computation. The FFI shim must request
  scratch via JAX's `ffi_call(..., input_output_aliases=...)` or an explicit scratch
  output, sized from `Grid2D` at trace time. This only matters for callers of the
  separable kernel; the naive kernels need no scratch.
- **Error reporting is asynchronous and non-throwing.** FFI handlers return an
  `XLA_FFI_Error*`, not a C++ exception, and a CUDA kernel launch error may not surface
  until a later sync anyway (stream-ordered execution). This repo's existing
  `CUDA_CHECK`-after-launch pattern (checks `cudaGetLastError`, which only reports
  *launch-configuration* errors, not in-kernel faults) is the right foundation for the
  shim's own error path, but the shim is responsible for translating that into the FFI
  error ABI — this repo's macro/exception style doesn't cross the boundary as-is.

---

## Testing implication for this repo

Nothing changes in *this* repo's test suite for FFI support specifically — the contract
the shim depends on is exactly what Tier 1–4 already verify (correctness of the
`_launch` functions against the CPU reference and the naive kernel). The one addition
worth tracking here: once the 1D/2D naive `_launch` functions exist, extend the existing
separable-vs-naive-vs-CPU comparisons ([`test_strategy.md`](test_strategy.md)) to call
the `_launch` entry points directly on pre-staged device buffers (bypassing the host
wrapper's alloc/copy) — that's the actual code path the FFI shim will exercise, and today
only the separable variant has any test coverage that calls `_launch` directly rather
than through its host-wrapper shell. *(Check whether this already exists in
`test_cuda_vs_cpu.cpp`/`test_randomized.cpp` before adding it — not verified as part of
this note.)*

---

## Open questions (not resolved here)

- Whether the solver repo statically links `libctransform_cuda` or dynamically loads it —
  affects how the FFI capsule symbol is exposed, but is entirely a solver-repo build
  concern.
- Whether `float32` is needed for the solver's first version — if so, item "float32
  validation" in [`todo.md`](todo.md) becomes a hard prerequisite, not just a nice-to-have,
  since the FFI target would otherwise have no validated `f32` path to register.
