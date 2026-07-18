# Python extension plan

> **Note**: The filename `python_extention_plan.md` contains a typo ("extention" → "extension"). The file is kept at its current path to avoid breaking references; rename it once a canonical rename is agreed upon and [`docs/index.md`](../index.md) is updated.

---

## Priority and prerequisites

1. C++ reference implementation is complete and tested (CPU sequential loop, verified analytically) — **done**
2. CUDA kernels are verified against the C++ reference (Tier 2 tests passing) — **done**
3. Numerical test suite passes (Tiers 1–4 from [`test_strategy.md`](test_strategy.md)) — **done**
4. A stable C++ API is defined — **done for the value-only kernels**: all three
   (`quadraticCTransform1D`, `quadraticCTransform2D`, `quadraticCTransform2DSeparable`)
   now have a device-pointer `_launch` layer (see
   [`api.md`](api.md#gpu-launch-layer--1d-and-2d-naive)), which is the layer the JAX FFI
   path below binds against. The argmin/index output (`todo.md`) is still an open
   question but doesn't block binding the value-only API — it would be an additive
   binding change later, not a breaking one.

All four prerequisites are met — **this is now the active next task**, not deferred
future work. Two Python entry points are planned, in build order:

1. **NumPy/pybind11-or-nanobind binding** (this doc) — wraps the host convenience
   wrappers. Build this first: lower-stakes, validates the CMake→Python packaging and
   template-instantiation surface before adding XLA complexity.
2. **JAX FFI target** (see [`jax_ffi_integration.md`](jax_ffi_integration.md)) — wraps
   the `_launch` functions directly as a registered XLA custom call, zero host
   round-trip. This is the one the actual downstream consumer (a JAX-based
   back-and-forth solver using `lax.while_loop`) needs — the NumPy binding's per-call
   host↔device copy would defeat the purpose of an iterative on-device solver.

---

## Intended Python API

### 1D

```python
import ctransform_cuda as ct

psi = ct.ctransform_1d(X, Y, phi)
# X:   np.ndarray, shape (N,), dtype float64
# Y:   np.ndarray, shape (M,), dtype float64
# phi: np.ndarray, shape (N,), dtype float64
# returns: psi, np.ndarray shape (M,), dtype float64
```

### 2D

```python
psi = ct.ctransform_2d(Xaxis0, Xaxis1, Yaxis0, Yaxis1, phi)
# Xaxis0: np.ndarray shape (nx0,), float64
# Xaxis1: np.ndarray shape (nx1,), float64
# Yaxis0: np.ndarray shape (ny0,), float64
# Yaxis1: np.ndarray shape (ny1,), float64
# phi:    np.ndarray shape (nx0, nx1), float64, C-contiguous
# returns: psi, np.ndarray shape (ny0, ny1), float64
```

A cost parameter (`cost="sq_euclidean"`) may be added when additional cost functions are supported.

---

## Array ownership and memory flow

The initial design performs explicit host↔device copies inside the binding:

```
Python ndarray (host)
  → cudaMemcpy H2D
  → DeviceBuffer (GPU)
  → kernel
  → DeviceBuffer (GPU)
  → cudaMemcpy D2H
  → Python ndarray (host)
```

Zero-copy (pinned host memory / `cudaMallocHost`) is a possible optimization but is not planned for the first version. The caller owns both input and output arrays.

---

## dtype semantics

| dtype | Status | Notes |
|---|---|---|
| float64 (double) | Default | Matches C++ template T=double |
| float32 (float) | Future | Requires validated T=float tests first |

The binding should raise `TypeError` if non-float64 arrays are passed (until float32 is validated).

---

## Shape and input validation

The binding should validate:

| Check | Exception |
|---|---|
| X, Y, phi have correct number of dimensions | `ValueError` |
| phi.shape == (N,) for 1D, (nx0, nx1) for 2D | `ValueError` |
| Arrays are C-contiguous | `ValueError` or auto-copy to contiguous |
| Any array is empty (N=0 or M=0) | `ValueError` |
| Arrays contain NaN or Inf | Optionally warn; kernel behavior is defined but may be surprising |

---

## Error handling

| Event | Python exception |
|---|---|
| Mismatched array shapes | `ValueError` |
| Non-contiguous input | `ValueError` with suggestion to call `.copy()` |
| CUDA allocation failure | `RuntimeError("cudaMalloc failed")` |
| CUDA kernel error | `RuntimeError("CUDA kernel error: <string>")` |
| CUDA sync error | `RuntimeError` |

Avoid calling `exit(1)` in binding code (the current C++ macro does this); wrap it in try/catch or replace with exception-based error propagation.

---

## Binding library choice

Decision deferred. Two candidates:

| Library | Pros | Cons |
|---|---|---|
| `pybind11` | Mature, widely used, good CMake support | Heavier, compile-time overhead |
| `nanobind` | Faster compile, smaller binary, modern API | Newer, less ecosystem coverage |

CMake integration:
- pybind11: `pybind11_add_module(ctransform_cuda src/python_bindings.cpp)`
- nanobind: `nanobind_add_module(ctransform_cuda src/python_bindings.cpp)`

Both require the CUDA device code to be compiled into a static library first, with the binding `.cpp` file linking against it.

---

## Packaging

Planned approach: `scikit-build-core` with `pyproject.toml` for pip-installable package. The GPU dependency means the wheel is platform- and CUDA-version-specific; no general PyPI upload is planned.

Directory layout (future), now including the JAX FFI module alongside the NumPy binding
(see [`jax_ffi_integration.md`](jax_ffi_integration.md)):
```
python/
├── ctransform_cuda/
│   ├── __init__.py
│   └── jax.py            # registers the FFI targets, exposes ctransform_1d/2d as
│                          # jax.ffi.ffi_call-wrapped functions
└── src/
    ├── bindings.cpp       # pybind11/nanobind module (NumPy binding)
    └── ffi_handlers.cpp   # extern "C" XLA FFI handlers, one per (dim, variant, dtype)
pyproject.toml
```

---

## Python test strategy

Python tests use `pytest` and compare against a pure NumPy reference:

```python
def ctransform_ref_1d(X, Y, phi):
    return np.array([
        np.min(0.5 * (X - y)**2 - phi)
        for y in Y
    ])

def test_1d_matches_reference():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, 100)
    Y = rng.uniform(0, 1, 80)
    phi = rng.uniform(-1, 1, 100)
    psi_cuda = ct.ctransform_1d(X, Y, phi)
    psi_ref  = ctransform_ref_1d(X, Y, phi)
    np.testing.assert_allclose(psi_cuda, psi_ref, atol=1e-12)
```

All Tier-1 through Tier-3 C++ tests should have Python equivalents once the binding exists.
