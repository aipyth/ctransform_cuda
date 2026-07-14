# Library reference (API)

This is the single reference for the library's public interface: the **naming
notation**, the **layered architecture**, and the **catalog** of available and planned
functions with their parameters and a short description of each. All public declarations
live in [`include/ctransform.hpp`](../../include/ctransform.hpp).

For the mathematics of the c-transform see [`../math/ctransform_overview.md`](../math/ctransform_overview.md);
for kernel-level implementation detail see
[`cuda_cpp_architecture.md`](cuda_cpp_architecture.md) and the per-algorithm design notes
(e.g. [`separable_kernel_spec.md`](separable_kernel_spec.md)).

---

## Architecture: layers

The library is organized in layers, from lowest (device compute) to highest
(host / Python):

1. **Compute kernels** (`__global__`, internal). The actual GPU computation. Not part of
   the public API; named with a `Kernel` suffix.
2. **Device-pointer launch layer** (`…_launch`). A host-callable function that launches
   the kernel(s) on data **already resident on the device**, taking a `cudaStream_t`. It
   does **no** allocation, **no** host↔device copy, and **no** synchronization — the
   caller owns all three. This is the layer a GPU-resident iterative solver (e.g. a
   Back-and-Forth optimal-transport solver) calls every iteration without a host
   round-trip, and it is the function registered with the Python/JAX FFI custom call.
3. **Host convenience wrappers**. Allocate device buffers, copy inputs host→device, call
   the launch layer on the default stream, synchronize, and copy the result back. These
   are the functions the NumPy-style Python binding wraps. Every current GPU entry point
   is a host wrapper of this kind.
4. **CPU reference** (`…CPU…`). Serial, single-threaded implementations used to validate
   the GPU path in tests. Not intended for performance.

**Design rule.** The mathematics of a given transform lives in exactly one place — its
kernels plus its launch layer. A host convenience wrapper is only marshalling
(allocate / copy / synchronize) and must not duplicate compute logic. Splitting each host
wrapper into a launch function plus a thin convenience wrapper is tracked in
[`todo.md`](todo.md).

Data is represented as **axis-separated tensor-product grids** in **row-major** order; see
[`memory_layout.md`](memory_layout.md). Full 2D coordinate pairs are never materialised.

---

## Naming notation

Public function names follow a fixed grammar, so a name fully determines what the function
does:

```
<cost> CTransform [CPU] <dim>D [<Variant>] [_launch]
```

| Component | Meaning | Examples |
|---|---|---|
| `<cost>` | cost-function family (lowercase prefix) | `quadratic` = ½‖x−y‖² |
| `CTransform` | the c-transform operator (fixed) | — |
| `[CPU]` | present → serial CPU reference; absent → GPU | `…CPU2D` vs `…2D` |
| `<dim>D` | domain dimensionality | `1D`, `2D` (`3D` planned) |
| `[<Variant>]` | algorithm variant; absent → the default naive brute-force kernel | `Separable` (`Envelope`/FH planned) |
| `[_launch]` | present → device-pointer/stream launch layer; absent → host convenience wrapper (or a CPU function) | `…Separable_launch` |

Examples:

- `quadraticCTransform2D` — GPU, 2D, naive, host wrapper
- `quadraticCTransformCPU2D` — CPU reference, 2D
- `quadraticCTransform2DSeparable` — GPU, 2D, separable variant, host wrapper
- `quadraticCTransform2DSeparable_launch` — GPU, 2D, separable, device-pointer launch layer

Internal `__global__` kernels are named descriptively with a `Kernel` suffix
(`quadraticCTransform2DKernel`, `separablePass1Kernel`) and are not part of the public
API.

---

## Grid structs

```cpp
struct Grid1D {
    std::size_t nx;   // number of source points
    std::size_t ny;   // number of target points
};

struct Grid2D {
    std::size_t nx0, nx1;   // source axis sizes
    std::size_t ny0, ny1;   // target axis sizes
};
```

---

## Function catalog

Status legend: **✓ implemented** · **◐ in progress** · **○ planned**.

| Function | Layer | Dim | Status |
|---|---|---|---|
| `quadraticCTransformCPU1D` | CPU reference | 1D | ✓ |
| `quadraticCTransformCPU2D` | CPU reference | 2D | ✓ |
| `quadraticCTransform1D` | host wrapper | 1D | ✓ |
| `quadraticCTransform2D` | host wrapper | 2D | ✓ |
| `quadraticCTransform2DSeparable` | host wrapper | 2D | ✓ |
| `quadraticCTransform2DSeparable_launch` | launch layer | 2D | ✓ |
| `quadraticCTransform1D_launch`, `quadraticCTransform2D_launch` | launch layer | 1D/2D | ○ |
| `quadraticCTransform3D…` | kernels + wrappers | 3D | ○ |
| `quadraticCTransform2DEnvelope` (Felzenszwalb–Huttenlocher O(n)/line) | variant | 2D | ○ |
| Python bindings | binding | — | ○ |

Planned items and their rationale are tracked in [`todo.md`](todo.md).

### CPU reference — 1D and 2D

Serial nested-loop implementations for correctness verification, not performance.
Signatures mirror the GPU host wrappers below (same parameters, same layout), named
`quadraticCTransformCPU1D` / `quadraticCTransformCPU2D`.

### GPU host wrappers — 1D

```cpp
template <typename T>
void quadraticCTransform1D(
    const T* X,    // source coordinates, shape (nx,)
    const T* Y,    // target coordinates, shape (ny,)
    const T* phi,  // source potential,   shape (nx,)
    T* out,        // output potential,   shape (ny,)  — caller-allocated
    Grid1D grid
);
```

Computes `out[iy] = min_ix { ½(X[ix]−Y[iy])² − phi[ix] }` for every target point.

### GPU host wrappers — 2D

```cpp
template <typename T>
void quadraticCTransform2D(              // naive joint double loop
    const T* Xaxis0, const T* Xaxis1,    // source axes, shapes (nx0,) and (nx1,)
    const T* Yaxis0, const T* Yaxis1,    // target axes, shapes (ny0,) and (ny1,)
    const T* phi,                         // source potential, shape (nx0*nx1,), row-major
    T* out,                               // output potential, shape (ny0*ny1,), row-major
    Grid2D grid
);

template <typename T>
void quadraticCTransform2DSeparable(     // separable two-pass algorithm; same result, faster
    const T* Xaxis0, const T* Xaxis1,
    const T* Yaxis0, const T* Yaxis1,
    const T* Phi,
    T* out,
    Grid2D grid
);
```

Both compute the 2D quadratic c-transform
`out[iy0,iy1] = min_{ix0,ix1} { ½((Xaxis0[ix0]−Yaxis0[iy0])² + (Xaxis1[ix1]−Yaxis1[iy1])²) − Phi[ix0,ix1] }`.
`quadraticCTransform2D` evaluates the joint minimization directly (O(nx0·nx1·ny0·ny1));
`quadraticCTransform2DSeparable` exploits additive separability of the cost to compute the
same result via two cheaper 1D passes (see [`separable_ctransform.md`](../math/separable_ctransform.md)
and [`separable_kernel_spec.md`](separable_kernel_spec.md)).

### GPU launch layer — 2D separable

```cpp
template <typename T>
void quadraticCTransform2DSeparable_launch(
    const T* dXaxis0, const T* dXaxis1,   // DEVICE pointers, already resident
    const T* dYaxis0, const T* dYaxis1,
    const T* dPhi,
    T* dOut,
    T* dScratchG,                          // caller-owned intermediate buffer, size nx0*ny1
    Grid2D grid,
    cudaStream_t stream                    // caller owns synchronization
);
```

Same computation as `quadraticCTransform2DSeparable`, but on device-resident data with no
allocation, copy, or synchronization — see the launch-layer description under
[Architecture](#architecture-layers). `dScratchG` is the intermediate buffer between the
two passes, allocated once by the caller and reused across iterations. The corresponding
host wrapper `quadraticCTransform2DSeparable` is a thin marshalling shell over this
function.

---

## Supported types

Explicit template instantiations are provided for `float` and `double`. No other types are
instantiated. Numerical guidance (when `double` is required) is in
[`../math/numerical_stability.md`](../math/numerical_stability.md).

## Ownership and allocation

All arrays are caller-owned. Host convenience wrappers allocate and free device memory
internally via `DeviceBuffer<T>` (see [`include/cuda_utils.cuh`](../../include/cuda_utils.cuh)).
The launch layer allocates nothing — every buffer, including scratch, is caller-provided.

## Row-major indexing (2D)

Source potential and output are stored in row-major (C) order:

```
phi[ix0 * nx1 + ix1]     out[iy0 * ny1 + iy1]
```

The X and Y grids are represented as separable tensor products of 1D axes.
