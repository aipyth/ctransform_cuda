# Public API

All declarations are in [`include/ctransform.hpp`](../../include/ctransform.hpp).

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

## CUDA (GPU) — 1D

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

## CUDA (GPU) — 2D

```cpp
template <typename T>
void quadraticCTransform2D(
    const T* Xaxis0, const T* Xaxis1,   // source axes, shapes (nx0,) and (nx1,)
    const T* Yaxis0, const T* Yaxis1,   // target axes, shapes (ny0,) and (ny1,)
    const T* phi,                        // source potential, shape (nx0*nx1,), row-major
    T* out,                              // output potential, shape (ny0*ny1,), row-major
    Grid2D grid
);
```

## CPU reference — 1D and 2D

Same signatures as the GPU wrappers above, named `quadraticCTransformCPU1D` / `quadraticCTransformCPU2D`. Serial implementation intended for correctness verification, not performance.

## Supported types

Explicit instantiations are provided for `float` and `double`. No other types are instantiated.

## Ownership and allocation

All arrays are caller-owned. The GPU wrappers allocate and free device memory internally via `DeviceBuffer<T>` (see [`include/cuda_utils.cuh`](../../include/cuda_utils.cuh)). The caller provides host pointers for all inputs and the output.

## Row-major indexing (2D)

Source potential and output are stored in row-major (C) order:

```
phi[ix0 * nx1 + ix1]     out[iy0 * ny1 + iy1]
```

The X and Y grids are represented as separable tensor products of 1D axes. Full 2D coordinate pairs are never materialised.
