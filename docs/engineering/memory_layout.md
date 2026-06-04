# Memory layout

## 1D layout

All arrays are flat 1D with sequential indexing.

| Array | Type | Size | Access |
|---|---|---|---|
| X | `const T*` | nx | `X[ix]`, ix ∈ [0, nx) |
| Y | `const T*` | ny | `Y[iy]`, iy ∈ [0, ny) |
| Phi | `const T*` | nx | `Phi[ix]`, ix ∈ [0, nx) |
| out | `T*` | ny | `out[iy]`, iy ∈ [0, ny) |

Each thread owns a single `iy` and reads all `nx` elements of X and Phi.

---

## 2D axis-separated representation

The source domain is a **tensor product grid**, not a flat list of 2D points:

```
X = Xaxis0[nx0] × Xaxis1[nx1]
```

This represents nx0 · nx1 source points (x₀, x₁) where x₀ ∈ Xaxis0 and x₁ ∈ Xaxis1. The axes are stored as separate 1D arrays; there is no explicit flat array of 2D coordinates.

Similarly:
```
Y = Yaxis0[ny0] × Yaxis1[ny1]
```

**Advantage**: axis storage costs O(nx0 + nx1) instead of O(nx0 · nx1). The cost function decomposition ½(d₀² + d₁²) allows computing d₀ and d₁ independently from the two axes, so no 2D coordinate array is ever needed.

---

## Phi storage (2D, row-major)

Phi is a 2D array stored in **row-major (C) order**: the axis-1 index varies fastest.

```
Phi[ix0 * nx1 + ix1]   for ix0 ∈ [0, nx0), ix1 ∈ [0, nx1)
```

- `ix0`: slow (outer) index, corresponding to Xaxis0
- `ix1`: fast (inner) index, corresponding to Xaxis1

The nested kernel loop matches this convention:
```cpp
for (std::size_t ix0 = 0; ix0 < nx0; ++ix0) {
    for (std::size_t ix1 = 0; ix1 < nx1; ++ix1) {
        // Phi[ix0 * nx1 + ix1]
    }
}
```

---

## Output storage (2D, row-major)

Output `out` uses the same row-major convention:

```
out[iy0 * ny1 + iy1]   for iy0 ∈ [0, ny0), iy1 ∈ [0, ny1)
```

- `iy0`: slow index, corresponding to Yaxis0
- `iy1`: fast index, corresponding to Yaxis1

---

## Thread-to-element mapping (2D kernel)

```
iy1 = blockIdx.x * blockDim.x + threadIdx.x   // fast axis (axis 1)
iy0 = blockIdx.y * blockDim.y + threadIdx.y   // slow axis (axis 0)
```

Block dimensions are `{16, 16}`. Within a warp (32 consecutive threads), `threadIdx.x` varies from 0 to 15 twice across two rows. Adjacent threads in x-direction differ by 1 in `iy1`.

Since consecutive iy1 values map to consecutive memory addresses in `out[iy0*ny1 + iy1]`, the write to `out` is **coalesced** when `ny1 ≥ 16` (which is always true for non-trivial grids).

---

## Global memory access patterns

| Access | Pattern | Notes |
|---|---|---|
| `out[iy0*ny1 + iy1]` | Coalesced write | Adjacent threads write adjacent addresses |
| `Phi[ix0*nx1 + ix1]` | Streaming sequential | Each thread reads full Phi in row-major order; L2 prefetching helps |
| `Xaxis0[ix0]` | Broadcast | All threads in a block share ix0 in the outer loop; L1/texture caching |
| `Xaxis1[ix1]` | Broadcast | All threads in a block share ix1 in the inner loop |
| `Yaxis0[iy0]` | Uniform per row | One read per thread, stored in register |
| `Yaxis1[iy1]` | One read per thread | Stored in register; adjacent threads read adjacent Yaxis1 entries → coalesced |

The most expensive access is `Phi`: it is read in full for every output thread. For a grid with ny0·ny1 output points and nx0·nx1 source points, the total Phi traffic is `ny0·ny1·nx0·nx1·sizeof(T)` bytes without caching.

---

## Byte sizes for reference

| Configuration | N source | M target | Phi (bytes, double) | out (bytes, double) |
|---|---|---|---|---|
| 1D, N=M=1000 | 1000 | 1000 | 8 KB | 8 KB |
| 1D, N=M=10000 | 10000 | 10000 | 80 KB | 80 KB |
| 2D, 32×32 source, 32×32 target | 1024 | 1024 | 8 KB | 8 KB |
| 2D, 128×128 source, 128×128 target | 16384 | 16384 | 128 KB | 128 KB |

Phi fits in L2 cache for small grids (typical L2: 4–40 MB on modern GPUs), which significantly reduces bandwidth for repeated reads across output threads.
