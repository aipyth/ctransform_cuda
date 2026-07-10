# CUDA C++ architecture

## Repository layout

```
ctransform_cuda/
├── include/
│   ├── ctransform.hpp      # public API: Grid structs, host wrapper declarations
│   ├── cuda_utils.cuh      # DeviceBuffer<T> RAII wrapper, CUDA_CHECK macro
│   └── print_utils.hpp     # debug print helpers (demo use only)
├── src/
│   ├── ctransform_naive.cu       # 1D kernel + host wrapper
│   ├── ctransform2d_naive.cu     # 2D kernel + host wrapper
│   ├── cpu_reference.cpp         # serial CPU reference implementations
│   └── ctransform_naive_demo.cpp # minimal usage example
└── tests/
    ├── test_ctransform_1d.cpp    # 1D CUDA correctness (closed-form cases)
    ├── test_cpu_reference.cpp    # CPU reference correctness (closed-form cases)
    ├── test_cuda_vs_cpu.cpp      # GPU vs CPU agreement (1D and 2D)
    └── test_perf.cpp             # timing benchmark (separate binary, not CTest)
```

---

## Host/device API boundary

The host wrappers in `ctransform_naive.cu` and `ctransform2d_naive.cu` follow a fixed pattern:

1. Allocate `DeviceBuffer<T>` objects for each array (triggers `cudaMalloc`)
2. Copy inputs host → device (`cudaMemcpy H2D`)
3. Compute launch parameters
4. Launch the kernel
5. Check `cudaGetLastError` and `cudaDeviceSynchronize`
6. Copy output device → host (`cudaMemcpy D2H`)
7. `DeviceBuffer` destructors free GPU memory on scope exit (RAII)

The device kernel receives only raw `const T* __restrict__` input pointers, a raw `T*` output pointer, and a Grid struct. No host-side types, no exceptions, and no virtual dispatch enter device code.

---

## 1D kernel: `quadraticCTransform1DKernel<T>`

Launch configuration:
```
threads_per_block = 256
blocks = ceil(ny / 256)
```

Each thread owns one output index `iy`. The thread loads `Y[iy]` into a register once, then scans all source indices `ix ∈ [0, nx)` sequentially, maintaining a running minimum.

---

## 2D kernel: `quadraticCTransform2DKernel<T>`

Launch configuration:
```
blockDim  = { 16, 16, 1 }
gridDim.x = ceil(ny1 / 16)   // fast axis (axis 1, contiguous in memory)
gridDim.y = ceil(ny0 / 16)   // slow axis (axis 0)
```

Thread index mapping:
```
iy1 = blockIdx.x * 16 + threadIdx.x   // fast axis
iy0 = blockIdx.y * 16 + threadIdx.y   // slow axis
```

Each thread loads `Yaxis0[iy0]` and `Yaxis1[iy1]` into registers, then iterates over all `(ix0, ix1)` pairs. The outer loop precomputes `d0 = Xaxis0[ix0] - yi0` before entering the inner `ix1` loop.

The `iy1` index is the fast-varying axis, so the output write `out[iy0*ny1 + iy1]` is coalesced across a warp.

---

## Reduction strategy

Both kernels use a **sequential per-thread scan**:

```cpp
T best = INFINITY;
for (...) {
    best = min(best, candidate);
}
out[iy] = best;
```

Properties:
- **Correct**: exact minimum within floating-point arithmetic
- **Deterministic**: no atomics, no parallel reduction, no race conditions
- **Simple**: no inter-thread communication
- **Not optimal for large N**: thread-serial inner loop is the bottleneck for large source grids; shared-memory tiling or warp-level reduction would improve throughput

`min()` is used instead of `fmin()` in device code: in C++20 `std::fmin` is declared `constexpr __host__` and cannot be called from `__global__` functions. The CUDA `min()` intrinsic maps to the same hardware instruction and has identical behavior for finite and ±∞ inputs.

---

## Template instantiations

The host wrappers are function templates. Their definitions live in `.cu` files (compiled by nvcc), which g++-compiled test files cannot see. Each `.cu` file therefore provides explicit instantiations at the bottom:

```cpp
template void quadraticCTransform<float>(...);
template void quadraticCTransform<double>(...);
```

This forces nvcc to emit the compiled symbols so the linker can resolve calls from `.cpp` test files.

---

## Error handling

`CUDA_CHECK` is defined in `include/cuda_utils.cuh` and wraps every CUDA call:

```cpp
#define CUDA_CHECK(x) do { cudaError_t e = (x); \
    if (e != cudaSuccess) fprintf(stderr, "%s:%d %s\n", \
        __FILE__, __LINE__, cudaGetErrorString(e)); } while(0)
```

Errors are reported to stderr. The macro does not abort; the caller sees a zero-initialised or garbage output if a prior CUDA call failed silently, so checking all return codes is essential.

---

## Build system

CMake 3.26+, C++20 and CUDA 20 standards, `CUDA_ARCHITECTURES native`.

GoogleTest (v1.14.0) is fetched via `FetchContent` at configure time. Per-developer CUDA compiler paths are set in `CMakeUserPresets.json` (gitignored) inheriting from the shared `CMakePresets.json`.
