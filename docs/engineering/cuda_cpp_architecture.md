# CUDA C++ architecture

## Module layout

```
ctransform_cuda/
├── src/
│   ├── ctransform_naive.cu       # 1D c-transform: kernel, DeviceBuffer, Grid, host driver
│   ├── ctransform2d_naive.cu     # 2D c-transform: kernel, DeviceBuffer, Grid2D, host driver
│   ├── vector_add.cu             # Vestigial demo (not part of the math project)
│   └── main.cpp                  # Vestigial demo driver (not part of the math project)
├── include/
│   └── vector_add.hpp            # Empty placeholder (not used)
└── tests/
    └── (empty)
```

The two mathematical source files are currently self-contained: each includes its own `DeviceBuffer` class, grid struct, host driver, and kernel. There is no shared header for common utilities.

---

## Shared data structures (currently duplicated)

Both c-transform files define:

**`DeviceBuffer<T>`** — RAII wrapper for GPU memory:
```cpp
template <typename T>
class DeviceBuffer {
    T* ptr_;
    std::size_t count_;
public:
    DeviceBuffer(std::size_t count);   // cudaMalloc
    ~DeviceBuffer();                   // cudaFree
    T* get();                          // raw pointer for kernel args
    std::size_t count() const;
};
```

**`Grid`** (1D):
```cpp
struct Grid {
    std::size_t nx;   // number of source points
    std::size_t ny;   // number of target points
};
```

**`Grid2D`** (2D):
```cpp
struct Grid2D {
    std::size_t nx0, nx1;   // source axes sizes
    std::size_t ny0, ny1;   // target axes sizes
};
```

These should eventually be extracted to a shared header in `include/`.

---

## Host/device API boundary

The host side is responsible for:

1. Allocating `DeviceBuffer` objects (triggers `cudaMalloc`)
2. Copying input arrays host → device (`cudaMemcpy H2D`)
3. Computing kernel launch parameters (grid/block dimensions)
4. Launching the kernel
5. Calling `cudaDeviceSynchronize` and checking the error code
6. Copying output array device → host (`cudaMemcpy D2H`)
7. `DeviceBuffer` destructors free GPU memory on scope exit

The device kernel receives only:
- Raw `const T* __restrict__` pointers to input arrays
- Raw `T*` pointer to output array
- Integer size parameters (nx, ny or nx0, nx1, ny0, ny1)

No host-side allocations, no C++ exceptions, and no virtual dispatch occur in device code.

---

## Kernel: `quadraticCTransform1D<T>`

```
__global__ void quadraticCTransform1D(
    const T* __restrict__ X,     // source coordinates [nx]
    const T* __restrict__ Y,     // target coordinates [ny]
    const T* __restrict__ Phi,   // source potential  [nx]
    T* out,                      // output potential  [ny]
    std::size_t nx,
    std::size_t ny
)
```

Launch configuration:
```
threads_per_block = 256
blocks = (ny + 255) / 256
```

Each thread owns one output index `iy = blockIdx.x * 256 + threadIdx.x`. If `iy >= ny` the thread exits immediately (boundary guard). The thread then scans all `ix ∈ [0, nx)` sequentially.

---

## Kernel: `quadraticCTransform2D<T>`

```
__global__ void quadraticCTransform2D(
    const T* __restrict__ Xaxis0,   // source axis 0  [nx0]
    const T* __restrict__ Xaxis1,   // source axis 1  [nx1]
    const T* __restrict__ Yaxis0,   // target axis 0  [ny0]
    const T* __restrict__ Yaxis1,   // target axis 1  [ny1]
    const T* __restrict__ Phi,      // source potential [nx0*nx1], row-major
    T* out,                         // output potential [ny0*ny1], row-major
    std::size_t nx0, std::size_t nx1,
    std::size_t ny0, std::size_t ny1
)
```

Launch configuration:
```
blockDim  = { 16, 16, 1 }
gridDim.x = (ny1 + 15) / 16    // fast axis (axis 1)
gridDim.y = (ny0 + 15) / 16    // slow axis (axis 0)
```

Thread index mapping:
```
iy1 = blockIdx.x * 16 + threadIdx.x   // fast axis
iy0 = blockIdx.y * 16 + threadIdx.y   // slow axis
```

Threads outside the domain (iy0 >= ny0 or iy1 >= ny1) exit immediately.

---

## Memory strategy

| Component | Strategy | Rationale |
|---|---|---|
| Y coordinate | Loaded into register once per thread | Reused across all nx iterations |
| X, Phi | Read from global memory in inner loop | Sequential streaming access |
| Output | Written once per thread | Coalesced across warp (iy1 is fast axis) |
| Shared memory | Not used | Correctness baseline; tiling is future work |

Register caching of `yi` (and `yi0`, `yi1` in 2D) avoids re-reading per iteration and is the main performance optimization in the current implementation.

---

## Reduction strategy

The minimum over the source domain is computed by a **sequential per-thread scan**:

```cpp
T result = INFINITY;
for (std::size_t ix = 0; ix < nx; ++ix) {
    result = fmin(result, cost - phi);
}
```

This is:
- **Correct**: produces the exact minimum within floating-point arithmetic
- **Deterministic**: no race conditions, no atomics, no parallel reduction
- **Simple**: one thread per output point; no inter-thread communication
- **Not optimal for large N**: for N >> 1, this thread serial loop is the bottleneck; warp-level or shared-memory parallel reduction would improve occupancy utilization

---

## Error handling

Both files use a `cudaCheck` macro (defined locally):

```cpp
#define cudaCheck(err, file, line) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err)); \
        exit(1); \
    }
```

All `cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`, and kernel launch errors are checked. On failure the process exits with code 1. There is no exception-based error path.

Known issue: the macro is duplicated across both files. It should be moved to a shared header.

---

## Build system

**Current state**: `CMakeLists.txt` defines only the `vector_add` demo target:

```cmake
add_library(my_cuda_lib STATIC src/vector_add.cu)
add_executable(vector_add_test src/main.cpp)
target_link_libraries(vector_add_test PRIVATE my_cuda_lib)
```

The c-transform executables were compiled manually and are committed to the project root as pre-built binaries (`ctransform_naive`, `ctransform2d`). They are **not CMake targets**.

**To fix**: add the following to `CMakeLists.txt`:

```cmake
add_executable(ctransform_naive src/ctransform_naive.cu)
set_target_properties(ctransform_naive PROPERTIES
    CUDA_STANDARD 17 CUDA_ARCHITECTURES Native)

add_executable(ctransform2d src/ctransform2d_naive.cu)
set_target_properties(ctransform2d PROPERTIES
    CUDA_STANDARD 17 CUDA_ARCHITECTURES Native)
```

CUDA architecture setting: `CMAKE_CUDA_ARCHITECTURES = Native` (targets the GPU present at build time).

---

## Test executable layout

The current executables (`ctransform_naive`, `ctransform2d`) are **self-testing demo drivers**:

- Hardcoded small inputs at the bottom of each `.cu` file
- Run the kernel, copy results back, print to stdout
- No automated pass/fail assertions
- Verification is by visual inspection

The intended evolution is to separate driver code into `tests/` with proper assert-based test cases. See `docs/engineering/test_strategy.md`.
