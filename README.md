# ctransform_cuda

A CUDA C++ library that computes the **discrete c-transform** — for the squared-Euclidean cost, exactly the grid-sampled **Moreau envelope**. It's the primitive behind optimal-transport duality (Kantorovich potentials, Hopf–Lax updates for Hamilton–Jacobi equations) and, more generally, any proximal-gradient / ADMM / operator-splitting method that needs a Moreau-envelope evaluation over a finite point set. See [`docs/`](docs/index.md) for full documentation, starting with [`docs/math/moreau_proximal.md`](docs/math/moreau_proximal.md) for the exact identity.

## Build

**Prerequisites:** CMake ≥ 3.26, CUDA ≥ 11, C++20-capable host compiler (GCC 10+ or Clang 10+).

If `nvcc` is not on your `PATH`, create `CMakeUserPresets.json` at the project root (already gitignored):

```json
{
  "version": 3,
  "configurePresets": [{
    "name": "local",
    "inherits": "default",
    "cacheVariables": {
      "CMAKE_CUDA_COMPILER": "/path/to/nvcc"
    }
  }],
  "buildPresets":  [{ "name": "local", "configurePreset": "local" }],
  "testPresets":   [{ "name": "local", "configurePreset": "local" }]
}
```

```bash
cmake --preset local
cmake --build --preset local
```

## Tests

```bash
ctest --preset local

# individual test names and timing:
./build/ctransform_tests

# single suite:
./build/ctransform_tests --gtest_filter="CudaVsCPU1D.*"

# list all tests without running:
./build/ctransform_tests --gtest_list_tests
```

## Demo and benchmark

```bash
./build/ctransform_naive_demo
./build/ctransform_perf
```

## Using this as a library

This library is meant to be embedded as the c-transform primitive inside a larger
GPU-resident solver (e.g. a Sinkhorn/back-and-forth optimal-transport loop) living in
another repo, not run standalone. Two integration points, from loosest to tightest
coupling:

- **Host convenience wrappers** — allocate, copy, launch, sync, copy back. Fine for
  calling once per host round-trip; not suited to a tight iterative loop.
- **Device-pointer launch layer** (`*_launch`) — no allocation, copy, or
  synchronization; takes device pointers and a `cudaStream_t` and lets the caller's own
  loop (CUDA, or JAX via an XLA custom call) own all three. This is the layer an
  external solver should call every iteration. **Status: implemented for the 2D
  separable kernel only** (`quadraticCTransform2DSeparable_launch`); the 1D and naive-2D
  equivalents are planned (see [`todo.md`](docs/engineering/todo.md)).

Full signatures and the layering rationale are in
[`docs/engineering/api.md`](docs/engineering/api.md). Until the launch layer covers all
kernels, treat this section as describing the intended integration shape, not a
complete/stable surface to build against yet.

## License

See [LICENSE](LICENSE).
