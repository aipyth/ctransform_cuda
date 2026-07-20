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
  external solver should call every iteration. **Status: implemented for all three
  kernels** (`quadraticCTransform1D_launch`, `quadraticCTransform2D_launch`,
  `quadraticCTransform2DSeparable_launch`).

Full signatures and the layering rationale are in
[`docs/engineering/api.md`](docs/engineering/api.md).

## Python bindings

A NumPy-facing pybind11 module wraps the 1D and 2D host wrappers; the JAX/XLA FFI target
described in [`docs/engineering/jax_ffi_integration.md`](docs/engineering/jax_ffi_integration.md)
is separate, planned work and not part of this binding.

### Environment setup

The compiled module is tied to the exact Python it was built against (the
`cpython-*-x86_64-linux-gnu` tag in the output filename encodes the ABI + version), so
build and test with the same interpreter — don't build in one env and `import` from
another. A dedicated env keeps this isolated from `base`:

```bash
conda create -n ctransform-dev python=3.13 numpy pytest git -y
conda activate ctransform-dev
```

`git` is included deliberately: CMake's `FetchContent` (used for googletest and
pybind11) shells out to `git clone`, and a minimal freshly created env won't have `git`
on its `PATH` even if `base` does — this fails with `could not find git for
clone of ..._-populate` if omitted.

If you switch environments (or created one without `git` and are adding it after the
fact), reconfigure from a clean build directory — CMake caches the detected Python
interpreter in `CMakeCache.txt` at first configure and won't pick up a newly activated
env otherwise:

```bash
rm -rf build
```

### Build and test

```bash
cmake -S . -B build --preset local   # --preset local is required if nvcc isn't on PATH, see Build above
cmake --build build -j
```

This produces `ctransform_cuda_py.cpython-*.so` somewhere under `build/`. Test it:

```bash
pytest python/tests/ -v
```

`python/tests/conftest.py` locates the built `.so` automatically, so no manual
`PYTHONPATH` export is needed. See
[`docs/engineering/python_extention_plan.md`](docs/engineering/python_extention_plan.md)
for the intended API shape and validation rules.

## License

See [LICENSE](LICENSE).
