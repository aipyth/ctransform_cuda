# ctransform_cuda

A standalone CUDA C++ library that computes the **discrete c-transform** operator — the core computational primitive of optimal transport duality. See [`docs/`](docs/index.md) for full documentation.

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

## License

See [LICENSE](LICENSE).
