# Installation and development guide

How to get a working environment, which of the **two build loops** to use when, and what
to do when a build fails. For what the code *does*, start at [`../index.md`](../index.md).

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| CMake | ≥ 3.26 | `presets` v3 support |
| CUDA | ≥ 11 | `nvcc` must be findable — see [§2](#2-telling-cmake-where-nvcc-is) |
| Host compiler | GCC 10+ / Clang 10+ | C++20 |
| Python | 3.10+ | one env, used for both building and importing |
| `git` | any | **required at build time**, not just for version control |

`git` is a genuine build dependency: CMake's `FetchContent` shells out to `git clone` for
googletest and pybind11. A freshly created conda env does not inherit `git` from `base`,
and omitting it fails with `could not find git for clone of ..._-populate`.

### Environment

```bash
conda create -n ctransform-dev python=3.13 numpy pytest git -y
conda activate ctransform-dev
pip install "jax[cuda12]"                    # only if you need the JAX FFI path
pip install scikit-build-core pybind11       # only for the pip loop, see §4
```

The compiled modules are tied to the exact interpreter they were built against — the
`cpython-313-x86_64-linux-gnu` tag in each `.so` filename encodes the Python version and
ABI. **Build and import from the same env.** If you switch envs, delete `build/` before
reconfiguring: CMake caches the detected interpreter in `CMakeCache.txt` at first
configure and will not notice that a different env is now active.

---

## 2. Telling CMake where `nvcc` is

If `nvcc` is on your `PATH`, skip this. Otherwise there are two mechanisms, and **they
are not interchangeable** — which one applies depends on which build loop you use.

`CMakeUserPresets.json` at the project root (gitignored, so it's the right home for
machine-specific paths):

```json
{
  "version": 3,
  "configurePresets": [{
    "name": "local",
    "inherits": "default",
    "cacheVariables": { "CMAKE_CUDA_COMPILER": "/path/to/nvcc" }
  }],
  "buildPresets":  [{ "name": "local", "configurePreset": "local" }],
  "testPresets":   [{ "name": "local", "configurePreset": "local" }]
}
```

This works for the CMake loop (§3). **`pip` does not read CMake presets**, so the pip
loop (§4) needs the `CUDACXX` environment variable instead. Both point at the same
`nvcc`; only the delivery mechanism differs.

---

## 3. Build loop A — CMake (C++/CUDA development)

The loop for working on kernels, the launch layer, and the C++ test suite.

```bash
cmake --preset local          # or: cmake -S . -B build   (nvcc on PATH)
cmake --build --preset local -j
```

Tests, demo, benchmark:

```bash
ctest --preset local
./build/ctransform_tests --gtest_filter="CudaVsCPU1D.*"
./build/ctransform_tests --gtest_list_tests
./build/ctransform_naive_demo
./build/ctransform_perf
```

`pytest python/tests/ -v` also runs against **this** loop's output —
`python/tests/conftest.py` inserts `build/` and `python/` onto `sys.path` ahead of
site-packages. So the Python test suite exercises CMake-built binaries even when a pip
install exists. See [§5](#5-the-two-so-copies) for why that matters.

`CTRANSFORM_BUILD_TESTS` (default `ON`) gates the demo, googletest fetch, test binary,
and perf binary. The Python extension targets are built unconditionally.

---

## 4. Build loop B — pip (consuming the package from another repo)

The loop that makes `import ctransform_cuda` work from *outside* this repo — a
downstream JAX solver, a notebook, anywhere.

```bash
conda activate ctransform-dev
CUDACXX=/path/to/nvcc pip install --no-build-isolation -e .
```

Three things in that command each earn their place:

- **`CUDACXX=...`** — the preset substitute, per [§2](#2-telling-cmake-where-nvcc-is).
  Omit it and CMake fails at `project()` with `No CMAKE_CUDA_COMPILER could be found`.
- **`--no-build-isolation`** — keeps the build in your current env instead of a
  throwaway one, so the CUDA toolchain and jaxlib headers stay visible. The cost is that
  pip will **not** install build dependencies for you: `scikit-build-core` and
  `pybind11` must already be in the env (see §1).
- **`-e`** — editable, so Python sources under `python/ctransform_cuda/` are picked up
  from the working tree without reinstalling. Compiled code is *not*; see §5.

The build is driven by [`../../pyproject.toml`](../../pyproject.toml), which sets
`CTRANSFORM_BUILD_TESTS=OFF` — the pip loop skips googletest and the C++ binaries
entirely and builds only the two extension modules.

Verify from outside the repo, with no `PYTHONPATH` set:

```bash
cd /tmp && python -c "
import ctransform_cuda, ctransform_cuda.jax
print(ctransform_cuda.__file__)"
```

---

## 5. The two `.so` copies

Both loops produce `_ctransform_ffi...so`, in different places, and **they do not update
each other**:

| Copy | Written by | Imported by |
|---|---|---|
| `python/ctransform_cuda/_ctransform_ffi...so` | `cmake --build` (via `LIBRARY_OUTPUT_DIRECTORY`) | `pytest` in-repo, via `conftest.py`'s path insert |
| `site-packages/ctransform_cuda/_ctransform_ffi...so` | `pip install -e .` | everything else, including downstream repos |

Consequence: **after changing `ffi_handlers.cpp` or `bindings.cpp`, `cmake --build` alone
does not change what a downstream repo imports** — re-run `pip install -e .` too. Symptom
when you forget: a downstream consumer keeps hitting the old behavior while `pytest`
passes, or the reverse.

Accepted for now rather than fixed. Tracked in [`todo.md`](todo.md#build--packaging) with
the candidate fixes.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `BackendUnavailable: Cannot import 'scikit_build_core.build'` | `--no-build-isolation` doesn't install build deps | `pip install scikit-build-core pybind11` |
| `No CMAKE_CUDA_COMPILER could be found` | pip ignores CMake presets | prefix with `CUDACXX=/path/to/nvcc` |
| `could not find git for clone of ..._-populate` | `git` missing from the env | `conda install git` |
| `Unknown CMake command "FetchContent_Declare"` | `include(FetchContent)` fell inside `if(CTRANSFORM_BUILD_TESTS)` | keep the `include` unconditional (fixed; regression guard: build once with `-DCTRANSFORM_BUILD_TESTS=OFF`) |
| `ImportError: undefined symbol` from `_ctransform_ffi` | jaxlib version changed under a stale build | reinstall: `pip install -e .` |
| Module imports but is stale / wrong values | the §5 two-copy problem | run both loops |
| CMake still finds the old Python after switching envs | cached in `CMakeCache.txt` | `rm -rf build` and reconfigure |

`_ctransform_ffi` compiles against jaxlib's headers, located at configure time by
`python3 -c "import jaxlib, ..."` in [`../../CMakeLists.txt`](../../CMakeLists.txt). That
couples the extension to the installed jaxlib's ABI — changing jaxlib versions requires a
rebuild, not just a reinstall of jax.

---

## 7. What is not yet distributable

`pip install -e .` works locally; publishing a wheel does not. Blockers are listed in
[`python_extention_plan.md`](python_extention_plan.md#distribution-what-would-have-to-change-first)
— chiefly `CUDA_ARCHITECTURES native`, which compiles for the building machine's GPU only.
