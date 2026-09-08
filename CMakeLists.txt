cmake_minimum_required(VERSION 3.26)

project(ctransform LANGUAGES CXX CUDA)

add_library(ctransform_lib STATIC
  src/ctransform1D_naive.cu
  src/ctransform2D_naive.cu
  src/ctransform2D_separable.cu
  src/cpu_reference.cpp
  )

target_include_directories(ctransform_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(ctransform_lib
    PUBLIC
        cxx_std_20
)

set_target_properties(ctransform_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES native
    )

option(CTRANSFORM_BUILD_TESTS "Build C++ test suite, demo, and perf binaries" ON)

# Used by both the (optional) googletest fetch below and the always-built pybind11
# fetch further down, so it must be included unconditionally.
include(FetchContent)

if(CTRANSFORM_BUILD_TESTS)
# ------- DEMO -------
add_executable(ctransform_naive_demo src/ctransform_naive_demo.cpp)
target_link_libraries(ctransform_naive_demo PRIVATE ctransform_lib)

# ------- TESTING ------
# - fetch googletest -
FetchContent_Declare(
    googletest
    GIT_REPOSITORY  https://github.com/google/googletest.git
    GIT_TAG         v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(CTest)

# build test binary
add_executable(ctransform_tests
    tests/test_ctransform_1d.cpp
    tests/test_cpu_reference.cpp
    tests/test_cuda_vs_cpu.cpp
    tests/test_randomized.cpp
    tests/test_stress.cpp
    tests/test_launch_layer.cu
)
target_link_libraries(ctransform_tests
    PRIVATE
    ctransform_lib  GTest::gtest_main
)

# register with ctest
add_test(NAME ctransform_tests COMMAND ctransform_tests)

# performance tests
add_executable(ctransform_perf tests/test_perf.cpp)
target_link_libraries(ctransform_perf PRIVATE ctransform_lib)

endif()

# ------- PYTHON EXTENSION --------
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Make pybind11 use modern FindPython instead of its legacy FindPythonLibsNew
# (which sets policy CMP0148 to OLD and warns).
set(PYBIND11_FINDPYTHON ON)

# Prefer the pip-installed pybind11 (its CMake config lives in site-packages);
# fall back to fetching it so a machine without it still configures.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_CMAKE_DIR_RESULT
  ERROR_QUIET
)
if(PYBIND11_CMAKE_DIR_RESULT EQUAL 0)
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not installed; fetching v3.1.0")
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.1.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(ctransform_cuda_py
  python/src/bindings.cpp
)

target_link_libraries(ctransform_cuda_py PRIVATE ctransform_lib)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import jaxlib, os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
  OUTPUT_VARIABLE JAXLIB_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE JAXLIB_INCLUDE_DIR_RESULT
  ERROR_QUIET
)

# Without jaxlib the include path above is empty and _ctransform_ffi fails much
# later with a confusing "xla/ffi/api/ffi.h: No such file or directory", so skip
# the target and say why instead.
if(NOT JAXLIB_INCLUDE_DIR_RESULT EQUAL 0)
  message(STATUS
    "jaxlib not importable from ${Python3_EXECUTABLE}; skipping the _ctransform_ffi "
    "target. Install it (pip install \"jax[cuda12]\") or point CMake at the right "
    "interpreter with -DPython3_EXECUTABLE=...")
else()
  pybind11_add_module(_ctransform_ffi python/src/ffi_handlers.cpp)
  # SYSTEM: jaxlib's XLA FFI headers emit -Wattributes and -Wreturn-type warnings
  # that are not ours to fix.
  target_include_directories(_ctransform_ffi
    SYSTEM PRIVATE
      ${JAXLIB_INCLUDE_DIR}
  )
  target_link_libraries(_ctransform_ffi
    PRIVATE
      ctransform_lib
  )
  set_target_properties(_ctransform_ffi PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/ctransform_cuda
  )
  install(TARGETS _ctransform_ffi DESTINATION ctransform_cuda)
endif()

install(TARGETS ctransform_cuda_py DESTINATION .)
