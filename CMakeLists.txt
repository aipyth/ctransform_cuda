cmake_minimum_required(VERSION 3.26)

project(ctransform, LANGUAGES CXX CUDA)

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

# ------- DEMO -------
add_executable(ctransform_naive_demo src/ctransform_naive_demo.cpp)
target_link_libraries(ctransform_naive_demo PRIVATE ctransform_lib)

# ------- TESTING ------
# - fetch googletest -
include(FetchContent)
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

# ------- PYTHON EXTENSION --------
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(ctransform_cuda_py
  python/src/bindings.cpp
)

target_link_libraries(ctransform_cuda_py PRIVATE ctransform_lib)
