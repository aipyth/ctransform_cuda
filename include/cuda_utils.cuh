#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>


#define CUDA_CHECK(x) do { cudaError_t e = (x); \
    if (e != cudaSuccess) fprintf(stderr, "%s:%d %s\n",\
            __FILE__, __LINE__, cudaGetErrorString(e)); } while(0)


template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count)
        : count_ (count)
    {
        CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    T* get() const { return ptr_; }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};
