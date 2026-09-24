#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>


#define CUDA_CHECK(x) do { cudaError_t _e = (x); \
    if (_e != cudaSuccess) \
        throw std::runtime_error( \
            std::string(__FILE__) + ":" + std::to_string(__LINE__) \
            + " " + cudaGetErrorString(_e)); \
    } while(0)


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


class CudaStream {
public:
  CudaStream() {
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking));
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  ~CudaStream() { if (s_) { cudaStreamDestroy(s_); } }

  cudaStream_t get() const { return s_; }

private:
  cudaStream_t s_ = nullptr;
};


class CudaEvent {
public:
  CudaEvent() { CUDA_CHECK(cudaEventCreate(&e_)); }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  ~CudaEvent() { if (e_) cudaEventDestroy(e_); }

  cudaEvent_t get() const { return e_; }

private:
  cudaEvent_t e_ = nullptr;
};
