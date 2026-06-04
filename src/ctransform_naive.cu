#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); \
    if (e != cudaSuccess) fprintf(stderr, "%s:%d %s\n",\
            __FILE__, __LINE__, cudaGetErrorString(e)); } while(0)


struct Grid {
    std::size_t nx;
    std::size_t ny;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count)
        : count_ (count)
    {
        CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(double)));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    double* get() const { return ptr_; }

private:
    double* ptr_ = nullptr;
    std::size_t count_ = 0;
};

// === Naive 1D kernel - c-tranform ===
// the Kantorovich dual admits the following c-tranform
// for a potentials phi(x)
//      phi^c (y) = min_x c(x, y) - phi(x)
// 
// Assuming the quadratic cost
//      c(x, y) = 1/2 | x - y |^2
// one obtains
//      phi^c _{iy} = min_{ix} 1/2 (x_{ix} - y_{iy})^2 - phi_{ix}

template <typename T>
__global__ void quadraticCTransform1D (
        const T* __restrict__ X,     // target coords
        const T* __restrict__ Y,     // source coords
        const T* __restrict__ Phi,   // source values
        T* out,
        const Grid grid
        ) {
    int iy = threadIdx.x + blockDim.x * blockIdx.x;
    if (iy >= grid.ny) return;

    T best = INFINITY;
    T yi = Y[iy];       // load INTO REGISTRY !
    T dx;
    T candidate;

    for (int ix = 0; ix < grid.nx; ix++) {
        dx = X[ix] - yi;
        candidate = 0.5 * dx * dx - Phi[ix];
        best = min(best, candidate);
    }

    out[iy] = best;
}


template <typename T>
void printVector(std::string s, const std::vector<T>& vec) {
    std::cout << s << ": ";
    for (T x : vec) std::cout << x << " ";
    std::cout << std::endl;
}

int main (void) {


    std::vector<double> hostX = { -0.5, 0.0, 0.5 };
    std::vector<double> hostY = { -0.2, 0.0, 0.2, 0.5 };

    std::size_t nx = hostX.size();
    std::size_t ny = hostY.size();
    Grid grid { nx, ny };

    std::size_t sizeX = grid.nx * sizeof(hostX[0]);
    std::size_t sizeY = grid.ny * sizeof(hostY[0]);


    std::vector<double> hostPhi = { 0.2, 0., 0.3 };

    printVector("x   grid", hostX);
    printVector("y   grid", hostY);
    printVector("Phi grid", hostPhi);

    DeviceBuffer devX(grid.nx);
    DeviceBuffer devY(grid.ny);
    DeviceBuffer devPhi(grid.nx);
    DeviceBuffer devPsi(grid.ny);

    CUDA_CHECK(cudaMemcpy(devX.get(), hostX.data(), sizeX, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devY.get(), hostY.data(), sizeY, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devPhi.get(), hostPhi.data(), sizeX, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (grid.ny + threads - 1) / threads;

    quadraticCTransform1D<<<blocks, threads>>> (devX.get(), devY.get(), devPhi.get(), devPsi.get(), grid);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> hostPsi(grid.ny);

    CUDA_CHECK(cudaMemcpy(hostPsi.data(), devPsi.get(), sizeY, cudaMemcpyDeviceToHost));

    printVector("psi", hostPsi);

    return 0;
}
