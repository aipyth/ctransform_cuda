#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <chrono>

typedef double TensType;

#define CUDA_CHECK(x) do { cudaError_t e = (x); \
    if (e != cudaSuccess) fprintf(stderr, "%s:%d %s\n",\
            __FILE__, __LINE__, cudaGetErrorString(e)); } while(0)


struct Grid2D {
    std::size_t nx0;
    std::size_t nx1;

    std::size_t ny0;
    std::size_t ny1;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count)
        : count_ (count)
    {
        CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(TensType)));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    double* get() const { return ptr_; }

private:
    TensType* ptr_ = nullptr;
    std::size_t count_ = 0;
};

// === Naive 2D kernel - c-tranform ===
// the Kantorovich dual admits the following c-tranform
// for a potentials phi(x)
//      phi^c (y) = min_x c(x, y) - phi(x)
// 
// Assuming the quadratic cost
//      c(x, y) = 1/2 || x - y ||^2 = 0.5 ( (x_0 - y_0)^2 + (x_1 - y_1)^2 )
// one obtains
//      phi^c _{iy} = min_{ix}
//                      0.5 ( (x_{ix0} - y_{iy0})^2 + (x_{ix1} - y_{iy1})^2 )
//                      - phi_{ix0 * nx1 + ix1}

template <typename T>
__global__ void quadraticCTransform2D (
        const T* __restrict__ Xaxis0,       // target coords
        const T* __restrict__ Xaxis1,       // target coords
        const T* __restrict__ Yaxis0,       // source coords
        const T* __restrict__ Yaxis1,       // source coords
        const T* __restrict__ Phi,          // source values
        T* out,
        const Grid2D grid
        ) {
    // within a warp axis x (theadIdx.x) is varying faster
    // but we're accessing the out array as iy*grid.ny1 + iy1
    // meaning that we're varying faster in y axis
    // => iy1 is a fast-varying (contiguous) index
    // => to have coaleasced access we need to vary accordingly
    std::size_t iy0 = threadIdx.y + blockDim.y * blockIdx.y;
    std::size_t iy1 = threadIdx.x + blockDim.x * blockIdx.x;

    if (iy0 >= grid.ny0 || iy1 >= grid.ny1) return;

    T best = INFINITY;
    T yi0 = Yaxis0[iy0];       // load INTO REGISTRY !
    T yi1 = Yaxis1[iy1];       // load INTO REGISTRY !
    T d0, d1;

    for (std::size_t ix0 = 0; ix0 < grid.nx0; ix0++) {
        d0 = Xaxis0[ix0] - yi0;
        for (std::size_t ix1 = 0; ix1 < grid.nx1; ix1++) {
            d1 = Xaxis1[ix1] - yi1;
            best = fmin(
                best,
                0.5 * (d0 * d0 + d1 * d1)
                    - Phi[ix0 * grid.nx1 + ix1]
            );
        }
    }

    out[iy0 * grid.ny1 + iy1] = best;
}


template <typename T>
void printVector2D(
    std::string s,
    const std::vector<T>& vec,
    std::size_t nx,
    std::size_t ny
) {
    std::cout << s << std::endl;;
    for (std::size_t ix = 0; ix < nx; ix++) {
        std::cout << "| ";
        for (std::size_t iy = 0; iy < ny; iy++) {
            std::cout << vec[ix * ny + iy] << " ";
        }
        std::cout << "|" << std::endl;
    }
}

int main (void) {

    std::vector<TensType> hostXaxis0 = { -0.5 , 0.5 };
    std::vector<TensType> hostXaxis1 = { 0.0, 1.0, 2.0 };
    std::vector<TensType> hostYaxis0 = { -0.5 , 0.5 };
    std::vector<TensType> hostYaxis1 = { -0.5 , 0.5 };

    std::size_t nx0 = hostXaxis0.size();
    std::size_t nx1 = hostXaxis1.size();
    std::size_t ny0 = hostYaxis0.size();
    std::size_t ny1 = hostYaxis1.size();
    Grid2D grid { .nx0=nx0, .nx1=nx1, .ny0=ny0, .ny1=ny1 };

    std::size_t sizeX0 = grid.nx0 * sizeof(hostXaxis0[0]);
    std::size_t sizeX1 = grid.nx1 * sizeof(hostXaxis1[0]);
    std::size_t sizeY0 = grid.ny0 * sizeof(hostYaxis0[0]);
    std::size_t sizeY1 = grid.ny1 * sizeof(hostYaxis1[0]);

    std::vector<double> hostPhi = {
        0.2, 0.3, 0.4,
        0.3, 0.1, 0.2
    };
    std::size_t sizePhi = grid.nx0 * grid.nx1 * sizeof(hostPhi[0]);

    // printVector2D("Xaxis0", hostX, grid.nx0, grid.nx1);
    // printVector2D("y   grid", hostY, grid.ny0, grid.ny1);
    printVector2D("Phi grid", hostPhi, grid.nx0, grid.nx1);

    DeviceBuffer devXaxis0(grid.nx0);
    DeviceBuffer devXaxis1(grid.nx1);
    DeviceBuffer devYaxis0(grid.ny0);
    DeviceBuffer devYaxis1(grid.ny1);
    DeviceBuffer devPhi(grid.nx0 * grid.nx1);
    DeviceBuffer devPsi(grid.ny0 * grid.ny1);

    CUDA_CHECK(cudaMemcpy(devXaxis0.get(), hostXaxis0.data(), sizeX0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devXaxis1.get(), hostXaxis1.data(), sizeX1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devYaxis0.get(), hostYaxis0.data(), sizeY0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devYaxis1.get(), hostYaxis1.data(), sizeY1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devPhi.get(), hostPhi.data(), sizePhi, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks(
        (grid.ny0 + threads.x - 1) / threads.x,
        (grid.ny1 + threads.y - 1) / threads.y
    );

    quadraticCTransform2D<<<blocks, threads>>> (
        devXaxis0.get(), devXaxis1.get(), devYaxis0.get(), devYaxis1.get(),
        devPhi.get(), devPsi.get(), grid);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> hostPsi(grid.ny0 * grid.ny1);
    std::size_t sizePsi = grid.ny0 * grid.ny1 * sizeof(hostPsi[0]);

    CUDA_CHECK(cudaMemcpy(hostPsi.data(), devPsi.get(), sizePsi, cudaMemcpyDeviceToHost));

    printVector2D("psi", hostPsi, grid.ny0, grid.ny1);

    return 0;
}
