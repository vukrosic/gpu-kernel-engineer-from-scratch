#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static void check_cuda(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(error) << std::endl;
        std::exit(1);
    }
}

__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char** argv) {
    int n = 1 << 20;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }

    std::vector<float> h_a(n), h_b(n), h_out(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_out = nullptr;
    check_cuda(cudaMalloc(&d_a, n * sizeof(float)), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_b, n * sizeof(float)), "cudaMalloc d_b");
    check_cuda(cudaMalloc(&d_out, n * sizeof(float)), "cudaMalloc d_out");
    check_cuda(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_a");
    check_cuda(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_b");

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_out, n);
    check_cuda(cudaGetLastError(), "vector_add_kernel");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check_cuda(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy d_out");

    for (int i = 0; i < n; ++i) {
        const float expected = h_a[i] + h_b[i];
        if (std::fabs(h_out[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << std::endl;
            return 1;
        }
    }

    std::cout << "vector_add passed for " << n << " elements" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return 0;
}
