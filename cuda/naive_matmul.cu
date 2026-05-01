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

__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }

    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        acc += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = acc;
}

int main() {
    const int m = 32;
    const int k = 64;
    const int n = 48;

    std::vector<float> h_a(m * k);
    std::vector<float> h_b(k * n);
    std::vector<float> h_c(m * n);
    std::vector<float> h_ref(m * n, 0.0f);

    for (int i = 0; i < m * k; ++i) {
        h_a[i] = static_cast<float>(i % 11);
    }
    for (int i = 0; i < k * n; ++i) {
        h_b[i] = static_cast<float>(i % 7);
    }
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            for (int i = 0; i < k; ++i) {
                h_ref[row * n + col] += h_a[row * k + i] * h_b[i * n + col];
            }
        }
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    check_cuda(cudaMalloc(&d_a, h_a.size() * sizeof(float)), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_b, h_b.size() * sizeof(float)), "cudaMalloc d_b");
    check_cuda(cudaMalloc(&d_c, h_c.size() * sizeof(float)), "cudaMalloc d_c");
    check_cuda(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy a");
    check_cuda(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy b");

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    check_cuda(cudaGetLastError(), "matmul_kernel");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check_cuda(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy c");

    for (int i = 0; i < m * n; ++i) {
        if (std::fabs(h_c[i] - h_ref[i]) > 1e-3f) {
            std::cerr << "Mismatch at index " << i << std::endl;
            return 1;
        }
    }

    std::cout << "naive_matmul passed for " << m << " x " << k << " and " << k << " x " << n << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
