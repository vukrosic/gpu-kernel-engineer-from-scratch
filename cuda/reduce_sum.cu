#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

static void check_cuda(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(error) << std::endl;
        std::exit(1);
    }
}

__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
    extern __shared__ float shared[];
    const unsigned int tid = threadIdx.x;
    const unsigned int idx = blockIdx.x * blockDim.x * 2 + tid;

    float value = 0.0f;
    if (idx < n) {
        value += input[idx];
    }
    if (idx + blockDim.x < n) {
        value += input[idx + blockDim.x];
    }

    shared[tid] = value;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

int main(int argc, char** argv) {
    int n = 1 << 20;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }

    std::vector<float> h_input(n);
    for (int i = 0; i < n; ++i) {
        h_input[i] = 1.0f + static_cast<float>(i % 7);
    }

    const float expected = std::accumulate(h_input.begin(), h_input.end(), 0.0f);

    float* d_current = nullptr;
    float* d_next = nullptr;
    check_cuda(cudaMalloc(&d_current, n * sizeof(float)), "cudaMalloc d_current");
    check_cuda(cudaMalloc(&d_next, n * sizeof(float)), "cudaMalloc d_next");
    check_cuda(cudaMemcpy(d_current, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy input");

    int current_n = n;
    const int threads = 256;
    while (current_n > 1) {
        const int blocks = (current_n + threads * 2 - 1) / (threads * 2);
        reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_current, d_next, current_n);
        check_cuda(cudaGetLastError(), "reduce_sum_kernel");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        current_n = blocks;
        std::swap(d_current, d_next);
    }

    float result = 0.0f;
    check_cuda(cudaMemcpy(&result, d_current, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy result");

    if (std::fabs(result - expected) > 1e-3f) {
        std::cerr << "Reduction mismatch: " << result << " vs " << expected << std::endl;
        return 1;
    }

    std::cout << "reduce_sum passed for " << n << " elements" << std::endl;

    cudaFree(d_current);
    cudaFree(d_next);
    return 0;
}
