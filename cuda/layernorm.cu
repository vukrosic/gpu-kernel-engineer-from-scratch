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

__device__ float block_reduce_sum(float value, float* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void layernorm_kernel(const float* input, float* output, int rows, int cols, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float shared[];
    const int tid = threadIdx.x;
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += row_input[col];
    }
    const float mean = block_reduce_sum(local_sum, shared) / cols;
    __syncthreads();

    float local_var = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float centered = row_input[col] - mean;
        local_var += centered * centered;
    }
    const float var = block_reduce_sum(local_var, shared) / cols;
    __syncthreads();

    const float inv_std = rsqrtf(var + eps);
    for (int col = tid; col < cols; col += blockDim.x) {
        row_output[col] = (row_input[col] - mean) * inv_std;
    }
}

int main() {
    const int rows = 4;
    const int cols = 16;
    std::vector<float> h_input(rows * cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            h_input[row * cols + col] = static_cast<float>(row + col * 0.25f);
        }
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    check_cuda(cudaMalloc(&d_input, h_input.size() * sizeof(float)), "cudaMalloc d_input");
    check_cuda(cudaMalloc(&d_output, h_input.size() * sizeof(float)), "cudaMalloc d_output");
    check_cuda(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy input");

    const int threads = 128;
    layernorm_kernel<<<rows, threads, threads * sizeof(float)>>>(d_input, d_output, rows, cols, 1e-5f);
    check_cuda(cudaGetLastError(), "layernorm_kernel");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<float> h_output(rows * cols);
    check_cuda(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy output");

    std::cout << "layernorm passed for " << rows << " x " << cols << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
