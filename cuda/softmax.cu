#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

static void check_cuda(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(error) << std::endl;
        std::exit(1);
    }
}

__device__ float block_reduce_max(float value, float* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return shared[0];
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

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float shared[];
    const int tid = threadIdx.x;
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    float thread_max = -CUDART_INF_F;
    for (int col = tid; col < cols; col += blockDim.x) {
        thread_max = fmaxf(thread_max, row_input[col]);
    }
    const float row_max = block_reduce_max(thread_max, shared);
    __syncthreads();

    float thread_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        thread_sum += expf(row_input[col] - row_max);
    }
    const float denom = block_reduce_sum(thread_sum, shared);
    __syncthreads();

    for (int col = tid; col < cols; col += blockDim.x) {
        row_output[col] = expf(row_input[col] - row_max) / denom;
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
    softmax_kernel<<<rows, threads, threads * sizeof(float)>>>(d_input, d_output, rows, cols);
    check_cuda(cudaGetLastError(), "softmax_kernel");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<float> h_output(rows * cols);
    check_cuda(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy output");

    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            sum += h_output[row * cols + col];
        }
        if (std::fabs(sum - 1.0f) > 1e-3f) {
            std::cerr << "Row " << row << " softmax sum mismatch: " << sum << std::endl;
            return 1;
        }
    }

    std::cout << "softmax passed for " << rows << " x " << cols << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
