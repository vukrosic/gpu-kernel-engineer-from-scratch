# Month 01: GPU Foundations

## Goal

Understand the GPU programming model and write the first correct kernels.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 1 | GPU mental model | Run the starter benchmark and explain CPU vs GPU execution. | You can explain why parallel workloads fit GPUs. |
| 2 | CUDA setup and first kernel | Write vector add and compare with a CPU reference. | Vector add is correct and benchmarked. |
| 3 | Tensor shapes, memory layout, indexing | Understand row-major layout, flattening, strides, and tensor indexing. | You can map tensor positions to flat memory. |
| 4 | Elementwise kernel patterns | Read copy, scale, square, ReLU, add, and axpy-shaped kernels. | You can explain the shared elementwise kernel pattern. |

## Minimum Viable Month

One correct vector add kernel, one test, and one benchmark.

## Portfolio Note

Explain the difference between CPU serial execution and GPU parallel execution
using your own vector add benchmark.
