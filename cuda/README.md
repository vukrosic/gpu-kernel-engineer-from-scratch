# CUDA Track

This folder is for CUDA C++ kernels and notes as the course expands.

## Learning Order

1. vector add
2. elementwise kernels
3. memory bandwidth experiments
4. reductions
5. synchronization and atomics
6. scans
7. tiled matmul
8. transformer-adjacent kernels

## Rules

- Write a CPU or NumPy reference first.
- Test correctness before benchmarking.
- Benchmark with warmup and repeated measurements.
- Record hardware and software details with each result.
