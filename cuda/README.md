# CUDA Track

This folder holds the standalone CUDA C++ starter kernels used by the course.
They are written as readable reference implementations, not highly tuned code.

## Files

- `vector_add.cu`
- `reduce_sum.cu`
- `softmax.cu`
- `naive_matmul.cu`

## Compile

Use `nvcc` on an NVIDIA machine:

```bash
nvcc -O2 -std=c++14 cuda/vector_add.cu -o build/vector_add
nvcc -O2 -std=c++14 cuda/reduce_sum.cu -o build/reduce_sum
nvcc -O2 -std=c++14 cuda/softmax.cu -o build/softmax
nvcc -O2 -std=c++14 cuda/naive_matmul.cu -o build/naive_matmul
```

## Learning Order

1. `vector_add.cu`
2. `reduce_sum.cu`
3. `softmax.cu`
4. `naive_matmul.cu`

## Rules

- Write a CPU or NumPy reference first.
- Test correctness before benchmarking.
- Benchmark with warmup and repeated measurements.
- Record hardware and software details with each result.
