# Skool Course Launch Post

## Title

GPU Kernel Engineer From Scratch: Course Launch + Weekly Challenge

## Post

Welcome to **GPU Kernel Engineer From Scratch**.

This course is for ML engineers who use PyTorch, CUDA, Triton, or AI systems work and want to actually understand what is happening underneath.

The goal is not to memorize random CUDA syntax.

The goal is to build the mental model and hands-on skill stack behind GPU kernels:

- what the CPU does vs what the GPU does
- host memory vs device memory
- what a CUDA kernel is
- how grids, blocks, threads, and warps work
- how one thread maps to one piece of data
- why bounds checks matter
- how tensors become flat memory
- row-major layout, strides, and indexing
- simple elementwise kernels like add, copy, scale, square, and ReLU
- memory bandwidth and why moving data can dominate performance
- benchmark basics: synchronization, warmups, repeats, median timing
- how to compare kernels against trusted CPU, NumPy, or PyTorch references
- reductions: sum, max, mean, row sum
- why reductions are harder than elementwise kernels
- shared memory reductions
- warp-level reductions
- how these ideas later connect to softmax, LayerNorm, matmul, Triton, attention, and transformer kernels

If you are coming from the YouTube video, the repo gives you the roadmap and the lessons.

The Skool community is where we do the work together:

- weekly implementation challenges
- debugging help
- code review
- benchmark review
- explanations in plain English
- accountability so you do not watch one video and disappear

This week we are starting with the two kernels from the video:

1. a simple CUDA addition kernel
2. one reduction kernel from the later part of the video

You do not need to make it perfect.

You need to make it real, run it, compare it, and post it.

## Weekly Challenge

Build and post two CUDA kernels:

### Part 1: Addition Kernel

Implement a simple elementwise addition kernel:

```text
C[i] = A[i] + B[i]
```

Your kernel should use the standard CUDA indexing pattern:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

And it must include a bounds check:

```cpp
if (i < n) {
    c[i] = a[i] + b[i];
}
```

Test sizes:

```text
N = 1
N = 17
N = 256
N = 1000
N = 1_000_000
```

The important test is `N = 1000`, because it does not divide cleanly by common block sizes like `256`.

That forces you to handle extra threads correctly.

### Part 2: Reduction Kernel

Implement one reduction kernel from the later part of the video.

Pick one:

- sum a 1D array into one value
- row sum for a 2D matrix
- block-level reduction using shared memory
- warp-level sum if you already know CUDA better

Minimum version:

```text
input:  [1, 2, 3, 4]
output: 10
```

Better version:

```text
input shape: [rows, cols]
output shape: [rows]
operation: each output is the sum of one row
```

The main idea:

```text
elementwise add: one input position writes one output position
reduction: many input positions contribute to fewer output positions
```

That is why reductions are harder and more interesting.

## What To Post In Skool

Post your submission with this format:

```text
Challenge: CUDA Addition + Reduction

Part 1: Addition Kernel
- code or repo link:
- sizes tested:
- correctness passed? yes/no:
- what does one thread compute?

Part 2: Reduction Kernel
- which reduction did you implement?
- code or repo link:
- input shape:
- output shape:
- correctness passed? yes/no:
- where do threads cooperate?

Benchmark
- did you time it? yes/no:
- timing method:
- warmups/repeats if used:
- CPU/NumPy/PyTorch reference used:

Question
- what confused you most?
- what do you want reviewed?
```

## What We Will Review

When you post, we will check:

- does the addition kernel have correct indexing?
- does it handle sizes that are not divisible by block size?
- does it avoid out-of-bounds reads and writes?
- does the reduction compare against a trusted reference?
- does the reduction avoid obvious race conditions?
- does the benchmark include synchronization or a clear timing method?
- can you explain the code in your own words?

Broken submissions are allowed.

If your kernel fails, post:

- the code
- the input size
- the expected output
- the actual output
- the error message
- what you already tried

That is exactly what the community is for.

## Why This Matters

These two kernels teach the core split in GPU programming:

```text
addition kernel -> independent outputs
reduction kernel -> cooperating threads
```

Most AI kernels are built from these ideas:

- map threads to data
- guard memory bounds
- read and write flat memory correctly
- compare against a trusted reference
- synchronize when threads cooperate
- measure performance honestly

If you can build and explain these two kernels, you have the first real foundation for GPU kernel engineering.

Post your work in Skool this week.

We will review it, help debug it, and use it as the starting point for the next GPU kernels challenge.
