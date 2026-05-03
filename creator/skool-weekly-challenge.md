# Skool Weekly Challenge

## Title

Weekly Challenge: Build Your First GPU Kernel

## Post

This week we are doing the first Skool GPU Kernels challenge:

Build and explain a correct `vector_add` kernel.

This is the right first challenge because vector add is simple enough to finish, but still teaches the core GPU ideas:

- host vs device memory
- launching a kernel
- grids, blocks, and threads
- thread indexing
- bounds checks
- correctness testing
- basic benchmarking

The goal is not to write the most optimized kernel.

The goal is to prove you understand how one simple operation gets split across many GPU threads.

## Challenge

Implement vector addition:

```text
C[i] = A[i] + B[i]
```

for arrays of different sizes.

Your kernel must work when the input size is not divisible by the block size.

That means this should work:

```text
N = 1000
block_size = 256
```

The last block will launch extra threads, and your kernel must avoid reading or writing out of bounds.

## Tasks

1. Run the project setup and baseline tests.
2. Implement a CUDA `vector_add` kernel.
3. Use this indexing pattern:

```text
i = blockIdx.x * blockDim.x + threadIdx.x
```

4. Add a bounds check:

```text
if i < N
```

5. Compare your CUDA output against a CPU, NumPy, or PyTorch reference.
6. Test at least these sizes:

```text
N = 1
N = 17
N = 256
N = 1000
N = 1_000_000
```

7. Benchmark CPU vs GPU for at least three sizes.
8. Write a short explanation of what each thread does.

## Submission

Post your submission in Skool with this format:

```text
Challenge: First GPU Kernel

Repo or code:

Correctness:
- sizes tested:
- all passed? yes/no:

Benchmark:
- CPU time:
- GPU time:
- input size:

Explanation:
- What does one thread compute?
- Why do we need the bounds check?
- What confused me:
- What I want reviewed:
```

You can submit even if it is not perfect.

The point of Skool is that members can have this checked, debugged, and reviewed.

## What We Will Check

When you post your challenge, we will look for:

- does the kernel compute the right result?
- does it handle odd input sizes?
- does it have a correct bounds check?
- does the benchmark avoid obvious timing mistakes?
- can you explain the indexing formula?
- do you know what part you are unsure about?

## Common Mistakes

Mistake 1:

Only testing `N = 1024`.

Why this is a problem:

That size hides indexing bugs because it divides nicely into common block sizes.

Mistake 2:

Forgetting the bounds check.

Why this is a problem:

The final block often has more threads than remaining elements.

Mistake 3:

Benchmarking only one tiny input.

Why this is a problem:

For small arrays, launch overhead can dominate and make the GPU look worse.

Mistake 4:

Saying "the GPU is faster" without showing the input size.

Why this is a problem:

Performance claims are meaningless without shape, dtype, device, and timing method.

## Questions To Answer

Answer these in your post:

- What does `blockIdx.x` mean?
- What does `threadIdx.x` mean?
- Why do we multiply `blockIdx.x` by `blockDim.x`?
- What happens if `N` is not divisible by the block size?
- At what input size did the GPU start to look useful?
- What is one thing you still do not fully understand?

## Minimum Version

If you are busy, submit the minimum version:

- one correct `vector_add` kernel
- one odd-size test
- one benchmark result
- three sentences explaining the indexing

That is enough to participate.

## Stretch Version

If you want extra practice:

- test multiple dtypes
- compare block sizes like 128, 256, 512
- add a PyTorch baseline
- plot input size vs runtime
- write the same kernel in Triton and compare the code

## Why This Challenge Matters

Almost every later GPU kernel uses the same foundation:

- map threads to data
- avoid out-of-bounds access
- compare against a trusted reference
- benchmark honestly
- explain what the hardware is doing

If you can do that for vector add, you have the first brick.

Next challenges will build from here:

- elementwise kernels
- memory bandwidth
- reductions
- softmax
- matmul
- Triton kernels

Start simple.

Make it correct.

Post it for review.
