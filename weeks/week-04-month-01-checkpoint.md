# Week 04: Month 01 Checkpoint

Week 04 is a checkpoint lesson.

The goal is not to introduce a new kernel. The goal is to make the first month
click as one connected story:

```text
CPU vs GPU mental model
-> vector add
-> grids, blocks, threads, and indexing
-> correctness before optimization
```

Month 01 is small on purpose. Before writing faster kernels, you need a clean
model of what a GPU kernel is doing.

## The Month 01 Story

A CPU and a GPU are both processors, but they are designed for different shapes
of work.

A CPU is strong at complex control flow:

```text
do this
then branch
then wait for data
then call another function
then handle an exception
```

A GPU is strong when the same kind of work can be applied to many pieces of data:

```text
do this operation for element 0
do this operation for element 1
do this operation for element 2
do this operation for element 3
...
```

That is why the first kernel in the course is vector add:

```text
out[i] = a[i] + b[i]
```

It is simple enough to understand fully, but real enough to show the structure
of GPU programming.

## What Vector Add Actually Teaches

Vector add is not important because addition is hard.

It is important because it teaches the basic kernel pattern:

```text
one thread owns one output element
```

For a vector with four elements:

```text
a   = [1, 2, 3, 4]
b   = [5, 6, 7, 8]
out = [6, 8, 10, 12]
```

The work can be split cleanly:

```text
thread 0 -> out[0] = a[0] + b[0]
thread 1 -> out[1] = a[1] + b[1]
thread 2 -> out[2] = a[2] + b[2]
thread 3 -> out[3] = a[3] + b[3]
```

No thread needs to know what the other threads are doing.

That independence is what makes vector add a good first GPU kernel.

## The Kernel Shape

Most early elementwise CUDA kernels have this shape:

```cpp
__global__ void kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = /* one element of work */;
    }
}
```

This is the first major pattern of the course.

The index line decides which data the thread owns:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

The bounds check keeps extra threads safe:

```cpp
if (i < n) {
```

The body performs one unit of work:

```cpp
out[i] = /* result for this element */;
```

Later kernels become more interesting, but this pattern does not disappear. It
keeps showing up in activation functions, normalization, preprocessing,
attention helpers, and many small pieces of deep learning workloads.

## Why Indexing Is A Correctness Concept

Indexing is easy to underestimate.

This line looks small:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

But it answers the most important question in the kernel:

```text
Which output element does this thread own?
```

If two threads write the same output index, the result may depend on timing.

If a thread writes outside the output array, the program may corrupt memory.

If the row-major formula is wrong, a matrix-shaped result may look scrambled.

So indexing is not just a performance detail. Indexing is part of the meaning of
the program.

## References Come First

A GPU kernel should not be trusted just because it compiles.

The safer workflow is:

```text
write the CPU or NumPy reference
write the GPU kernel
compare the GPU output against the reference
only then care about speed
```

A reference implementation is usually boring:

```python
def vector_add_reference(a, b):
    return a + b
```

That boring function is valuable because it gives you the expected answer.

The GPU version has more ways to fail:

```text
wrong index formula
missing bounds check
wrong launch size
wrong memory copy
wrong dtype
wrong shape assumption
```

The reference keeps the kernel honest.

## Tests And Benchmarks Answer Different Questions

Tests answer:

```text
Did the code produce the expected result?
```

Benchmarks answer:

```text
How long did the code take?
```

They are related, but they are not the same.

A test can pass while the implementation is slow.

A benchmark can be fast while the implementation is wrong.

That is why the course keeps both ideas separate:

```text
correctness first
measurement second
optimization third
```

This order matters. Optimizing a wrong kernel only makes the wrong answer arrive
faster.

## What Month 01 Gives You

By the end of Month 01, the important idea is not "I know all of CUDA."

The important idea is:

```text
I can read a simple kernel and explain how threads map to data.
```

That includes:

```text
what a thread is
what a block is
what a grid is
why the index formula exists
why bounds checks exist
why references exist
why tests and benchmarks both matter
```

This is the foundation for the rest of the course.

## The Mental Model To Keep

When reading a new kernel, ask four questions:

```text
1. What output is this kernel trying to compute?
2. Which thread owns which output element?
3. What prevents invalid memory access?
4. What reference result proves this kernel is correct?
```

For vector add, the answers are simple:

```text
1. out[i] = a[i] + b[i]
2. i = blockIdx.x * blockDim.x + threadIdx.x
3. if (i < n)
4. NumPy or CPU vector add
```

That same reading habit will make harder kernels less mysterious.

## Why This Checkpoint Matters

GPU programming can become noisy quickly:

```text
CUDA syntax
memory copies
launch configuration
thread hierarchy
benchmark timing
hardware details
```

The checkpoint compresses the noise into a small set of durable ideas.

Month 01 is not about writing impressive kernels yet. It is about being able to
look at a simple kernel and say:

```text
I understand what work is being done,
which thread does each piece of work,
how the output is checked,
and why the result can be trusted.
```

That is enough to move into memory movement, bandwidth, and performance with a
much clearer head.
