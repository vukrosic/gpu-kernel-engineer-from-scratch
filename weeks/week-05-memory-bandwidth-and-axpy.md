# Week 05: Memory Bandwidth And AXPY

Week 04 showed that many useful kernels share the same elementwise pattern:

```text
compute index
check bounds
read inputs
write output
```

Week 05 asks a different question:

```text
How much data does the kernel move?
```

This is where GPU performance starts to make more sense. A kernel can do very
little math and still take time because it has to move many bytes through
memory.

## The Core Idea

GPUs are fast at math, but math is not the only cost.

Every elementwise kernel moves data:

```text
read from input memory
write to output memory
```

For a huge array, that memory movement can dominate the runtime.

That is what memory bandwidth measures:

```text
how many bytes can move per second
```

If a kernel mostly reads and writes arrays, it may be limited by bandwidth
instead of arithmetic.

## Copy Is The Simplest Memory Kernel

A copy kernel does almost no math:

```text
out[i] = x[i]
```

CUDA-shaped code:

```cpp
__global__ void copy_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = x[i];
    }
}
```

For each element, the kernel does:

```text
read  one float from x
write one float to out
```

A `float` is 4 bytes.

So each element moves:

```text
4 bytes read + 4 bytes written = 8 bytes
```

For `1,000,000` elements:

```text
1,000,000 * 8 bytes = 8,000,000 bytes
```

That is about 8 MB of memory traffic.

The copy kernel is useful because it shows the cost of moving data with almost
no math mixed in.

## Scale Adds A Tiny Amount Of Math

Scale multiplies each element by a scalar:

```text
out[i] = alpha * x[i]
```

CUDA-shaped code:

```cpp
__global__ void scale_kernel(const float* x, float* out, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = alpha * x[i];
    }
}
```

For each element, the kernel does:

```text
read  x[i]
multiply by alpha
write out[i]
```

The memory traffic is still:

```text
4 bytes read + 4 bytes written = 8 bytes
```

The math added one multiplication.

For many large arrays, that multiplication is cheap compared with the memory
traffic.

This is the first important performance lesson:

```text
Adding a small amount of math does not always change the bottleneck.
```

## Add Reads More Data

Add uses two input arrays:

```text
out[i] = a[i] + b[i]
```

CUDA-shaped code:

```cpp
__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
```

For each element, the kernel does:

```text
read  a[i]
read  b[i]
write out[i]
```

For `float32`, that is:

```text
4 bytes + 4 bytes + 4 bytes = 12 bytes
```

The arithmetic is one addition.

The memory movement is three array accesses.

That ratio is why simple elementwise kernels are often memory-bound.

## AXPY

AXPY means:

```text
a times x plus y
```

The formula is:

```text
out[i] = alpha * x[i] + y[i]
```

CUDA-shaped code:

```cpp
__global__ void axpy_kernel(
    const float* x,
    const float* y,
    float* out,
    float alpha,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = alpha * x[i] + y[i];
    }
}
```

For each element, AXPY does:

```text
read  x[i]
read  y[i]
multiply x[i] by alpha
add y[i]
write out[i]
```

For `float32`, the memory traffic is:

```text
4 bytes for x[i]
4 bytes for y[i]
4 bytes for out[i]
= 12 bytes per element
```

The math is:

```text
1 multiply
1 add
```

AXPY is a good teaching kernel because it has more math than add, but it still
moves enough data that memory bandwidth matters.

## Bytes Moved

A useful habit is to estimate bytes moved before thinking about speed.

For `float32` arrays:

```text
one element = 4 bytes
```

Copy:

```text
read x + write out
= 4 + 4
= 8 bytes per element
```

Scale:

```text
read x + write out
= 4 + 4
= 8 bytes per element
```

Add:

```text
read a + read b + write out
= 4 + 4 + 4
= 12 bytes per element
```

AXPY:

```text
read x + read y + write out
= 4 + 4 + 4
= 12 bytes per element
```

For `n` elements:

```cpp
bytes_moved = bytes_per_element * n;
```

Example for AXPY with `1,000,000` elements:

```text
12 bytes * 1,000,000 = 12,000,000 bytes
```

That is about 12 MB.

## Bandwidth

Bandwidth is bytes moved divided by time.

```text
bandwidth = bytes_moved / seconds
```

If AXPY moves 12 MB and takes `0.0001` seconds:

```text
bandwidth = 12 MB / 0.0001 s
bandwidth = 120,000 MB/s
bandwidth = 120 GB/s
```

The exact number depends on hardware, timing method, array size, and whether the
measurement includes memory allocation or copies.

The mental model is more important than the exact number right now:

```text
large arrays make memory movement visible
small arrays can be dominated by overhead
```

## Arithmetic Intensity

Arithmetic intensity means:

```text
how much math happens per byte moved
```

Copy has almost no arithmetic:

```text
0 floating-point operations
8 bytes moved per element
```

Scale has:

```text
1 multiply
8 bytes moved per element
```

AXPY has:

```text
1 multiply + 1 add
12 bytes moved per element
```

These are low-intensity kernels.

They move a lot of memory compared with the amount of math they do.

Low arithmetic intensity often means:

```text
memory bandwidth is the bottleneck
```

High arithmetic intensity means:

```text
math throughput may become the bottleneck
```

Matrix multiplication is a classic higher-intensity kernel because each loaded
value can be reused for many multiply-add operations. Elementwise kernels
usually do not have that kind of reuse.

## Why Bigger Arrays Matter

For tiny arrays, runtime can be dominated by overhead:

```text
kernel launch overhead
Python overhead
framework dispatch
memory allocation
timing noise
```

For large arrays, the cost of moving data becomes easier to see.

That is why performance experiments often use large inputs and repeat the same
operation many times.

The goal is not to make the benchmark look impressive.

The goal is to measure the part of the work you actually care about.

## What A Benchmark Should Include Later

A serious GPU timing helper eventually needs:

```text
warmup runs
repeated measurements
GPU synchronization
stable input allocation
clear reporting of bytes moved
```

Warmups help avoid measuring one-time setup.

Repeats help reduce noise.

Synchronization matters because GPU work is often asynchronous. The CPU can
launch work and continue before the GPU has actually finished.

Stable allocation matters because memory allocation can hide the true kernel
cost.

Week 07 will go deeper into timing. For Week 05, focus on the concept:

```text
runtime only makes sense when you know how much data moved
```

## Why This Comes Before Coalescing

Week 06 is about contiguous and strided memory access.

Before you can understand why access pattern matters, you need to understand
that memory movement itself is expensive.

Week 05 teaches:

```text
how many bytes move
```

Week 06 teaches:

```text
how those bytes are accessed
```

Both matter.

A kernel can move the same number of bytes but run differently because one
access pattern is friendlier to the hardware.

## The Mental Model

When reading a simple performance kernel, ask:

```text
How many arrays are read?
How many arrays are written?
How many bytes does each element use?
How much math happens per element?
Is this likely limited by memory or math?
```

For copy, scale, add, and AXPY, the answer is usually:

```text
this kernel is mostly about moving memory
```

That is the real lesson of Week 05.

The operation line may look like math, but the runtime often comes from the
bytes traveling to and from memory.
