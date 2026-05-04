# Week 04: Elementwise Kernel Patterns

Week 03 explained how tensor shapes map to flat memory.

Week 04 uses that indexing model to understand a small family of useful kernels:

```text
copy
scale
square
ReLU
add
axpy-shaped updates
```

These kernels are simple, but they matter because the same structure appears
inside larger deep learning systems.

The lesson is:

```text
Once the thread owns the right output element, the operation can change.
```

## The Elementwise Pattern

An elementwise operation computes each output element independently.

For one input:

```text
out[i] = f(x[i])
```

For two inputs:

```text
out[i] = f(a[i], b[i])
```

The important word is independently.

To compute `out[10]`, the kernel should not need `out[9]` or `out[11]`.

That independence makes elementwise operations a natural fit for GPU threads:

```text
thread 0 -> output element 0
thread 1 -> output element 1
thread 2 -> output element 2
...
```

Week 02 used this idea for vector add. Week 04 generalizes it.

## A Shared Kernel Skeleton

Most 1D elementwise kernels start with the same structure:

```cpp
__global__ void kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = /* one element of work */;
    }
}
```

The indexing part answers:

```text
Which output element does this thread own?
```

The operation part answers:

```text
What value should be written there?
```

Keep those two ideas separate in your head.

Indexing chooses the position.

The math chooses the value.

## Copy Kernel

The simplest elementwise kernel is copy:

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

This kernel does almost no math.

It reads one value and writes one value:

```text
read x[i]
write out[i]
```

That makes copy a useful baseline later when you start thinking about memory
bandwidth.

## Scale Kernel

Scale multiplies every input element by the same number:

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

The indexing did not change.

Only the operation changed:

```cpp
out[i] = alpha * x[i];
```

This is the pattern to notice. Many kernels differ by only one or two lines once
the indexing is correct.

## Square Kernel

Square uses one input and writes one output:

```text
out[i] = x[i] * x[i]
```

CUDA-shaped code:

```cpp
__global__ void square_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float value = x[i];
        out[i] = value * value;
    }
}
```

This version stores `x[i]` in a local variable:

```cpp
float value = x[i];
```

That makes the operation easier to read and avoids writing `x[i]` twice.

For a beginner kernel, clarity matters.

## ReLU Kernel

ReLU is common in neural networks:

```text
out[i] = max(x[i], 0)
```

CUDA-shaped code:

```cpp
__global__ void relu_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float value = x[i];
        out[i] = value > 0.0f ? value : 0.0f;
    }
}
```

This introduces a tiny branch:

```cpp
value > 0.0f ? value : 0.0f
```

Read it as:

```text
if value is positive, keep it
otherwise write zero
```

The operation is different from copy, scale, and square. The ownership pattern
is the same.

## Add Kernel

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

This is the Week 02 vector add kernel in its general form.

For each output element, the thread reads:

```text
a[i]
b[i]
```

And writes:

```text
out[i]
```

That read/write shape will become important in Week 05.

## Axpy-Shaped Kernel

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

This still follows the elementwise rule:

```text
one output element depends only on matching input elements
```

It is more interesting than add because it mixes multiplication and addition,
but it is still easy to reason about.

## Out-Of-Place And In-Place

Most beginner kernels should be written out-of-place:

```text
read x
write out
```

Example:

```cpp
out[i] = x[i] * x[i];
```

The input and output are different arrays.

An in-place kernel modifies the input array directly:

```cpp
x[i] = x[i] * x[i];
```

In-place can save memory, but it is easier to misuse.

For early kernels, out-of-place is easier to test:

```text
input stays unchanged
output contains the result
```

That makes correctness checks simpler.

## The Same Kernels On A Matrix

If the input is a `height x width` matrix, the operation is still elementwise.

For ReLU:

```text
out[row, col] = max(x[row, col], 0)
```

The 2D kernel uses the indexing from Week 03:

```cpp
__global__ void relu_matrix_kernel(const float* x, float* out, int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int i = row * width + col;
        float value = x[i];
        out[i] = value > 0.0f ? value : 0.0f;
    }
}
```

The operation is still just:

```cpp
out[i] = value > 0.0f ? value : 0.0f;
```

The extra work is converting the thread position into the correct memory index:

```cpp
int i = row * width + col;
```

## Shape Agreement

For two-input elementwise kernels, the input arrays must agree on shape.

For add:

```text
a shape   = height x width
b shape   = height x width
out shape = height x width
```

Then each element lines up:

```text
out[row, col] = a[row, col] + b[row, col]
```

Flattened:

```cpp
out[i] = a[i] + b[i];
```

If the shapes do not agree, `a[i]` and `b[i]` may not represent the same logical
position.

Broadcasting can handle some shape differences in frameworks like NumPy and
PyTorch, but a simple CUDA kernel does not get broadcasting for free. You must
write the indexing rules yourself.

## What Changes And What Stays Fixed

Across copy, scale, square, ReLU, add, and axpy, the stable parts are:

```text
compute the thread's output position
check that the position is valid
read the input values for that position
write one output value
```

The changing part is the formula:

```text
copy:   out[i] = x[i]
scale:  out[i] = alpha * x[i]
square: out[i] = x[i] * x[i]
ReLU:   out[i] = max(x[i], 0)
add:    out[i] = a[i] + b[i]
axpy:   out[i] = alpha * x[i] + y[i]
```

This is why elementwise kernels are a good learning step.

They let you practice the kernel structure without hiding the lesson behind
complicated math.

## The Mental Model

When reading an elementwise kernel, ask:

```text
What is the output shape?
Which output element does this thread own?
Which input elements does that output need?
Is the bounds check protecting the full shape?
Does this thread write exactly one output value?
```

For a correct beginner elementwise kernel, the answer should be simple:

```text
one valid thread writes one valid output element
```

Week 03 taught how to find the right memory address.

Week 04 teaches how many useful kernels reuse that same address and only change
the operation.
