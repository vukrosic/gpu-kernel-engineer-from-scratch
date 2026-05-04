# Week 10: Naive Reduction Kernels

Week 09 introduced reductions:

```text
many input values -> fewer output values
```

Week 10 teaches the first implementation shape:

```text
one worker computes one complete reduction output
```

This is not the fastest reduction strategy.

It is the clearest first strategy.

Before learning shared memory or warp-level cooperation, you need to understand
the simplest correct version.

## The Naive Row Sum Idea

For a matrix:

```text
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
]
```

Row sum produces:

```text
[10, 26]
```

A naive GPU mapping is:

```text
one thread computes one row sum
```

So:

```text
thread 0 -> sum row 0 -> out[0]
thread 1 -> sum row 1 -> out[1]
```

Inside each thread, the work is serial:

```text
load value 0
add value 1
add value 2
add value 3
write result
```

This is simple and correct, but it does not use many threads inside a row.

That is why it is called naive.

## CPU-Shaped Reference

The reference is the trusted answer.

Python-shaped row sum:

```python
def row_sum_reference(x):
    rows = len(x)
    cols = len(x[0])
    out = []

    for row in range(rows):
        total = 0.0
        for col in range(cols):
            total += x[row][col]
        out.append(total)

    return out
```

This code makes the reduction shape visible:

```text
outer loop chooses the output row
inner loop reduces across columns
```

The inner loop is where many input values become one output value.

## Flat Memory Version

CUDA kernels often receive a flat pointer, not a nested list.

For a row-major matrix:

```text
index = row * width + col
```

The same reference can be written with flat indexing:

```python
def row_sum_flat(x, height, width):
    out = []

    for row in range(height):
        total = 0.0
        for col in range(width):
            i = row * width + col
            total += x[i]
        out.append(total)

    return out
```

This is closer to the kernel shape.

It combines Week 03 indexing with Week 09 reductions.

## Naive CUDA-Shaped Row Sum

A naive CUDA-shaped row sum looks like this:

```cpp
__global__ void row_sum_naive(
    const float* x,
    float* out,
    int height,
    int width
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height) {
        float total = 0.0f;

        for (int col = 0; col < width; ++col) {
            int i = row * width + col;
            total += x[i];
        }

        out[row] = total;
    }
}
```

Read it in three parts.

First, choose the output row:

```cpp
int row = blockIdx.x * blockDim.x + threadIdx.x;
```

Second, reduce across columns:

```cpp
for (int col = 0; col < width; ++col) {
    int i = row * width + col;
    total += x[i];
}
```

Third, write one output:

```cpp
out[row] = total;
```

One thread owns one row output.

That thread loops through the row.

## Why The Bounds Check Changed

In elementwise kernels, the bounds check usually protects an element index:

```cpp
if (i < n) {
```

In naive row sum, the thread owns a row:

```cpp
if (row < height) {
```

The valid output positions are:

```text
out[0] through out[height - 1]
```

So the bounds check protects the output row.

Inside the loop, `col` is controlled by:

```cpp
col < width
```

Together, those two checks keep the row and column valid.

## Naive Row Max

Row max has the same shape as row sum.

The operation changes from addition to maximum.

CUDA-shaped code:

```cpp
__global__ void row_max_naive(
    const float* x,
    float* out,
    int height,
    int width
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height) {
        float best = x[row * width];

        for (int col = 1; col < width; ++col) {
            int i = row * width + col;
            best = best > x[i] ? best : x[i];
        }

        out[row] = best;
    }
}
```

The initial value is different.

For sum, use:

```cpp
float total = 0.0f;
```

For max, use the first value in the row:

```cpp
float best = x[row * width];
```

That avoids a bad initial value when the row contains negative numbers.

## Output Shape

For an input matrix:

```text
height x width
```

Naive row sum writes:

```text
height output values
```

The output index is the row:

```cpp
out[row] = total;
```

This is why one thread per row is a natural first mapping.

It matches the output shape:

```text
one output value per row
```

## Memory Access Pattern

For row-major input, each row is contiguous.

Inside one thread:

```text
x[row * width + 0]
x[row * width + 1]
x[row * width + 2]
x[row * width + 3]
```

Those addresses are adjacent.

That part is friendly.

But the GPU has another issue:

```text
one thread is doing a lot of serial work
```

If `width` is large, each thread loops through many columns alone.

The memory access can be simple while the parallelism is still weak.

## Why Naive Can Be Slow

The naive row sum is correct, but it has limits.

If a row has `4096` columns:

```text
one thread performs 4096 loads and additions
```

If there are only a few rows, the kernel may not launch enough useful parallel
work.

The problem is:

```text
parallelism happens across rows, not inside each row
```

For small `height` and large `width`, that is a poor match for the GPU.

Later reduction kernels will split one row across multiple threads.

Week 10 keeps the simple version so the output ownership is obvious.

## Naive Does Not Mean Useless

Naive kernels are valuable because they establish a baseline.

A baseline tells you:

```text
what correct output looks like
how simple the first implementation can be
what later optimizations must beat
```

Do not skip the naive version.

Optimized reductions are easier to understand when you can compare them against
the simple one-thread-per-output version.

## Correctness Edge Cases

For row reductions, check:

```text
height = 1
width = 1
negative values
large width
non-square matrices
rows with equal max values
```

For sum, floating-point order can matter slightly:

```text
(a + b) + c may not equal a + (b + c) exactly
```

So comparisons often use a tolerance:

```text
close enough within floating-point error
```

For max, the value should match exactly for normal finite inputs.

## Reading Existing Reduction Files

When you open a reduction implementation, look for:

```text
which axis is reduced
which thread owns each output
where the accumulator is initialized
how input indices are computed
where the final output is written
what reference checks correctness
```

This keeps the code readable even before it becomes fast.

## The Mental Model

When reading a naive reduction kernel, ask:

```text
What output element does this thread own?
Which input values contribute to that output?
What loop combines those values?
What is the initial accumulator value?
Where is the final result written?
Why is this correct but not fully parallel?
```

For naive row sum:

```text
thread owns: one row
inputs:      all columns in that row
operation:   addition
initial:     0
output:      out[row]
limit:       one thread loops through the whole row
```

The real lesson of Week 10 is:

```text
the simplest correct reduction gives one worker one output and lets that worker loop
```

Week 11 can then teach how multiple workers cooperate on the same output.
