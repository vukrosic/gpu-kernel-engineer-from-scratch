# Week 18: Fused Row-Wise Softmax

Week 17 taught softmax as a stable math pipeline:

```text
row max -> exp(x - row max) -> row sum -> divide
```

Week 18 teaches the kernel question:

```text
how do you do those steps without moving the row through global memory more
than necessary?
```

That is what fusion means here.

Fusion does not change the math.

Fusion changes how much intermediate data you write and reread.

## Unfused Softmax

A very literal implementation might split softmax into separate passes:

```text
kernel 1: compute row max
kernel 2: compute shifted exponentials
kernel 3: compute row sum
kernel 4: divide exponentials by row sum
```

This is easy to reason about.

It can also move a lot of data:

```text
read scores
write row max
read scores again
write exponentials
read exponentials
write row sum
read exponentials again
write output
```

The math is correct.

The memory traffic is expensive.

## Fused Softmax

A fused row-wise softmax tries to keep the work together:

```text
one kernel handles one row or one row tile
```

Inside that kernel:

```text
load row values
reduce to row max
compute exponentials
reduce to row sum
write normalized outputs
```

The goal is to avoid writing intermediate exponentials to global memory unless
you really need to.

The kernel still does two reductions:

```text
max reduction
sum reduction
```

The difference is that the row stays close to the threads doing the work.

## CPU-Shaped Reference

This reference keeps the stable softmax math clear:

```python
import math

def softmax_row(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]
```

That is the answer a GPU kernel must match.

The GPU version can change the execution strategy.

It cannot change the result.

## Kernel-Shaped Pipeline

The row-wise kernel shape is:

```text
one block owns one row
threads cooperate across columns
```

For a row of 8 values:

```text
thread 0 loads x[0]
thread 1 loads x[1]
...
thread 7 loads x[7]
```

Then the block cooperates:

```text
reduce max
compute exp values
reduce sum
normalize each value
```

This should feel like a combination of earlier weeks:

```text
Week 11: block-level reductions
Week 12: warp-level reductions
Week 13: synchronization
Week 17: stable softmax math
```

## CUDA-Shaped Teaching Kernel

This teaching version assumes:

```text
width <= blockDim.x
one block handles one row
blockDim.x is a power of two
```

```cpp
__global__ void softmax_row_fused(
    const float* x,
    float* out,
    int height,
    int width
) {
    extern __shared__ float scratch[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= height) {
        return;
    }

    float value = -INFINITY;
    if (tid < width) {
        value = x[row * width + tid];
    }

    scratch[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = scratch[0];

    float exp_value = 0.0f;
    if (tid < width) {
        exp_value = expf(x[row * width + tid] - row_max);
    }

    scratch[tid] = exp_value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = scratch[0];

    if (tid < width) {
        out[row * width + tid] = exp_value / row_sum;
    }
}
```

Read it as a lesson, not a final optimized kernel.

Real softmax kernels often use warp shuffles, vectorized loads, masks, and more
careful storage.

The point here is the fused structure.

## Step 1: Load Values For Max

Softmax must start by finding the row max:

```cpp
float value = -INFINITY;
if (tid < width) {
    value = x[row * width + tid];
}
```

Inactive threads use negative infinity.

That matters because max reduction should ignore inactive positions:

```text
max(real value, negative infinity) = real value
```

Using zero would be wrong for rows where every real value is negative.

## Step 2: Reduce To Row Max

The max reduction is the same tree shape as earlier reductions:

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
        scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
}
```

After this loop:

```text
scratch[0] holds the row max
```

Every thread can read:

```cpp
float row_max = scratch[0];
```

The row max is shared by the whole row.

## Step 3: Compute Exponentials

Each valid thread computes one shifted exponential:

```cpp
float exp_value = 0.0f;
if (tid < width) {
    exp_value = expf(x[row * width + tid] - row_max);
}
```

This is the stable softmax trick from Week 17.

The important detail is that `exp_value` stays in a register.

The kernel also stores it in shared memory because the block needs to reduce
the sum:

```cpp
scratch[tid] = exp_value;
```

## Step 4: Reduce To Row Sum

The second reduction sums the exponentials:

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
        scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
}
```

After this loop:

```text
scratch[0] holds the sum of shifted exponentials
```

That value is the denominator for the row.

## Step 5: Normalize And Write

Each valid thread writes one output:

```cpp
if (tid < width) {
    out[row * width + tid] = exp_value / row_sum;
}
```

Unlike a reduction, softmax writes one value per input position.

The reductions produce shared row statistics.

The final division is elementwise.

## Why This Is Fused

This kernel keeps the whole row pipeline together:

```text
load scores
compute row max
compute exponentials
compute row sum
write probabilities
```

The intermediate `exp_value` is not written to global memory.

That is the memory win.

The kernel still reads the input row twice in this simple teaching version:

```text
once for max
once for exp
```

A more advanced version may keep values in registers or shared memory to reduce
that reread, depending on row size and resource pressure.

Fusion is always a tradeoff:

```text
less global memory traffic
more work inside one kernel
more pressure on registers/shared memory
```

## What To Watch For

Fused softmax can fail in several ways:

```text
using zero instead of negative infinity for inactive max lanes
forgetting the max shift
dividing by the wrong row sum
mixing values from different rows
assuming every row fits in one block
writing intermediate values to global memory unnecessarily
```

The first job is correctness.

Then measure whether fusion helped.

## The Core Pattern

When reading fused softmax, ask:

```text
What does one block or warp own?
Where is the row max computed?
Where are exponentials stored?
Where is the row sum computed?
How many times is global memory read or written?
Which values stay in registers or shared memory?
What assumptions are made about row width?
```

Softmax is a normalization kernel.

Most of its performance comes from how well the kernel moves and reuses row
data.

## Bridge To Week 19

Week 19 moves from softmax to LayerNorm.

LayerNorm also computes row-level statistics and then writes one output per
element.

The statistics change:

```text
softmax: max and sum
LayerNorm: mean and variance
```
