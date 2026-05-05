# Week 19: LayerNorm Kernel Mental Model

LayerNorm normalizes one row using statistics from that same row.

For each row, it computes:

```text
mean
variance
normalized value
optional scale and bias
```

Week 19 teaches LayerNorm as a kernel-shaped pipeline.

It is important because it combines:

```text
reductions for row statistics
elementwise math for outputs
memory reuse for performance
```

## What LayerNorm Does

Given a row:

```text
[2, 4, 6, 8]
```

The mean is:

```text
(2 + 4 + 6 + 8) / 4 = 5
```

The centered values are:

```text
[-3, -1, 1, 3]
```

The variance is:

```text
((-3)^2 + (-1)^2 + 1^2 + 3^2) / 4 = 5
```

The normalized values are:

```text
(x - mean) / sqrt(variance + eps)
```

So the output is roughly:

```text
[-1.342, -0.447, 0.447, 1.342]
```

The row now has mean near zero and variance near one.

## CPU Reference

A simple reference:

```python
def layernorm_row(xs, gamma=None, beta=None, eps=1e-5):
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    inv_std = 1.0 / (var + eps) ** 0.5

    out = []
    for i, x in enumerate(xs):
        y = (x - mean) * inv_std
        if gamma is not None:
            y *= gamma[i]
        if beta is not None:
            y += beta[i]
        out.append(y)

    return out
```

The optional `gamma` and `beta` are learned parameters:

```text
gamma: scale
beta: bias
```

The normalization happens first.

Scale and bias happen after.

## Kernel Pipeline

A row-wise LayerNorm kernel has this shape:

```text
load row values
reduce to row sum
compute mean
reduce squared differences to variance
compute inverse standard deviation
normalize each element
apply gamma and beta
write output
```

Two stages are reductions:

```text
sum for mean
sum of squared differences for variance
```

The final stage is elementwise:

```text
(x - mean) * inv_std * gamma + beta
```

That is why LayerNorm feels similar to softmax.

Both compute row statistics, then use those statistics for each output element.

## CUDA-Shaped Teaching Kernel

This teaching version assumes:

```text
width <= blockDim.x
one block handles one row
blockDim.x is a power of two
```

```cpp
__global__ void layernorm_row(
    const float* x,
    const float* gamma,
    const float* beta,
    float* out,
    int height,
    int width,
    float eps
) {
    extern __shared__ float scratch[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= height) {
        return;
    }

    float value = 0.0f;
    if (tid < width) {
        value = x[row * width + tid];
    }

    scratch[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    float mean = scratch[0] / width;

    float diff = 0.0f;
    if (tid < width) {
        diff = value - mean;
    }

    scratch[tid] = diff * diff;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    float variance = scratch[0] / width;
    float inv_std = rsqrtf(variance + eps);

    if (tid < width) {
        float y = diff * inv_std;
        y = y * gamma[tid] + beta[tid];
        out[row * width + tid] = y;
    }
}
```

This is not the most optimized LayerNorm.

It is the readable shape.

## Step 1: Sum The Row

Each thread loads one value:

```cpp
float value = 0.0f;
if (tid < width) {
    value = x[row * width + tid];
}
```

Then the block reduces those values:

```cpp
scratch[tid] = value;
__syncthreads();
```

After the sum reduction:

```cpp
float mean = scratch[0] / width;
```

The mean is shared by every output element in the row.

## Step 2: Sum Squared Differences

Variance needs the distance from the mean:

```cpp
float diff = value - mean;
```

Then each thread contributes:

```cpp
diff * diff
```

The block reduces those squared differences:

```text
sum((x - mean)^2)
```

Then divides by width:

```cpp
float variance = scratch[0] / width;
```

That gives the row variance.

## Step 3: Normalize

The inverse standard deviation is:

```cpp
float inv_std = rsqrtf(variance + eps);
```

The epsilon prevents division by zero or unstable values when variance is tiny.

Each valid thread writes:

```cpp
float y = diff * inv_std;
y = y * gamma[tid] + beta[tid];
out[row * width + tid] = y;
```

One row statistic.

Many elementwise outputs.

## Why Gamma And Beta Are Per Column

LayerNorm usually has one `gamma` and one `beta` per feature position.

For a row width of 4:

```text
gamma[0], gamma[1], gamma[2], gamma[3]
beta[0],  beta[1],  beta[2],  beta[3]
```

Every row reuses the same parameter vectors.

That means:

```text
row changes
feature position chooses gamma and beta
```

The parameter index is usually `tid` or column index, not the row index.

## Memory Traffic

A simple LayerNorm may read the row more than once.

An optimized kernel tries to keep useful values in registers or shared memory:

```text
keep x value for final normalization
reuse mean for variance
reuse variance for final output
write output once
```

The hard part is resource balance.

Keeping more values close can reduce global memory traffic, but it can increase
register or shared-memory pressure.

That is a recurring GPU engineering tradeoff.

## LayerNorm Vs Softmax

Softmax:

```text
row max
exp
row sum
divide
```

LayerNorm:

```text
row mean
row variance
normalize
scale and bias
```

Both are row-wise.

Both need reductions.

Both write one output per input.

Softmax produces probabilities that sum to one.

LayerNorm produces normalized activations.

## The Core Pattern

When reading LayerNorm code, ask:

```text
What is one normalization row?
How is the mean computed?
How is the variance computed?
Where is epsilon applied?
Are gamma and beta indexed by feature position?
How many times is the input row read?
Which values stay in registers or shared memory?
```

LayerNorm is one of the first kernels where math, memory, and model structure
all show up at the same time.

## Bridge To Week 20

Week 20 teaches RMSNorm.

RMSNorm is related to LayerNorm, but it removes the mean subtraction.

That makes it simpler and common in modern LLMs.
