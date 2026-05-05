# Week 20: RMSNorm Kernel

RMSNorm is a normalization layer used in many modern LLMs.

It looks close to LayerNorm, but it removes one major step:

```text
LayerNorm: subtract mean, divide by standard deviation
RMSNorm:   divide by root mean square
```

Week 20 teaches RMSNorm because it is simpler than LayerNorm and very practical
for transformer kernels.

## The RMSNorm Formula

For a row `x`, RMSNorm computes:

```text
rms = sqrt(mean(x_i^2) + eps)
out_i = x_i / rms * weight_i
```

There is no row mean.

There is no centering step:

```text
x_i - mean
```

The row is scaled by its root mean square.

## CPU Reference

Simple reference:

```python
def rmsnorm_row(xs, weight, eps=1e-6):
    mean_square = sum(x * x for x in xs) / len(xs)
    inv_rms = 1.0 / (mean_square + eps) ** 0.5
    return [x * inv_rms * weight[i] for i, x in enumerate(xs)]
```

The reduction is:

```text
sum(x_i * x_i)
```

The elementwise output is:

```text
x_i * inv_rms * weight_i
```

That is the whole forward pass.

## Hand Example

Input:

```text
[1, 2, 3, 4]
```

Squares:

```text
[1, 4, 9, 16]
```

Mean square:

```text
(1 + 4 + 9 + 16) / 4 = 7.5
```

Root mean square:

```text
sqrt(7.5) = 2.739
```

With all weights equal to 1, output is:

```text
[0.365, 0.730, 1.095, 1.461]
```

The row is scaled by its magnitude.

It is not shifted to have mean zero.

## Kernel Pipeline

The row-wise RMSNorm kernel shape is:

```text
load row values
square each value
reduce sum of squares
divide by width
compute inverse root mean square
multiply each input by inv_rms and weight
write output
```

Only one reduction is needed:

```text
sum of squares
```

That is why RMSNorm is simpler than LayerNorm.

## CUDA-Shaped Teaching Kernel

This teaching version assumes:

```text
width <= blockDim.x
one block handles one row
blockDim.x is a power of two
```

```cpp
__global__ void rmsnorm_row(
    const float* x,
    const float* weight,
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

    scratch[tid] = value * value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    float mean_square = scratch[0] / width;
    float inv_rms = rsqrtf(mean_square + eps);

    if (tid < width) {
        out[row * width + tid] = value * inv_rms * weight[tid];
    }
}
```

Read it as:

```text
one reduction
one row statistic
one output per input
```

## Step 1: Square Values

Each thread loads one value:

```cpp
float value = 0.0f;
if (tid < width) {
    value = x[row * width + tid];
}
```

Then contributes its square:

```cpp
scratch[tid] = value * value;
```

Inactive threads contribute zero.

That is safe for a sum of squares.

## Step 2: Reduce Sum Of Squares

The block reduction is familiar:

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
        scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
}
```

After the reduction:

```text
scratch[0] = sum of squares for the row
```

Then:

```cpp
float mean_square = scratch[0] / width;
```

## Step 3: Compute Inverse RMS

The scaling factor is:

```cpp
float inv_rms = rsqrtf(mean_square + eps);
```

`rsqrtf` computes reciprocal square root:

```text
1 / sqrt(mean_square + eps)
```

Using the reciprocal lets the final output use multiplication instead of
division.

## Step 4: Write Output

Each valid thread writes:

```cpp
out[row * width + tid] = value * inv_rms * weight[tid];
```

The weight is per feature position.

There is usually no beta term in basic RMSNorm.

That is another difference from LayerNorm.

## RMSNorm Vs LayerNorm

LayerNorm:

```text
mean reduction
variance reduction
subtract mean
scale by inverse standard deviation
gamma and beta
```

RMSNorm:

```text
sum-of-squares reduction
scale by inverse RMS
weight
```

RMSNorm is less work.

It also preserves the direction of the activation vector differently because it
does not subtract the mean.

For this course, the systems lesson is simple:

```text
removing one reduction can matter
```

## Why LLM Engineers Care

RMSNorm is common in LLM blocks because it is:

```text
simple
fast
stable enough for many architectures
easy to fuse with surrounding operations
```

It is a good kernel to understand before attention and transformer fusion.

The same engineering questions keep appearing:

```text
how many reductions?
how many reads?
how many writes?
what stays in registers?
what is shared across the row?
```

## The Core Pattern

When reading RMSNorm code, ask:

```text
What row is being normalized?
How is sum of squares computed?
Where is epsilon applied?
Is weight indexed by feature position?
Is the input value kept for the final write?
How does this differ from LayerNorm?
```

RMSNorm is a clean example of a practical ML kernel where a small math change
removes a meaningful amount of GPU work.

## Bridge To Week 21

Week 21 starts matrix multiplication.

Matmul is different from normalization:

```text
normalization is mostly row statistics and memory movement
matmul is mostly repeated multiply-add compute
```

The next lessons shift from row-wise reductions to the compute pattern behind
most deep learning layers.
