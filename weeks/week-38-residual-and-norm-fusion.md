# Week 38: Residual And Norm Fusion

Week 20 already taught RMSNorm as a standalone normalization kernel.

Week 38 teaches the next engineering question:

```text
what happens when normalization sits next to residual adds and other tensor ops?
```

This is where transformer kernels start to feel like systems work.

## Step 1: Locate The Pattern

A transformer block often has residual structure:

```python
x = x + attention_out
y = norm(x)
```

or:

```python
x = x + mlp_out
y = norm(x)
```

Written as separate operations, this can mean:

```text
kernel 1: residual add
kernel 2: normalization
```

The output of the add is written to memory, then read again by normalization.

Fusion asks whether that intermediate write can be avoided.

## Step 2: Separate Elementwise And Row-Wise Work

Residual add is elementwise:

```text
tmp[row, col] = x[row, col] + residual[row, col]
```

Normalization is row-wise:

```text
mean or variance is computed across columns
each output uses row-level statistics
```

That difference matters.

Fusion is easy when both operations are elementwise.

Fusion is more complex when one operation needs a row reduction.

## Step 3: Write The Reference Shape

A clean reference for residual plus RMSNorm is:

```python
def residual_rmsnorm(x, residual, weight, eps=1e-6):
    tmp = x + residual
    variance = tmp.pow(2).mean(dim=-1, keepdim=True)
    normalized = tmp * torch.rsqrt(variance + eps)
    return normalized * weight
```

Read the dataflow:

```text
add residual
compute row variance
scale row
apply weight
```

The fused kernel must preserve this order.

## Step 4: Understand The Two-Pass Shape

For each row, normalization needs a statistic.

That creates two conceptual passes:

```text
pass 1: read row, add residual, accumulate sum of squares
pass 2: read row again or reuse values, scale and store output
```

If the row fits in one program or block, the kernel may keep intermediate
values in registers.

If the row is too wide, the implementation may need multiple programs,
temporary storage, or a more advanced reduction strategy.

## Step 5: Know What Gets Reused

In a fused residual norm, the temporary value is:

```text
tmp = x + residual
```

That value is needed for:

```text
variance computation
final normalized output
```

The engineering question is:

```text
can tmp stay close to the computation instead of being written to global memory?
```

For moderate row widths, a Triton program can often load a row block, compute
`tmp`, reduce it, and store the final output.

## Step 6: Track Precision

Normalization is sensitive to accumulation precision.

Even if inputs are `float16`, reductions often accumulate in `float32`:

```python
tmp_f32 = tmp.float()
variance = tmp_f32.pow(2).mean(dim=-1, keepdim=True)
```

The output may return to the input dtype.

This is normal.

The baseline and custom kernel should agree on the intended precision behavior.

## Step 7: Compare Fusion To The Unfused Version

The unfused path writes `tmp`:

```text
read x
read residual
write tmp
read tmp for norm
write output
```

The fused path tries to do:

```text
read x
read residual
compute row statistic
write output
```

It may still read values more than once internally, but it avoids exposing the
residual-add output as a full global-memory tensor.

That is the memory benefit.

## The Core Pattern

For residual plus norm fusion:

```text
identify the temporary tensor
ask whether the next operation needs a row statistic
compute the temporary value close to the reduction
accumulate statistics in stable precision
scale and store the final output
compare to the unfused PyTorch reference
benchmark only after correctness passes
```

This is a more realistic fusion pattern than bias plus activation because it
mixes elementwise work with row-wise reduction.

## Bridge To Week 39

Week 39 starts attention from its smallest pieces: QK scores and masking.

The same dataflow habit will matter again, because attention is mostly about
which tensors are read, which intermediate values are materialized, and which
ones can be avoided.
