# Week 28: Triton Row-Wise Softmax

Week 28 puts several ideas together:

```text
program instance
offset block
mask
row max reduction
row sum reduction
elementwise divide
```

The result is a Triton-shaped row-wise softmax.

## One Program Per Row

For the teaching version:

```text
one program handles one row
```

The row is:

```python
row = tl.program_id(0)
```

The columns are:

```python
cols = tl.arange(0, BLOCK_SIZE)
```

The mask is:

```python
mask = cols < width
```

This assumes:

```text
width <= BLOCK_SIZE
```

That keeps the lesson focused on the softmax pipeline.

## Triton Softmax Shape

The kernel shape:

```python
@triton.jit
def softmax_kernel(x_ptr, out_ptr, height, width, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < width

    values = tl.load(
        x_ptr + row * width + cols,
        mask=mask,
        other=-float("inf"),
    )

    row_max = tl.max(values, axis=0)
    shifted = values - row_max
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    tl.store(out_ptr + row * width + cols, result, mask=mask)
```

This is stable softmax from Week 17, expressed as one Triton program per row.

## Why The Mask Uses Negative Infinity

The first reduction is max:

```python
row_max = tl.max(values, axis=0)
```

Invalid positions must not affect the max.

So the load uses:

```python
other=-float("inf")
```

Then:

```text
max(real values, negative infinity) = max(real values)
```

That keeps boundary positions harmless.

## The Two Reductions

Softmax has two row reductions:

```text
row_max = max(values)
denominator = sum(exp(values - row_max))
```

In Triton:

```python
row_max = tl.max(values, axis=0)
denominator = tl.sum(numerator, axis=0)
```

Those lines are compact because Triton handles the block reduction expression.

The programmer still controls the row ownership, masks, and output layout.

## Fused Pipeline

The Triton softmax keeps the row pipeline together:

```text
load row
compute max
compute exp
compute sum
normalize
store row
```

The intermediate values live inside the program expression instead of being
separate global-memory arrays.

That is why Triton is pleasant for this kind of AI kernel.

## What Changes For Wider Rows

If `width > BLOCK_SIZE`, one program cannot cover the row.

Then the kernel needs a different strategy:

```text
split row into chunks
compute partial max values
combine max values
compute partial sums
combine sums
normalize
```

Real production softmax kernels handle many row sizes.

This lesson teaches the clean first case.

## The Core Pattern

When reading Triton softmax, ask:

```text
What row does one program own?
What is BLOCK_SIZE relative to width?
What mask protects the row boundary?
What value is used for masked max loads?
Where are max and sum reductions?
Where is the final masked store?
```

Softmax is simple when the row fits in one block.

The harder engineering begins when row sizes, masks, and performance targets
change.

## Bridge To Week 29

Week 29 moves from row-wise kernels to Triton matmul.

The mental model changes from:

```text
one program owns one row
```

to:

```text
one program owns one output tile
```
