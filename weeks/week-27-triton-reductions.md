# Week 27: Triton Reductions

Triton reductions build on the same block model as vector add.

The difference is the output shape.

Elementwise Triton:

```text
BLOCK_SIZE inputs -> BLOCK_SIZE outputs
```

Reduction Triton:

```text
BLOCK_SIZE inputs -> one value or fewer values
```

Week 27 teaches row-wise sum and max as Triton-shaped reductions.

## Row-Wise Ownership

For a matrix:

```text
height x width
```

A simple Triton reduction can map:

```text
one program -> one row
```

The program id chooses the row:

```python
row = tl.program_id(0)
```

The offsets choose columns:

```python
cols = tl.arange(0, BLOCK_SIZE)
```

The row pointer base is:

```python
row_start = x_ptr + row * width
```

Then the program loads:

```python
values = tl.load(row_start + cols, mask=cols < width, other=0.0)
```

## Row Sum

A row sum in Triton:

```python
@triton.jit
def row_sum_kernel(x_ptr, out_ptr, height, width, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < width

    values = tl.load(x_ptr + row * width + cols, mask=mask, other=0.0)
    total = tl.sum(values, axis=0)

    tl.store(out_ptr + row, total)
```

The important line is:

```python
total = tl.sum(values, axis=0)
```

It reduces the block of loaded values into one scalar for the program.

## Row Max

For max, the masked value changes.

Sum can use:

```text
0.0
```

Max should use:

```text
negative infinity
```

Triton-shaped max:

```python
values = tl.load(x_ptr + row * width + cols, mask=mask, other=-float("inf"))
best = tl.max(values, axis=0)
tl.store(out_ptr + row, best)
```

Using zero would be wrong if all real values are negative.

The `other` value must match the reduction identity.

## Width Assumption

This simple lesson assumes:

```text
width <= BLOCK_SIZE
```

If the row is wider, one program cannot cover the whole row.

Then the design needs partial reductions:

```text
programs reduce row chunks
another stage combines chunk results
```

That is the same partial-result idea from CUDA reductions.

Triton changes the syntax.

It does not remove the algorithmic shape.

## Why Triton Reductions Feel Compact

In CUDA, you manually wrote:

```text
shared memory
stride loop
barriers
```

In Triton, `tl.sum` or `tl.max` expresses the block reduction directly:

```python
tl.sum(values, axis=0)
```

That is productive.

But you still need to know:

```text
what values were loaded
which lanes were masked
what identity value was used
what output shape is expected
```

The mental model still matters.

## The Core Pattern

When reading a Triton reduction, ask:

```text
What does one program reduce?
What is the valid mask?
What identity value is used for invalid positions?
Is the output one value per row, block, or tile?
What happens if the row is wider than the block?
```

If those answers are clear, the reduction is usually readable.

## Bridge To Week 28

Week 28 combines Triton reductions into row-wise softmax:

```text
row max
exp
row sum
divide
```

This is where Triton starts to feel very good for AI kernels.
