# Week 26: Triton Vector Add And Masks

Week 25 introduced Triton program instances.

Week 26 uses the simplest useful kernel to make that concrete:

```text
out = x + y
```

The math is easy.

The lesson is the indexing and masking pattern.

## Why Vector Add Still Matters

Vector add is not impressive by itself.

It is useful because every part of the Triton mental model is visible:

```text
program id
offset block
mask
masked load
elementwise operation
masked store
```

If this shape is clear, larger Triton kernels become easier.

## The Edge Block

Suppose:

```text
n = 10
BLOCK_SIZE = 4
```

The programs cover:

```text
program 0 -> [0, 1, 2, 3]
program 1 -> [4, 5, 6, 7]
program 2 -> [8, 9, 10, 11]
```

Only elements 8 and 9 are valid in the last program.

So the mask is:

```text
[True, True, False, False]
```

That mask is what makes the same kernel work for uneven sizes.

## Masked Load

The Triton load:

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

means:

```text
load x[offset] where mask is true
use 0.0 where mask is false
```

The `other` value matters because masked-off lanes still need some value inside
the program expression.

For vector add, zero is safe for inactive lanes because the result will not be
stored where the mask is false.

## Masked Store

The store:

```python
tl.store(out_ptr + offsets, out, mask=mask)
```

means:

```text
write only valid output positions
```

Without the store mask, the kernel could write past the end of the array.

Both load and store masks are part of correctness.

## Full Kernel Shape

A clean vector add kernel:

```python
@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    out = x + y

    tl.store(out_ptr + offsets, out, mask=mask)
```

Read it as:

```text
this program owns one contiguous block of elements
```

The program does not need an `if` around each scalar element.

The mask handles the boundary for the whole block.

## Launch Grid

The number of programs is:

```python
grid = (triton.cdiv(n, BLOCK_SIZE),)
```

`cdiv` is ceiling division.

For `n = 10` and `BLOCK_SIZE = 4`:

```text
ceil(10 / 4) = 3 programs
```

The last program is partial.

That is normal.

## Masks In Bigger Kernels

The same idea appears in:

```text
row-wise softmax
reductions
matmul boundary tiles
batched matmul
```

The mask may become 2D:

```text
row offsets valid
column offsets valid
```

But the idea stays the same:

```text
compute a block-shaped set of offsets
mark which positions are legal
load and store only legal positions
```

## The Core Pattern

When reading masked Triton code, ask:

```text
What offsets are being created?
Which dimension can run past the end?
What value is used for masked loads?
Is the store masked too?
Would the operation still be correct for the last block?
```

Most beginner Triton bugs are mask bugs.

Treat masks as correctness code, not decoration.

## Bridge To Week 27

Week 27 moves from elementwise work to reductions in Triton.

The next question is:

```text
how does a block of values become one row statistic?
```
