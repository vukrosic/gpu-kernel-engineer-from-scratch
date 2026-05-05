# Week 31: Batched Matmul Indexing

Batched matmul repeats matrix multiplication across a batch dimension.

Instead of:

```text
A: M x K
B: K x N
C: M x N
```

you have:

```text
A: BATCH x M x K
B: BATCH x K x N
C: BATCH x M x N
```

Week 31 teaches how the indexing changes.

The matmul inside each batch is the same.

The base pointer changes.

## One Batch Item

For batch `b`, the matmul is:

```text
C[b] = A[b] @ B[b]
```

The dot product is still:

```text
C[b, row, col] = sum(A[b, row, k] * B[b, k, col])
```

The batch dimension chooses which matrix pair to use.

## Flat Indexing

For row-major contiguous tensors:

```text
A[b, row, k] = A[b * M * K + row * K + k]
B[b, k, col] = B[b * K * N + k * N + col]
C[b, row, col] = C[b * M * N + row * N + col]
```

The batch stride is the size of one matrix:

```text
A batch stride = M * K
B batch stride = K * N
C batch stride = M * N
```

Most batched matmul bugs come from mixing these strides.

## Triton Program IDs

A batched Triton matmul can use three program dimensions:

```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
pid_b = tl.program_id(2)
```

`pid_b` chooses the batch item.

The tile offsets for M and N are the same as normal matmul.

The base pointers shift by batch:

```python
a_base = A + pid_b * M * K
b_base = B + pid_b * K * N
c_base = C + pid_b * M * N
```

Then the normal tile formulas apply from those bases.

## Batched Tile Shape

One program owns:

```text
one batch item
one BLOCK_M x BLOCK_N output tile
```

That means:

```text
program_id(2) selects batch
program_id(0) selects output rows
program_id(1) selects output columns
```

The kernel grid might look like:

```python
grid = (
    triton.cdiv(M, BLOCK_M),
    triton.cdiv(N, BLOCK_N),
    batch,
)
```

The first two dimensions tile the output matrix.

The third repeats that tiling for every batch item.

## Why Batching Helps

Batched matmul is common when many small matrix multiplications need to run.

Examples:

```text
attention heads
small inference batches
grouped operations
many independent projections
```

Batching gives the GPU more work to schedule.

It can improve utilization when each individual matmul is small.

## Layout Matters

This lesson assumes contiguous layout:

```text
batch-major row-major matrices
```

Real frameworks can have different strides.

The general formula is:

```text
base + b * stride_batch + row * stride_row + col * stride_col
```

Contiguous layout is a special case.

Understanding the stride version prepares you for PyTorch integration later.

## The Core Pattern

When reading batched matmul, ask:

```text
Which program id selects the batch?
What is the batch stride for A, B, and C?
Does each batch item use its own A and B?
Are M, N, and K shared across the batch?
Is the tensor contiguous or strided?
What grid dimension covers batch?
```

Batched matmul is not new math.

It is matmul plus one more indexing dimension.

## Bridge To Week 32

Week 32 moves to profiling and benchmark comparisons.

Once kernels have several shape and tuning choices, you need a disciplined way
to answer:

```text
what is actually faster, and why?
```
