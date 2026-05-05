# Week 29: Triton Matmul Basics

Triton matmul uses the same math as CUDA matmul:

```text
C = A @ B
```

The difference is how the tile is expressed.

In Triton, one program instance usually owns one output tile:

```text
BLOCK_M x BLOCK_N values of C
```

Week 29 teaches that mapping.

## One Program Owns One C Tile

For C shaped:

```text
M x N
```

A Triton matmul program might own:

```text
BLOCK_M rows
BLOCK_N columns
```

The program ids choose the tile:

```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
```

The row offsets are:

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
```

The column offsets are:

```python
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
```

Together, those offsets describe a rectangle of C.

## The K Loop

Matmul still reduces across K.

Triton matmul moves through K in blocks:

```text
BLOCK_K at a time
```

For each K block:

```text
load A tile: BLOCK_M x BLOCK_K
load B tile: BLOCK_K x BLOCK_N
accumulate into C tile
```

Then the program advances to the next K block.

## Accumulator Tile

The accumulator is a matrix:

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
```

That means one program is accumulating many C elements at once.

Each K block adds more products:

```python
acc += tl.dot(a, b)
```

This is the Triton equivalent of:

```text
for k:
  total += A[row, k] * B[k, col]
```

but done for a whole output tile.

## Teaching Kernel Shape

A simplified Triton matmul shape:

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K, BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * K + (k0 + offs_k[None, :]))
        b = tl.load(B + (k0 + offs_k[:, None]) * N + offs_n[None, :])
        acc += tl.dot(a, b)

    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)
```

This version omits boundary masks to keep the tile idea visible.

Real kernels need masks for M, N, and K boundaries.

## Reading The A Tile

The A tile uses:

```python
offs_m[:, None] * K + (k0 + offs_k[None, :])
```

Shape:

```text
BLOCK_M x BLOCK_K
```

Rows come from `offs_m`.

Columns come from the current K block.

## Reading The B Tile

The B tile uses:

```python
(k0 + offs_k[:, None]) * N + offs_n[None, :]
```

Shape:

```text
BLOCK_K x BLOCK_N
```

Rows come from the current K block.

Columns come from `offs_n`.

## Writing The C Tile

The C tile uses:

```python
offs_m[:, None] * N + offs_n[None, :]
```

Shape:

```text
BLOCK_M x BLOCK_N
```

That is the output tile owned by the program.

## Masks

Real matmul needs masks:

```text
offs_m < M
offs_n < N
k0 + offs_k < K
```

Invalid A or B positions usually load zero.

Invalid C positions are not stored.

This is the same mask story from Weeks 25-28, now in 2D.

## The Core Pattern

When reading Triton matmul, ask:

```text
What C tile does one program own?
What are BLOCK_M, BLOCK_N, and BLOCK_K?
How are A and B tile offsets built?
Where does tl.dot happen?
What masks protect M, N, and K boundaries?
What dtype is used for accumulation?
```

Triton matmul is tile math plus careful indexing.

If the offsets are clear, the kernel becomes readable.

## Bridge To Week 30

Week 30 teaches the performance knobs:

```text
BLOCK_M
BLOCK_N
BLOCK_K
num_warps
num_stages
```

The next question is how to choose them.
