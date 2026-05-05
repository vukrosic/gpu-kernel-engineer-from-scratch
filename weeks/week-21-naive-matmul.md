# Week 21: Naive Matrix Multiplication

Matrix multiplication is the compute pattern behind most deep learning layers.

Before tiling, tensor cores, Triton, or attention, you need the simple version:

```text
one output element is one dot product
```

Week 21 teaches that baseline.

It is not fast.

It is the reference shape that all faster matmul kernels preserve.

## The Shapes

Matrix multiplication combines:

```text
A: M x K
B: K x N
```

The output is:

```text
C: M x N
```

The shared dimension is `K`.

That means:

```text
A columns must equal B rows
```

Example:

```text
A is 2 x 3
B is 3 x 4
C is 2 x 4
```

Each output element `C[i, j]` uses:

```text
row i of A
column j of B
```

## One Output Cell

For:

```text
A row:    [1, 2, 3]
B column: [4, 5, 6]
```

The dot product is:

```text
1 * 4 + 2 * 5 + 3 * 6 = 32
```

That is one output cell.

Matrix multiplication repeats that for every output row and column.

## CPU Reference

The clearest reference is three loops:

```python
def matmul_reference(a, b):
    m = len(a)
    k = len(a[0])
    n = len(b[0])

    out = [[0.0 for _ in range(n)] for _ in range(m)]

    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total += a[i][p] * b[p][j]
            out[i][j] = total

    return out
```

Read the loops as:

```text
i chooses output row
j chooses output column
p walks across the dot product
```

The inner loop is the reduction:

```text
sum over K
```

So matmul is many dot-product reductions arranged in a 2D output grid.

## Row-Major Indexing

CUDA kernels often use flat pointers.

For row-major matrices:

```text
A[i, p] = A[i * K + p]
B[p, j] = B[p * N + j]
C[i, j] = C[i * N + j]
```

Those formulas matter.

Most matmul bugs are indexing bugs.

Keep the dimensions attached to each matrix:

```text
A uses K as row width
B uses N as row width
C uses N as row width
```

## Naive CUDA Shape

The simplest CUDA mapping is:

```text
one thread computes one C element
```

A CUDA-shaped kernel:

```cpp
__global__ void matmul_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float total = 0.0f;

        for (int p = 0; p < K; ++p) {
            float a = A[row * K + p];
            float b = B[p * N + col];
            total += a * b;
        }

        C[row * N + col] = total;
    }
}
```

This is the kernel to understand before tiling.

## Step 1: Choose Output Position

The output is 2D, so the thread computes a 2D coordinate:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

`row` chooses the output row.

`col` chooses the output column.

The bounds check protects the output:

```cpp
if (row < M && col < N) {
```

This is the same habit as elementwise kernels, but now the index is 2D.

## Step 2: Walk The Shared Dimension

The inner loop walks `K`:

```cpp
for (int p = 0; p < K; ++p) {
```

Each step loads:

```cpp
float a = A[row * K + p];
float b = B[p * N + col];
```

Then accumulates:

```cpp
total += a * b;
```

The output cell is complete only after all `K` products have been added.

## Step 3: Write One Output

After the dot product:

```cpp
C[row * N + col] = total;
```

One thread writes one output cell.

That makes the naive kernel simple:

```text
no atomics
no shared memory
no barriers
```

Each thread owns its output.

## Why The Naive Version Is Slow

The naive kernel repeats loads.

Suppose neighboring threads compute:

```text
C[row, 0]
C[row, 1]
C[row, 2]
C[row, 3]
```

They all need the same values from row `A[row, :]`.

But the naive version may load those A values separately for each output
column.

Likewise, neighboring output rows reuse values from B.

The naive kernel does not organize that reuse.

That is the performance problem tiling will solve.

## Memory Access Pattern

For A:

```cpp
A[row * K + p]
```

As `p` increases, this walks across a row of A.

That is contiguous.

For B:

```cpp
B[p * N + col]
```

For one thread, as `p` increases, this walks down a column of B.

In row-major memory, that jumps by `N`.

The access pattern is not equally friendly for both matrices.

This is one reason matmul optimization gets interesting.

## Correctness Questions

When checking a naive matmul, ask:

```text
Is A shaped M x K?
Is B shaped K x N?
Is C shaped M x N?
Does the inner loop run over K?
Does A use row * K + p?
Does B use p * N + col?
Does C use row * N + col?
```

If those are right, the baseline is probably right.

If any of those are wrong, the kernel may pass square-matrix tests and fail
rectangular shapes.

Always test rectangular shapes.

## The Core Pattern

Naive matmul has four parts:

```text
1. Map one thread to one output cell.
2. Use row and column to select A row and B column.
3. Reduce across K with multiply-add.
4. Store the result in C.
```

This is the baseline.

Every optimized matmul still computes the same dot products.

The optimization is about how values are loaded, reused, and scheduled.

## Bridge To Week 22

Week 22 teaches tiling.

The key question becomes:

```text
can a block load a small tile of A and B once, then reuse those values for many
output elements?
```

That is where matmul starts to look like real GPU engineering.
