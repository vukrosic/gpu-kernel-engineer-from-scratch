# Week 22: Tiled Matrix Multiplication

Week 21 taught naive matmul:

```text
one thread computes one output element
```

That version is correct, but it reloads many values.

Week 22 teaches the first performance idea:

```text
load small tiles of A and B, reuse them for multiple output elements
```

This is tiling.

## The Reuse Problem

Consider a small output tile:

```text
C[row 0, col 0]
C[row 0, col 1]
C[row 1, col 0]
C[row 1, col 1]
```

Those four outputs reuse the same nearby data:

```text
rows from A
columns from B
```

The naive kernel does not make that reuse explicit.

Each output thread walks through its own dot product and loads what it needs.

Tiling groups the work so a block can share loaded values.

## What A Tile Means

A tile is a small rectangular chunk of a matrix.

Example A tile:

```text
A tile, 2 x 3:
[a00 a01 a02]
[a10 a11 a12]
```

Example B tile:

```text
B tile, 3 x 2:
[b00 b01]
[b10 b11]
[b20 b21]
```

Multiplying those tiles contributes to a C tile:

```text
C tile, 2 x 2:
[c00 c01]
[c10 c11]
```

The shared dimension inside the tile is still `K`.

The math has not changed.

Only the work grouping changed.

## Tiled Matmul Shape

For full matmul:

```text
C = A @ B
```

A block can own one output tile:

```text
block (0, 0) computes top-left C tile
block (0, 1) computes next C tile to the right
block (1, 0) computes next C tile down
```

Inside each block:

```text
load one A tile
load one B tile
multiply-accumulate into C tile
move to next K tile
repeat
```

The block accumulates partial sums until the full `K` dimension is covered.

## Why Shared Memory Helps

The block can load tiles into shared memory:

```text
A tile -> shared memory
B tile -> shared memory
```

Then many threads reuse those shared values.

That is better than every thread loading the same global memory values again.

The pattern:

```text
global memory -> shared memory -> many multiply-adds
```

This is one of the most important GPU kernel patterns.

## A Small Python Tile Sketch

This sketch computes one output tile.

```python
def matmul_tile(a, b, row0, col0, tile_m, tile_n, tile_k):
    k_total = len(a[0])
    acc = [[0.0 for _ in range(tile_n)] for _ in range(tile_m)]

    for k0 in range(0, k_total, tile_k):
        for i in range(tile_m):
            for j in range(tile_n):
                for kk in range(tile_k):
                    k = k0 + kk
                    acc[i][j] += a[row0 + i][k] * b[k][col0 + j]

    return acc
```

This is not handling boundaries yet.

It shows the idea:

```text
compute a small output rectangle
move across K in chunks
accumulate partial dot products
```

## CUDA-Shaped Tiled Kernel

This teaching kernel assumes square tiles and simple dimensions:

```text
TILE x TILE thread block
M, N, and K are multiples of TILE
```

```cpp
#define TILE 16

__global__ void matmul_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float total = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[threadIdx.y][threadIdx.x] = A[row * K + (k0 + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];

        __syncthreads();

        for (int kk = 0; kk < TILE; ++kk) {
            total += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = total;
}
```

This is the classic tiled matmul teaching shape.

Real kernels add boundary checks, vectorization, register tiling, tensor cores,
and careful layouts.

Start with the shape.

## Step 1: Map A Block To A C Tile

The block chooses an output tile:

```cpp
int row = blockIdx.y * TILE + threadIdx.y;
int col = blockIdx.x * TILE + threadIdx.x;
```

Each thread still owns one output cell in this simple version.

The difference is that neighboring threads in the block cooperate on loading
input tiles.

## Step 2: Load A And B Tiles

Each thread loads one A value:

```cpp
As[threadIdx.y][threadIdx.x] = A[row * K + (k0 + threadIdx.x)];
```

Each thread also loads one B value:

```cpp
Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];
```

After these loads, the block has a tile of A and a tile of B in shared memory.

## Step 3: Wait For The Tile

The block must wait:

```cpp
__syncthreads();
```

Without this barrier, a thread could start multiplying before another thread
finished loading a shared value.

This is the synchronization lesson from Week 13.

## Step 4: Reuse The Tile

Each thread computes:

```cpp
for (int kk = 0; kk < TILE; ++kk) {
    total += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
}
```

For one output cell:

```text
use one row of As
use one column of Bs
accumulate TILE products
```

The loaded tile values are reused by many threads in the block.

That is the point of tiling.

## Step 5: Move Along K

The outer loop advances through K:

```cpp
for (int k0 = 0; k0 < K; k0 += TILE)
```

Each iteration contributes one chunk of the dot product.

The final output is complete only after all K tiles have been processed.

The second barrier protects the next tile load:

```cpp
__syncthreads();
```

It prevents some threads from overwriting shared memory while others are still
using it.

## What Tiling Improves

Naive matmul:

```text
load values directly from global memory for each output
```

Tiled matmul:

```text
load tiles from global memory once
reuse tile values from shared memory
```

The math is identical.

The memory behavior changes.

That is the first big matmul performance step.

## Boundary Checks

The teaching kernel assumes dimensions are multiples of `TILE`.

Real kernels need checks:

```text
row < M
col < N
k0 + threadIdx.x < K
k0 + threadIdx.y < K
```

Out-of-bounds tile positions should usually load zero.

That preserves the dot product without reading invalid memory.

## Tile Size Tradeoffs

Larger tiles can improve reuse.

They also use more shared memory and more threads:

```text
TILE 16 -> 256 threads for a 16 x 16 block
TILE 32 -> 1024 threads for a 32 x 32 block
```

Larger is not automatically better.

Tile size affects:

```text
shared memory use
register pressure
occupancy
memory coalescing
work per block
```

Week 23 will focus on those tradeoffs.

## The Core Pattern

When reading tiled matmul, ask:

```text
What C tile does one block own?
What A tile is loaded?
What B tile is loaded?
Where is shared memory used?
Where are the barriers?
How does the loop advance through K?
What happens at matrix boundaries?
```

Tiled matmul is the first place where many earlier lessons converge:

```text
indexing
shared memory
synchronization
memory reuse
reductions through K
performance tradeoffs
```

## Bridge To Week 23

Week 23 teaches tile sizes and occupancy.

The next question is:

```text
how do tile dimensions, registers, shared memory, and occupancy shape real
matmul performance?
```
