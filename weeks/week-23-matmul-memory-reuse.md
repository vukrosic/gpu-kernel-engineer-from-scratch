# Week 23: Matmul Memory Reuse

Week 22 introduced tiled matmul:

```text
load A tile
load B tile
reuse them while computing a C tile
```

Week 23 slows down and asks why that helps.

The answer is memory reuse.

Matmul does a lot of multiply-add work, but the performance story is still
about moving data at the right level of the memory hierarchy.

## The Naive Reuse Problem

In naive matmul, one thread computes one output:

```text
C[row, col] = sum(A[row, k] * B[k, col])
```

Now look at neighboring outputs:

```text
C[row, col]
C[row, col + 1]
C[row, col + 2]
```

They all use the same A row:

```text
A[row, 0], A[row, 1], A[row, 2], ...
```

Naive matmul does not explicitly share those A values.

Each output thread may load them again.

That repeated loading is the waste tiling is trying to reduce.

## Reuse In A C Tile

Suppose one block computes a 2 x 2 output tile:

```text
C00 C01
C10 C11
```

To compute that tile, the block needs:

```text
two rows from A
two columns from B
```

An A value like `A[0, k]` contributes to:

```text
C00
C01
```

A B value like `B[k, 0]` contributes to:

```text
C00
C10
```

So values from A and B are reused across multiple output cells.

Tiling makes that reuse visible.

## Arithmetic Intensity

Arithmetic intensity means:

```text
how much compute you do per byte moved
```

Elementwise kernels usually have low arithmetic intensity:

```text
read a few values
do a tiny amount of math
write one value
```

Matmul can have high arithmetic intensity if the kernel reuses loaded values:

```text
load tile values once
use them for many multiply-adds
```

That is why matmul can become compute-heavy instead of purely memory-heavy.

The catch is that the kernel must actually organize the reuse.

## Global Memory To Shared Memory

The core tiled pattern is:

```text
global memory -> shared memory -> registers/accumulators
```

Global memory is large but slower.

Shared memory is small but faster and block-local.

Registers are private to each thread and fastest.

A tiled kernel tries to:

```text
load A and B tiles from global memory
store them in shared memory
reuse them many times from shared memory
accumulate results in registers
write C once
```

That is the memory hierarchy story behind tiled matmul.

## Accumulators Are Reuse Too

Each thread keeps a running total:

```cpp
float total = 0.0f;
```

That `total` lives in a register.

The thread updates it across K tiles:

```cpp
total += As[row_in_tile][kk] * Bs[kk][col_in_tile];
```

The output is not written after every partial product.

It is written once at the end:

```cpp
C[row * N + col] = total;
```

Keeping partial sums in registers is another form of reuse.

The kernel reuses the accumulator instead of repeatedly writing partial results
to memory.

## A Reuse Counting Example

For a 16 x 16 C tile and a K tile of 16:

```text
A tile: 16 x 16 = 256 values
B tile: 16 x 16 = 256 values
C tile: 16 x 16 = 256 outputs
```

The block loads:

```text
512 input values
```

Then performs:

```text
16 x 16 x 16 = 4096 multiply-add steps
```

Those tile values are reused across many operations.

Without reuse, the kernel would spend much more time asking global memory for
the same data.

## Reuse Has A Price

More reuse usually means bigger tiles.

Bigger tiles can require:

```text
more shared memory
more registers
more threads
more synchronization
```

So reuse is not free.

The engineering question is:

```text
does the extra reuse pay for the extra resource pressure?
```

That is why Week 24 looks at occupancy, registers, and tile size.

## The Core Pattern

When reading matmul code, ask:

```text
Which values are loaded from global memory?
Which values are stored in shared memory?
How many outputs reuse each loaded value?
Where are partial sums accumulated?
How many times is C written?
What resource gets larger when tile size grows?
```

Matmul performance starts to make sense when you can point to reuse.

Without reuse, tiling is just extra indexing.

With reuse, tiling is the reason the kernel can do much more math per byte.

## Bridge To Week 24

Week 24 teaches the cost side of tiling:

```text
shared memory
registers
threads per block
occupancy
```

The next lesson is about why the largest tile is not automatically the best
tile.
