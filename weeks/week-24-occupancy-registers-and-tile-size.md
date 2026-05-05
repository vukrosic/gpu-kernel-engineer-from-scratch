# Week 24: Occupancy, Registers, And Tile Size

Week 23 showed why larger tiles can improve reuse.

Week 24 teaches the other half:

```text
larger tiles also consume more GPU resources
```

The goal is not to memorize an occupancy formula.

The goal is to understand why a kernel can become slower even when the tile
looks more reusable.

## Occupancy

Occupancy is about how many warps can be active on a streaming multiprocessor.

High occupancy can help hide latency:

```text
while one warp waits on memory, another warp can run
```

But occupancy is not the same as performance.

A kernel can have high occupancy and still be slow.

A kernel can have lower occupancy and be fast if each warp does useful work.

Use occupancy as a signal, not a final score.

## The Resource Budget

Each block uses resources:

```text
threads
registers
shared memory
```

The GPU has a limited amount of those resources per SM.

If each block uses more resources, fewer blocks can fit at the same time.

That can reduce occupancy.

The basic tradeoff is:

```text
more resources per block -> fewer active blocks
fewer resources per block -> more active blocks
```

Neither side is automatically correct.

## Registers

Registers hold thread-local values:

```text
accumulators
loaded scalar values
temporary indexes
intermediate math
```

In matmul, accumulators are especially important.

If one thread computes one output, it may need one accumulator:

```cpp
float total = 0.0f;
```

If one thread computes multiple outputs, it may need several accumulators:

```cpp
float acc00 = 0.0f;
float acc01 = 0.0f;
float acc10 = 0.0f;
float acc11 = 0.0f;
```

More accumulators can improve reuse.

They also use more registers.

Too many registers can reduce occupancy or cause spilling.

Spilling means values that should live in registers are moved to memory.

That is usually bad.

## Shared Memory

Tiled matmul uses shared memory for A and B tiles:

```cpp
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];
```

For `TILE = 16`:

```text
As: 256 floats
Bs: 256 floats
total: 512 floats
```

For `TILE = 32`:

```text
As: 1024 floats
Bs: 1024 floats
total: 2048 floats
```

Doubling tile width quadruples tile storage.

That can reduce how many blocks fit on the SM.

## Threads Per Block

A simple square tiled kernel often uses:

```text
TILE x TILE threads
```

For `TILE = 16`:

```text
256 threads
```

For `TILE = 32`:

```text
1024 threads
```

1024 threads is often the maximum block size.

That leaves little flexibility.

Real matmul kernels often separate:

```text
tile size
thread block shape
work per thread
```

The teaching kernel keeps them tied together because it is easier to see.

Optimized kernels make the mapping more flexible.

## A Simple Tile Report

A lightweight way to reason about candidates:

```python
def tile_report(tile):
    tile_m, tile_n, tile_k = tile
    a_values = tile_m * tile_k
    b_values = tile_k * tile_n
    outputs = tile_m * tile_n
    return {
        "tile": tile,
        "loaded_values": a_values + b_values,
        "outputs": outputs,
        "multiply_adds": tile_m * tile_n * tile_k,
    }
```

This does not predict final speed.

It makes the tradeoff visible:

```text
how much data is loaded?
how much compute is done?
how many outputs are produced?
```

## Bigger Is Not Always Better

A bigger tile may:

```text
reuse more data
do more work per block
```

But it may also:

```text
use too much shared memory
use too many registers
reduce active blocks
make scheduling less flexible
increase boundary waste
```

The best tile depends on:

```text
matrix shape
hardware
data type
kernel implementation
memory layout
```

That is why tuning exists.

## What To Measure

When comparing tile sizes, measure:

```text
correctness first
runtime
achieved bandwidth or FLOP/s
occupancy signals
register count
shared memory use
whether performance changes with shape
```

One shape is not enough.

A tile that works well for large square matrices may be poor for skinny or
batched shapes.

## The Core Pattern

When thinking about tile size, ask:

```text
What reuse does the larger tile buy?
How much shared memory does it cost?
How many registers does each thread need?
How many threads are in the block?
How many blocks can stay active?
Which matrix shapes is this tile meant for?
```

GPU tuning is not guessing bigger numbers.

It is choosing a resource tradeoff and measuring whether the tradeoff paid off.

## Bridge To Week 25

Week 25 starts Triton.

Triton keeps the same kernel engineering questions:

```text
what tile does one program own?
how are masks handled?
what block sizes are chosen?
what gets reused?
```

The syntax changes, but the mental model you built for CUDA still matters.
