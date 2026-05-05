# Week 12: Warp-Level Reductions

Week 11 taught this pattern:

```text
threads in one block cooperate through shared memory
```

That is a real GPU pattern, but it is not the smallest unit of cooperation.

Inside a block, threads are grouped into warps.

On NVIDIA GPUs, a warp is usually:

```text
32 threads
```

Week 12 teaches how reduction thinking changes when the cooperating group is one
warp instead of the whole block.

## The New Unit: A Warp

A block contains many threads.

Those threads are executed in smaller groups called warps:

```text
block
  warp 0: threads 0-31
  warp 1: threads 32-63
  warp 2: threads 64-95
  ...
```

Each thread inside a warp has a lane:

```text
lane 0
lane 1
lane 2
...
lane 31
```

You can think of the lane as the thread's position inside the warp.

For example:

```cpp
int lane = threadIdx.x % 32;
```

The warp is important because those 32 lanes execute together.

That does not mean every lane always does useful work.

It means the hardware schedules the warp as a group.

## Why Warps Matter For Reductions

Week 11 used shared memory:

```text
thread writes value to shared memory
block waits
thread reads another shared value
block waits
```

That is clear and useful.

But inside a single warp, lanes can exchange values more directly.

Instead of writing every partial value to shared memory, one lane can read a
value from another lane using a warp shuffle operation.

The high-level idea is:

```text
lane value -> exchanged with another lane -> added into local value
```

The reduction still has the same tree shape:

```text
32 values -> 16 partials -> 8 partials -> 4 partials -> 2 partials -> 1 output
```

The difference is where the values move.

Shared-memory block reduction:

```text
register -> shared memory -> register
```

Warp-level reduction:

```text
register in one lane -> register in another lane
```

That is why warp-level reductions can be faster and cleaner for small groups.

## The Reduction Tree Inside A Warp

Imagine 8 lanes instead of 32 so the shape is easier to see:

```text
lane 0: 1
lane 1: 2
lane 2: 3
lane 3: 4
lane 4: 5
lane 5: 6
lane 6: 7
lane 7: 8
```

At offset 4:

```text
lane 0 adds lane 4
lane 1 adds lane 5
lane 2 adds lane 6
lane 3 adds lane 7
```

Now the useful partials are:

```text
lane 0: 6
lane 1: 8
lane 2: 10
lane 3: 12
```

At offset 2:

```text
lane 0 adds lane 2
lane 1 adds lane 3
```

Now:

```text
lane 0: 16
lane 1: 20
```

At offset 1:

```text
lane 0 adds lane 1
```

Now:

```text
lane 0: 36
```

The full 32-lane version uses these offsets:

```text
16, 8, 4, 2, 1
```

That should feel familiar from Week 11.

The tree is the same.

The communication method is different.

## Warp Shuffle

CUDA exposes warp value exchange through shuffle operations.

The one to understand first is:

```cpp
__shfl_down_sync(mask, value, offset)
```

Read it like this:

```text
give me the value from the lane offset positions below me
```

If the offset is 16:

```text
lane 0 reads lane 16
lane 1 reads lane 17
lane 2 reads lane 18
...
```

If the offset is 8:

```text
lane 0 reads lane 8
lane 1 reads lane 9
lane 2 reads lane 10
...
```

The `value` stays in a register.

No shared-memory array is needed for the values being exchanged inside the warp.

## A Warp Sum Function

A common warp sum helper looks like this:

```cpp
__device__ float warp_sum(float value) {
    unsigned mask = 0xffffffff;

    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);

    return value;
}
```

This function reduces one value per lane.

After it runs:

```text
lane 0 has the sum for the warp
```

Other lanes may contain partial values.

For a normal reduction output, lane 0 is the lane you use.

## Why Lane 0 Writes

After a warp reduction, the final result is usually in lane 0:

```cpp
int lane = threadIdx.x % 32;

float sum = warp_sum(value);

if (lane == 0) {
    out[warp_id] = sum;
}
```

The `lane == 0` check matters because the warp has one final result.

You do not want all 32 lanes writing to the same output position.

This is the same idea as Week 11:

```text
many workers cooperate
one worker writes the result
```

The writer changed from:

```text
thread 0 in the block
```

to:

```text
lane 0 in the warp
```

## A Warp-Level Row Sum Shape

Suppose each row has 32 values.

Then one warp can reduce one row:

```text
warp 0 -> row 0
warp 1 -> row 1
warp 2 -> row 2
```

The CUDA-shaped kernel looks like this:

```cpp
__global__ void row_sum_warp(
    const float* x,
    float* out,
    int height,
    int width
) {
    int lane = threadIdx.x % 32;
    int warp_in_block = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;
    int row = blockIdx.x * warps_per_block + warp_in_block;

    float value = 0.0f;
    if (row < height && lane < width) {
        value = x[row * width + lane];
    }

    float sum = warp_sum(value);

    if (row < height && lane == 0) {
        out[row] = sum;
    }
}
```

This teaching version assumes:

```text
width <= 32
blockDim.x is a multiple of 32
```

The important indexes are `lane`, `warp_in_block`, and `row`.

## Reading The Mapping

This line gives the lane inside the warp:

```cpp
int lane = threadIdx.x % 32;
```

This line gives which warp the thread belongs to inside the block:

```cpp
int warp_in_block = threadIdx.x / 32;
```

This line says how many warps fit in a block:

```cpp
int warps_per_block = blockDim.x / 32;
```

This line maps one warp to one row:

```cpp
int row = blockIdx.x * warps_per_block + warp_in_block;
```

So if `blockDim.x = 128`, there are 4 warps per block:

```text
block 0 warp 0 -> row 0
block 0 warp 1 -> row 1
block 0 warp 2 -> row 2
block 0 warp 3 -> row 3
block 1 warp 0 -> row 4
block 1 warp 1 -> row 5
```

That is the core mapping.

## What Happened To Shared Memory?

The warp reduction did not need shared memory for the 32 values inside the warp.

The values stayed in registers and moved through shuffle operations.

That is useful when the reduction fits inside one warp.

But shared memory is still important.

If a block has several warps and each warp produces one partial result, the
kernel may store those warp results in shared memory:

```text
warp 0 produces partial sum
warp 1 produces partial sum
warp 2 produces partial sum
warp 3 produces partial sum
shared memory stores those partials
one warp reduces the partials
```

So the common real pattern is:

```text
warp-level reduction inside each warp
shared memory between warps
warp-level reduction again for final block result
```

Week 11 and Week 12 work together.

Shared memory is for cooperation across warps in a block.

Shuffle operations are for cooperation inside one warp.

## What The Mask Means

The first argument to `__shfl_down_sync` is a mask:

```cpp
unsigned mask = 0xffffffff;
```

This full mask means:

```text
all 32 lanes are participating
```

That is fine for the simplest teaching examples.

Real kernels often need a smaller mask when only part of the warp is active.

For example, if a row has only 20 values:

```text
lanes 0-19 have real values
lanes 20-31 are inactive for this row
```

The inactive lanes should not accidentally contribute junk.

For this beginner version, we avoided that by setting inactive values to zero:

```cpp
float value = 0.0f;
if (row < height && lane < width) {
    value = x[row * width + lane];
}
```

That is enough for sum.

For max, inactive lanes need a value like negative infinity instead of zero.

## Sum, Max, And Softmax

Warp reductions are not only for sum.

For max, the helper has the same structure:

```cpp
__device__ float warp_max(float value) {
    unsigned mask = 0xffffffff;

    value = fmaxf(value, __shfl_down_sync(mask, value, 16));
    value = fmaxf(value, __shfl_down_sync(mask, value, 8));
    value = fmaxf(value, __shfl_down_sync(mask, value, 4));
    value = fmaxf(value, __shfl_down_sync(mask, value, 2));
    value = fmaxf(value, __shfl_down_sync(mask, value, 1));

    return value;
}
```

This matters later because softmax needs both:

```text
row max
row sum
```

LayerNorm and RMSNorm also need reductions.

Once you understand warp-level sum and max, many ML kernels become less
mysterious.

## Warp-Level Vs Block-Level

Use this mental comparison:

```text
block-level reduction:
  works across many threads in a block
  often uses shared memory
  often needs __syncthreads()

warp-level reduction:
  works across lanes in one warp
  often uses shuffle operations
  avoids shared memory for the inside-warp exchange
```

Neither one replaces the other.

They solve different cooperation problems.

For a small reduction:

```text
one warp may be enough
```

For a larger reduction:

```text
many warps produce partials
shared memory combines those partials
```

For a huge reduction:

```text
many blocks produce partials
another kernel combines those partials
```

Reduction design is mostly choosing the right cooperation level.

## The Core Pattern

A warp-level reduction has five parts:

```text
1. Give each lane one value.
2. Use shuffle operations to exchange values between lanes.
3. Reduce with offsets 16, 8, 4, 2, 1.
4. Keep the final result in lane 0.
5. Let lane 0 write or pass the result onward.
```

That is the lesson to keep.

The syntax can look strange at first, but the shape is still the same reduction
tree you already know.

## Bridge To Week 13

Week 11 used block-wide barriers.

Week 12 used warp-level cooperation.

Week 13 teaches synchronization more directly:

```text
when do threads need to wait?
what can go wrong when they do not?
what does a race condition look like?
```

That is the next GPU engineering skill: not only making threads cooperate, but
knowing exactly when cooperation requires synchronization.
