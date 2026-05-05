# Week 11: Block-Level Reductions With Shared Memory

Week 10 showed the simplest correct reduction kernel:

```text
one thread computes one complete output
```

For row sum, that meant one thread looped across the whole row.

That is correct, but it wastes most of the GPU. If a row has 1024 values, one
thread does 1024 loads and 1024 additions while nearby threads could have helped.

Week 11 teaches the next implementation shape:

```text
one block cooperates to compute one partial or complete reduction output
```

The new idea is not a new math operation.

The new idea is cooperation.

## The Problem With The Naive Row Sum

Start with one row:

```text
[1, 2, 3, 4, 5, 6, 7, 8]
```

The naive Week 10 shape is:

```text
thread 0:
  total = 0
  total += 1
  total += 2
  total += 3
  total += 4
  total += 5
  total += 6
  total += 7
  total += 8
  out[0] = total
```

Only one thread is doing useful work for that row.

A block-level reduction changes the shape:

```text
thread 0 loads 1
thread 1 loads 2
thread 2 loads 3
thread 3 loads 4
thread 4 loads 5
thread 5 loads 6
thread 6 loads 7
thread 7 loads 8
```

Then the block combines the values.

The row is still reduced to one number, but the work is shared.

## What A Block Owns

For the first shared-memory reduction, use this simple mapping:

```text
one block owns one row
```

For a matrix with 4 rows:

```text
block 0 -> row 0
block 1 -> row 1
block 2 -> row 2
block 3 -> row 3
```

Inside each block:

```text
thread 0 -> one column value
thread 1 -> one column value
thread 2 -> one column value
...
```

This is different from Week 10.

Week 10:

```text
one thread owns one row
```

Week 11:

```text
one block owns one row
many threads cooperate on that row
```

That is the main mental model.

## Why Shared Memory Appears

Threads in the same block need a place to put their loaded values so other
threads in the block can help combine them.

That place is shared memory.

You can think of shared memory as a small scratchpad owned by one block:

```text
global memory:
  large
  slower
  visible to all blocks

shared memory:
  small
  faster
  visible only inside one block
```

In a reduction, shared memory is useful because the block can do this:

```text
load values from global memory
store them in shared memory
combine shared values in stages
write one result back to global memory
```

Shared memory is not the final output.

It is temporary workspace for cooperation.

## The Reduction Tree

Suppose the row has 8 values:

```text
[1, 2, 3, 4, 5, 6, 7, 8]
```

The block can reduce them in stages.

Stage 1:

```text
1 + 5 = 6
2 + 6 = 8
3 + 7 = 10
4 + 8 = 12
```

Now only 4 useful partials remain:

```text
[6, 8, 10, 12]
```

Stage 2:

```text
6 + 10 = 16
8 + 12 = 20
```

Now 2 useful partials remain:

```text
[16, 20]
```

Stage 3:

```text
16 + 20 = 36
```

The final answer is:

```text
36
```

This is a tree-shaped reduction:

```text
8 values -> 4 partials -> 2 partials -> 1 output
```

The amount of work did not disappear.

The work was spread across the block.

## A Python Version Of The Idea

Here is the shape without CUDA syntax:

```python
def block_reduce_sum(values):
    scratch = values[:]
    active = len(scratch)

    while active > 1:
        half = active // 2

        for i in range(half):
            scratch[i] = scratch[i] + scratch[i + half]

        active = half

    return scratch[0]
```

Read the loop as:

```text
combine the first half with the second half
then only keep the first half
repeat until one value remains
```

For 8 values:

```text
active = 8
active = 4
active = 2
active = 1
```

CUDA does not run that `for` loop with one CPU worker.

CUDA uses threads inside the block to do the pairwise additions in parallel.

## CUDA Shape: One Block, One Row

Here is the high-level CUDA-shaped version.

This teaching version assumes:

```text
blockDim.x is a power of two
width <= blockDim.x
```

That keeps the reduction tree easy to see.

Do not read it all at once.

```cpp
__global__ void row_sum_block(
    const float* x,
    float* out,
    int height,
    int width
) {
    extern __shared__ float scratch[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= height) {
        return;
    }

    float value = 0.0f;
    if (tid < width) {
        value = x[row * width + tid];
    }

    scratch[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        out[row] = scratch[0];
    }
}
```

Now split it into pieces.

## Step 1: Choose The Row

The block chooses the row:

```cpp
int row = blockIdx.x;
```

That means:

```text
block 0 handles row 0
block 1 handles row 1
block 2 handles row 2
```

The thread chooses its position inside the block:

```cpp
int tid = threadIdx.x;
```

That means:

```text
thread 0 handles column-like position 0
thread 1 handles column-like position 1
thread 2 handles column-like position 2
```

The row check protects the output row:

```cpp
if (row >= height) {
    return;
}
```

## Step 2: Load One Value Per Thread

Each thread loads at most one value:

```cpp
float value = 0.0f;
if (tid < width) {
    value = x[row * width + tid];
}
```

The indexing is still the Week 03 row-major formula:

```text
index = row * width + col
```

Here, `tid` is acting like the column:

```text
index = row * width + tid
```

The `tid < width` check matters because the block may have more threads than
the row has columns.

Extra threads load zero so they do not change the sum.

## Step 3: Store Into Shared Memory

Each thread writes its value into the block scratchpad:

```cpp
scratch[tid] = value;
```

After this line, shared memory looks like the row values:

```text
scratch[0] = x[row, 0]
scratch[1] = x[row, 1]
scratch[2] = x[row, 2]
...
```

For a sum, inactive positions can be zero:

```text
scratch[valid values] = row values
scratch[extra positions] = 0
```

## Step 4: Wait For The Block

The block must wait:

```cpp
__syncthreads();
```

This means:

```text
do not continue until every thread in the block reaches this point
```

Without this barrier, thread 0 might start adding before another thread has
finished writing its value into `scratch`.

That would be a race condition.

The code might work sometimes and fail other times.

GPU bugs often look like that.

## Step 5: Reduce In Stages

The reduction loop is:

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
        scratch[tid] += scratch[tid + stride];
    }

    __syncthreads();
}
```

If `blockDim.x` is 8, the strides are:

```text
4, 2, 1
```

At stride 4:

```text
thread 0 adds scratch[0] + scratch[4]
thread 1 adds scratch[1] + scratch[5]
thread 2 adds scratch[2] + scratch[6]
thread 3 adds scratch[3] + scratch[7]
```

At stride 2:

```text
thread 0 adds scratch[0] + scratch[2]
thread 1 adds scratch[1] + scratch[3]
```

At stride 1:

```text
thread 0 adds scratch[0] + scratch[1]
```

After the final stage:

```text
scratch[0] contains the row sum
```

## Why The Barrier Is Inside The Loop

After each stage, some threads update shared memory.

The next stage depends on those updates.

So the block must wait after every stage:

```cpp
__syncthreads();
```

This is the pattern:

```text
read shared memory
write partial result
wait
read the new partials
write smaller partials
wait
```

The barrier turns many independent thread actions into one ordered block-level
algorithm.

## Step 6: Write One Output

Only one thread writes the result:

```cpp
if (tid == 0) {
    out[row] = scratch[0];
}
```

That avoids multiple threads writing to the same output.

For row sum:

```text
one block produces one row output
```

So thread 0 acts as the final writer for the block.

## What This Improves

The naive Week 10 kernel did this:

```text
one thread loads every value in a row
one thread performs every addition
```

The block-level version does this:

```text
many threads load row values
many threads build partial sums
one thread writes the final sum
```

This can be much faster for wide rows because the work is spread across threads.

It also teaches the pattern used by many real GPU kernels:

```text
load a tile
cooperate inside the block
write a smaller result
```

You will see the same pattern again in softmax, layer normalization, and tiled
matrix multiplication.

## What This Does Not Solve Yet

This first block-level design is still limited.

It assumes one row can fit into one block:

```text
width <= blockDim.x
```

If a row has more values than one block can load, the kernel needs another
level of reduction:

```text
block 0 reduces part of the row
block 1 reduces another part of the row
another kernel combines the partial outputs
```

That gives this shape:

```text
many values -> block partials -> final output
```

This is common in real reduction kernels.

The first kernel produces partial sums.

The second kernel reduces the partial sums.

## Sum And Max Use The Same Structure

The tree structure is not only for sum.

For row max, the operation changes:

```cpp
scratch[tid] = max(scratch[tid], scratch[tid + stride]);
```

The structure stays the same:

```text
load values
store in shared memory
combine in stages
write one output
```

The identity value changes.

For sum, inactive threads should contribute:

```text
0
```

For max, inactive threads should contribute:

```text
negative infinity
```

That way extra threads do not change the answer.

## The Core Pattern

A block-level shared-memory reduction has six parts:

```text
1. Map one block to one output region.
2. Let each thread load one or more input values.
3. Store those values in shared memory.
4. Synchronize the block.
5. Reduce shared memory in stages.
6. Let one thread write the block result.
```

This is the lesson to keep.

The code details will change from kernel to kernel, but this structure keeps
coming back.

## How To Read Reduction Kernels Now

When you see a shared-memory reduction, ask these questions:

```text
What does one block own?
What does one thread load?
Where is shared memory filled?
Where are the barriers?
How does the stride shrink?
Which thread writes the result?
Is the result final or only partial?
```

Those questions make reduction code much easier to read.

Without them, the kernel looks like a pile of indexes and barriers.

With them, the kernel becomes a small cooperative algorithm.

## Bridge To Week 12

Week 11 reduced values through shared memory and block-wide barriers.

Week 12 goes one level deeper:

```text
what happens when the cooperating threads are inside the same warp?
```

Warp-level reduction thinking matters because threads in a warp execute together.

That changes how reductions can be written and how much synchronization they
need.
