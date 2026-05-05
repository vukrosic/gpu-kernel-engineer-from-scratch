# Week 16: Parallel Scan Implementation

Week 15 defined scan:

```text
each output contains information from earlier positions
```

Week 16 teaches how a block can compute a scan in parallel.

The goal is not to memorize one perfect scan kernel.

The goal is to understand the implementation shape:

```text
load values into shared memory
move information across larger distances
synchronize between stages
write scanned values back out
```

## The Serial Baseline

The CPU-shaped inclusive scan is:

```python
def inclusive_scan(values):
    out = []
    running = 0

    for value in values:
        running += value
        out.append(running)

    return out
```

For:

```text
[1, 2, 3, 4, 5, 6, 7, 8]
```

The result is:

```text
[1, 3, 6, 10, 15, 21, 28, 36]
```

That loop is correct, but it is serial.

Each output waits for the previous output.

The GPU version needs a staged way to share prefix information.

## Hillis-Steele Scan

The first parallel scan shape to understand is Hillis-Steele.

It uses distances that double:

```text
1, 2, 4, 8, ...
```

At each distance, position `i` adds the value from `i - distance` if that
position exists.

For 8 values:

```text
distance 1
distance 2
distance 4
```

After those stages, every position has received all earlier values.

## Stage Example

Start:

```text
[1, 2, 3, 4, 5, 6, 7, 8]
```

After distance 1:

```text
[1, 3, 5, 7, 9, 11, 13, 15]
```

After distance 2:

```text
[1, 3, 6, 10, 14, 18, 22, 26]
```

After distance 4:

```text
[1, 3, 6, 10, 15, 21, 28, 36]
```

The output is now the inclusive scan.

The key idea is:

```text
each stage doubles how far information has traveled
```

## Why A Temporary Value Is Needed

In a parallel scan stage, every thread must read the old stage values before
any thread overwrites them.

This is wrong:

```cpp
if (tid >= offset) {
    scratch[tid] += scratch[tid - offset];
}
```

Why?

Because one thread might update `scratch[tid - offset]` before another thread
reads it.

The safer staged shape is:

```cpp
float add = 0.0f;

if (tid >= offset) {
    add = scratch[tid - offset];
}

__syncthreads();

scratch[tid] += add;

__syncthreads();
```

First, every thread reads what it needs.

Then the block waits.

Then every thread writes the next stage.

Then the block waits again.

That is Week 13 synchronization in action.

## CUDA-Shaped Inclusive Block Scan

This teaching version scans one block-sized chunk.

It assumes:

```text
n <= blockDim.x
one block handles the whole input
```

```cpp
__global__ void inclusive_scan_block(
    const float* x,
    float* out,
    int n
) {
    extern __shared__ float scratch[];

    int tid = threadIdx.x;

    float value = 0.0f;
    if (tid < n) {
        value = x[tid];
    }

    scratch[tid] = value;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        float add = 0.0f;

        if (tid >= offset) {
            add = scratch[tid - offset];
        }

        __syncthreads();

        scratch[tid] += add;

        __syncthreads();
    }

    if (tid < n) {
        out[tid] = scratch[tid];
    }
}
```

Read this as a staged data movement algorithm, not as a mysterious loop.

## Step 1: Load Into Shared Memory

Each thread loads one value:

```cpp
float value = 0.0f;
if (tid < n) {
    value = x[tid];
}
```

Then stores it:

```cpp
scratch[tid] = value;
__syncthreads();
```

The barrier makes sure the whole input chunk is available before scan stages
begin.

## Step 2: Move Across Distance 1

At offset 1:

```text
thread 1 reads thread 0's value
thread 2 reads thread 1's value
thread 3 reads thread 2's value
```

The code:

```cpp
if (tid >= offset) {
    add = scratch[tid - offset];
}
```

For `offset = 1`, each thread reads one position behind.

Then:

```cpp
scratch[tid] += add;
```

Now each position knows about its immediate left neighbor.

## Step 3: Move Across Larger Distances

At offset 2:

```text
each position receives information from two positions behind
```

At offset 4:

```text
each position receives information from four positions behind
```

The offsets double:

```cpp
for (int offset = 1; offset < blockDim.x; offset *= 2)
```

That is how prefix information spreads across the block.

## Step 4: Write The Output

After all stages:

```cpp
if (tid < n) {
    out[tid] = scratch[tid];
}
```

Every valid thread writes one scanned output.

Unlike reduction, many threads write results.

Reduction:

```text
many inputs -> one output
```

Scan:

```text
many inputs -> many prefix outputs
```

## Exclusive Scan From Inclusive Scan

An exclusive scan can be derived from the inclusive result.

For input:

```text
[1, 2, 3, 4]
```

Inclusive:

```text
[1, 3, 6, 10]
```

Exclusive:

```text
[0, 1, 3, 6]
```

The exclusive output is the inclusive output shifted right, with zero at the
front:

```cpp
if (tid == 0) {
    out[tid] = 0.0f;
} else if (tid < n) {
    out[tid] = scratch[tid - 1];
}
```

That works for sum because the identity value is zero.

For other operations, the first value must be that operation's identity.

## Block Scan Is Not Whole-Array Scan Yet

The teaching kernel scans one block-sized chunk.

Real arrays are often much larger.

For a large array, the shape becomes:

```text
1. scan each block locally
2. store each block total
3. scan the block totals
4. add the scanned block offsets back to each block
```

This is the scan version of the partial-result pattern:

```text
local result first
global correction later
```

You have seen this idea in reductions and histograms.

## Work Efficiency

Hillis-Steele is easy to understand, but it does more additions than necessary.

Another common scan algorithm is Blelloch scan.

It has two phases:

```text
upsweep: build partial totals like a reduction tree
downsweep: distribute prefix information back down the tree
```

Blelloch is more work-efficient.

Hillis-Steele is often the clearer first lesson.

The engineering habit is:

```text
learn the obvious staged version first
then optimize once the dependencies are clear
```

## The Core Pattern

When reading scan code, ask:

```text
Is this inclusive or exclusive?
What does one block scan?
Where are values stored between stages?
What distance is each stage using?
Why is each barrier needed?
How are block-level results combined for large arrays?
```

Scan is where synchronization starts to feel unavoidable.

The output is only correct if each stage sees a consistent previous stage.

## Bridge To Week 17

Week 17 starts softmax.

Softmax is not just a formula.

As a kernel, it combines several ideas you already learned:

```text
row-wise max reduction
exponentials
row-wise sum reduction
normalization
careful memory movement
```

The next lesson begins with the math before turning it into kernel structure.
