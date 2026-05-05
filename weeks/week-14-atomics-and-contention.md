# Week 14: Atomics And Contention

Week 13 taught barriers:

```text
wait until shared data is ready
```

Week 14 teaches atomics:

```text
update shared data as one indivisible operation
```

Atomics are useful when many threads need to modify the same memory location.

They are also easy to overuse.

The engineering question is not only:

```text
does this need an atomic?
```

It is also:

```text
how much contention will this atomic create?
```

## The Lost Update Problem

Suppose two threads both increment the same counter:

```cpp
counter = counter + 1;
```

That line looks like one operation, but it is really closer to:

```text
read counter
add 1
write counter
```

If two threads do this at the same time, a bad interleaving is possible:

```text
counter starts at 0

thread A reads 0
thread B reads 0
thread A writes 1
thread B writes 1
```

Two increments happened.

The counter only increased once.

That is a lost update.

## Atomic Add

An atomic add makes the update indivisible:

```cpp
atomicAdd(&counter, 1);
```

Read it as:

```text
add 1 to counter without allowing another thread to interrupt the update
```

The operation is protected.

Multiple threads can still call it.

The hardware makes sure each update is counted.

## Histogram: The Natural Atomic Example

A histogram counts how many inputs fall into each bucket.

For values:

```text
[0, 1, 1, 2, 2, 2, 3]
```

The histogram is:

```text
bin 0 -> 1
bin 1 -> 2
bin 2 -> 3
bin 3 -> 1
```

The CPU version is simple:

```python
def histogram(values, bins):
    counts = [0] * bins

    for value in values:
        counts[value] += 1

    return counts
```

In a GPU version, many threads process values at once.

If many values map to the same bin, many threads update the same counter.

That is exactly where atomics appear.

## CUDA-Shaped Histogram

A simple histogram kernel looks like this:

```cpp
__global__ void histogram_atomic(
    const int* values,
    int* counts,
    int n,
    int bins
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int bin = values[i];

        if (0 <= bin && bin < bins) {
            atomicAdd(&counts[bin], 1);
        }
    }
}
```

Read it in three parts.

First, choose one input:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Second, map it to a bin:

```cpp
int bin = values[i];
```

Third, safely increment the shared counter:

```cpp
atomicAdd(&counts[bin], 1);
```

Without the atomic, updates to the same bin could be lost.

## What The Atomic Protects

The atomic does not protect the whole kernel.

It protects one memory update:

```cpp
counts[bin] += 1;
```

That matters because atomics are precise tools.

They are not the same as barriers.

A barrier says:

```text
everyone in this block waits here
```

An atomic says:

```text
this one memory update happens safely
```

Those are different coordination problems.

## Contention

Contention happens when many threads want the same memory location.

Low contention:

```text
values are spread across many bins
```

High contention:

```text
most values go to the same bin
```

High contention can make atomics slow because many updates must be serialized.

The code is parallel.

The hottest counter becomes a bottleneck.

## Skewed Inputs

This input is friendly:

```text
[0, 1, 2, 3, 0, 1, 2, 3]
```

Updates are spread across bins.

This input is hostile:

```text
[2, 2, 2, 2, 2, 2, 2, 2]
```

Every thread wants `counts[2]`.

The atomic is still correct.

It may be slow.

Correctness and performance are separate questions.

## Privatization

One common optimization is privatization.

Instead of every thread updating one global histogram directly, each block can
build a smaller private histogram first:

```text
block 0 builds private counts
block 1 builds private counts
block 2 builds private counts
```

Then the private histograms are combined:

```text
global counts = block 0 counts + block 1 counts + block 2 counts
```

This reduces global atomic pressure.

It may still use atomics, but it moves some contention into faster shared memory
or into a second reduction step.

The pattern is familiar:

```text
local partial result first
global result later
```

You saw the same idea in reductions.

## Atomic Add Is Not Only For Histograms

Atomics also appear in:

```text
counting valid elements
building sparse outputs
accumulating gradients
work queues
deduplication
graph algorithms
```

The common shape is:

```text
many workers discover contributions to a shared result
```

If the destination is shared and updates can collide, atomics may be needed.

## When Atomics Are A Good First Design

Atomics are often a good first implementation when:

```text
correctness matters more than peak speed
the code is simpler with direct shared updates
contention is expected to be low
the result is a counter or accumulator
```

They are often a performance problem when:

```text
many threads update one hot location
the update happens inside a very tight loop
the atomic is on global memory
the algorithm could use partial results instead
```

Start with correctness.

Then measure contention.

Then decide whether the atomic needs redesign.

## The Core Pattern

When reading atomic code, ask:

```text
What memory location is shared?
Which threads can update it?
Can two updates collide?
What operation must be atomic?
Is the input likely to create hot spots?
Can partial results reduce contention?
```

Those questions are the difference between "I used atomicAdd" and actual GPU
engineering.

## Bridge To Week 15

Atomics update shared state directly.

Scan solves a different problem:

```text
each position needs to know how much came before it
```

That idea shows up in compaction, indexing, offsets, and many parallel
algorithms.
