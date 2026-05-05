# Week 13: Synchronization And Barriers

Weeks 11 and 12 introduced cooperative reductions.

That cooperation only works when threads agree on when shared data is ready.

Week 13 teaches the rule behind that:

```text
if one thread writes data that another thread reads, you must know when the
write becomes safe to read
```

That is synchronization.

The most important first tool is the block barrier:

```cpp
__syncthreads();
```

## The Problem: Correct Threads, Wrong Program

Imagine four threads in a block.

Each thread writes one value into shared memory:

```text
thread 0 writes scratch[0]
thread 1 writes scratch[1]
thread 2 writes scratch[2]
thread 3 writes scratch[3]
```

Then thread 0 wants to add them:

```text
scratch[0] + scratch[1] + scratch[2] + scratch[3]
```

The dangerous version is:

```cpp
scratch[tid] = x[tid];

if (tid == 0) {
    out[0] = scratch[0] + scratch[1] + scratch[2] + scratch[3];
}
```

Each individual line looks reasonable.

The problem is timing.

Thread 0 might read `scratch[2]` before thread 2 has written it.

That is a race condition:

```text
the final answer depends on the order threads happen to run
```

GPU programs must not depend on lucky timing.

## The Barrier

The fixed version adds a barrier:

```cpp
scratch[tid] = x[tid];

__syncthreads();

if (tid == 0) {
    out[0] = scratch[0] + scratch[1] + scratch[2] + scratch[3];
}
```

Read `__syncthreads()` as:

```text
every thread in this block must arrive here before any thread continues
```

After the barrier, thread 0 can safely assume all earlier shared-memory writes
from the block have happened.

The barrier does not make the math faster.

It makes the program correct.

## Barriers Are Block-Local

This is important:

```text
__syncthreads() only synchronizes threads inside the same block
```

It does not synchronize the whole grid.

Block 0 cannot use `__syncthreads()` to wait for block 1.

That is why large reductions often use more than one kernel:

```text
kernel 1: each block writes partial results
kernel 2: reduce the partial results
```

A kernel launch boundary acts like a global ordering point between kernels.

Inside one kernel, blocks mostly run independently.

## Where Barriers Show Up In Reductions

Week 11 used this shared-memory reduction shape:

```cpp
scratch[tid] = value;
__syncthreads();

for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
        scratch[tid] += scratch[tid + stride];
    }

    __syncthreads();
}
```

There are two reasons for the barriers.

First, after loading:

```text
all scratch values must exist before any thread combines them
```

Second, after each stage:

```text
the next stage must read the updated partial sums, not old values
```

That gives the rhythm:

```text
write shared memory
wait
read shared memory
write partials
wait
read updated partials
write smaller partials
wait
```

This rhythm is one of the core patterns in GPU kernels.

## A Race Condition In Plain Language

A race condition is not just "parallel code is hard."

It is more specific:

```text
two or more operations access the same data
at least one operation writes
the program result depends on which operation happens first
```

Example:

```text
thread 0 reads scratch[1]
thread 1 writes scratch[1]
```

If the read happens before the write, thread 0 gets an old value.

If the write happens before the read, thread 0 gets the new value.

Same code.

Different answer.

That is a correctness bug.

## Not Every Shared-Memory Use Needs A Barrier

If each thread only reads and writes its own location:

```cpp
scratch[tid] = x[tid] * 2.0f;
out[tid] = scratch[tid];
```

There is no cross-thread dependency.

Thread 0 does not need thread 1's value.

In that case, a barrier would not protect anything important.

A useful question is:

```text
does any thread read data written by another thread?
```

If yes, think carefully about synchronization.

If no, a barrier may only slow the kernel down.

## The Dangerous Barrier: Divergence

All threads in a block must reach the same `__syncthreads()` call.

This is dangerous:

```cpp
if (tid < 16) {
    __syncthreads();
}
```

Only some threads enter the `if`.

Those threads wait for the rest of the block.

But the rest of the block never arrives at that barrier.

That can hang the block.

The safer shape is:

```cpp
if (tid < 16) {
    scratch[tid] = x[tid];
}

__syncthreads();
```

Now every thread reaches the barrier.

Only some threads did the work before it.

## Barrier Placement

Barrier placement is about dependencies.

Use a barrier after writes that other threads will read:

```cpp
scratch[tid] = value;
__syncthreads();
```

Use a barrier between stages when stage 2 reads values written by stage 1:

```cpp
scratch[tid] += scratch[tid + stride];
__syncthreads();
```

Do not place barriers inside branches that only some block threads enter.

Do not add barriers just to "be safe" without knowing what data dependency they
protect.

Correctness first.

Then remove unnecessary waiting when you can explain why it is unnecessary.

## The Core Pattern

When reading a kernel with synchronization, ask:

```text
What data is shared?
Which thread writes it?
Which thread reads it?
What must be true before the read?
Does every thread in the block reach the barrier?
Is the barrier block-local enough for this dependency?
```

Those questions turn synchronization from magic into engineering.

## Bridge To Week 14

Barriers make threads wait at a known point.

Atomics solve a different coordination problem:

```text
many threads want to update the same memory location
```

Week 14 teaches what happens when the shared state is not a scratchpad stage,
but a counter or accumulator that many threads touch.
