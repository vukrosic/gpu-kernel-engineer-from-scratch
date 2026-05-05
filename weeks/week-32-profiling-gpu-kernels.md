# Week 32: Profiling GPU Kernels

By Week 32, you have seen several kernel families:

```text
elementwise
reductions
softmax
normalization
matmul
Triton kernels
```

Now the question becomes:

```text
how do you know what is slow?
```

Week 32 teaches profiling as an engineering habit.

## Timing Is Not Profiling

Timing answers:

```text
how long did this take?
```

Profiling tries to answer:

```text
why did it take that long?
```

A timing table might say:

```text
kernel A: 40 us
kernel B: 25 us
```

A profile can help explain whether the difference comes from:

```text
memory traffic
occupancy
launch overhead
uncoalesced access
too many registers
poor tile size
small problem size
```

You need both.

## What To Compare

For this course, useful comparisons include:

```text
CPU or NumPy reference
PyTorch baseline
CUDA kernel
Triton kernel
different tile sizes
different row widths
different batch sizes
```

Each comparison must name the shape.

Never report:

```text
Triton is faster
```

without saying:

```text
for which shape, dtype, hardware, and timing method
```

## Benchmark Hygiene

A fair GPU benchmark should include:

```text
warmup runs
repeat runs
synchronization
correctness check
fixed input shapes
clear dtype
median or stable summary
```

GPU work is asynchronous.

If you time without synchronization, you may only time the launch, not the work.

This is one of the easiest ways to fool yourself.

## Profiling Questions

When a kernel is slow, ask:

```text
Is the problem too small?
Is launch overhead dominating?
Is memory access coalesced?
Is global memory traffic too high?
Is occupancy too low?
Are registers spilling?
Is shared memory limiting active blocks?
Is the tile shape wrong for the matrix shape?
```

The profile does not automatically fix the kernel.

It tells you where to look.

## Reading A Timeline

A GPU timeline can show:

```text
CPU launches
GPU kernel execution
memory copies
gaps between kernels
overlap
```

If there are large gaps, the problem may not be the kernel math.

It may be launch overhead, synchronization, data transfer, or Python overhead.

This matters because ML systems performance is often end-to-end.

A fast kernel inside a slow pipeline is not enough.

## Reading Kernel Metrics

Kernel metrics can point toward bottlenecks:

```text
memory throughput
achieved occupancy
registers per thread
shared memory per block
warp stalls
tensor core use
```

Do not memorize every metric at once.

Start with the question:

```text
what bottleneck do I suspect?
```

Then look for evidence that supports or contradicts it.

## A Result Note Shape

A useful profiling note looks like:

```text
Kernel:
Implementation:
Hardware:
Shape:
Dtype:
Baseline:
Timing method:
Result:
Profiler observation:
Hypothesis:
Next change:
```

The hypothesis is important.

It turns profiling from screenshots into engineering reasoning.

## Common False Wins

Be careful with:

```text
forgetting synchronization
comparing different shapes
ignoring correctness
using one run
including data allocation in one timing but not another
benchmarking tiny inputs only
claiming speedup without a baseline
```

A benchmark should make you more honest, not more dramatic.

## The Core Pattern

When profiling, ask:

```text
What exactly was measured?
What baseline was used?
Did correctness pass?
What shape and dtype were used?
What bottleneck does the profile suggest?
What is the next smallest experiment?
```

Profiling is the loop:

```text
measure
form hypothesis
change one thing
measure again
```

## Bridge To Week 33

Week 33 starts PyTorch baselines.

That is the right next step because custom kernels do not exist in isolation.

They need to be compared against the tools ML engineers already use.
