# Week 07: Timing Harness And Benchmarking

Week 05 taught:

```text
how many bytes move
```

Week 06 taught:

```text
how those bytes are accessed
```

Week 07 teaches:

```text
how to measure runtime without lying to yourself
```

Benchmarking is not just calling a timer around some code. GPU work has setup
costs, warmup effects, asynchronous execution, and noisy measurements.

The goal of a timing harness is to make measurements boring and repeatable.

## The Problem With One Timing Number

A single timing result can be misleading.

Suppose you time the same operation five times:

```text
0.81 ms
0.79 ms
1.42 ms
0.80 ms
0.78 ms
```

Which number is the truth?

The `1.42 ms` run may include extra noise:

```text
first-time setup
background system activity
memory allocation
cache effects
Python overhead
GPU scheduling noise
```

If you report only one number, you may accidentally report noise instead of the
kernel behavior.

That is why benchmark code repeats measurements.

## What A Timer Measures

In Python, a simple CPU timer looks like this:

```python
import time

start = time.perf_counter()
result = work()
end = time.perf_counter()

elapsed = end - start
```

This measures wall-clock time on the CPU side.

That is fine for normal Python work.

GPU work is trickier because GPU operations can be asynchronous.

## Asynchronous GPU Work

Asynchronous means:

```text
the CPU can launch GPU work and continue before the GPU has finished
```

The CPU says:

```text
GPU, start this kernel.
```

Then the CPU may move on.

If you stop the CPU timer immediately, you may measure only the launch, not the
actual GPU work.

That is why GPU timing often needs synchronization.

## Synchronization

Synchronization means:

```text
wait until the GPU has finished the queued work
```

In PyTorch, the idea looks like:

```python
torch.cuda.synchronize()
start = time.perf_counter()

work()

torch.cuda.synchronize()
end = time.perf_counter()
```

The first synchronization clears earlier queued work.

The second synchronization waits for this measured work to finish.

Without synchronization, the measured time can be too small.

That creates fake speed.

## CUDA Events

CUDA also has GPU-side timing events.

The shape looks like:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
work()
end.record()

torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

The event records happen on the GPU timeline.

The final synchronization waits until the event has completed.

CUDA events are often better for timing GPU kernels because they measure GPU
elapsed time more directly than a CPU wall-clock timer.

The beginner rule is:

```text
CPU timer needs synchronization around GPU work.
CUDA events still need synchronization before reading the result.
```

## Warmup

The first few runs are often slower or different.

Warmup runs help avoid measuring setup effects.

A warmup loop looks like:

```python
for _ in range(10):
    work()
```

For GPU work, warmup may trigger:

```text
kernel compilation
memory pool setup
cache effects
library initialization
autotuning
```

You usually do not want these one-time costs inside the final timing number.

Warmup does not make the benchmark dishonest.

It makes the benchmark measure the steady-state operation more clearly.

## Repeats

After warmup, run the measured operation multiple times.

```python
samples = []

for _ in range(20):
    start = time.perf_counter()
    work()
    end = time.perf_counter()
    samples.append(end - start)
```

The point of repeats is not to make the code faster.

The point is to see variation.

If the samples are:

```text
0.80 ms, 0.81 ms, 0.79 ms, 0.82 ms
```

The measurement is stable.

If the samples are:

```text
0.80 ms, 1.70 ms, 0.76 ms, 3.20 ms
```

The timing environment is noisy, or the benchmark includes something unstable.

## Median Vs Mean

The mean is the average:

```text
sum(samples) / number_of_samples
```

The median is the middle value after sorting:

```text
sorted(samples)[middle]
```

Example:

```text
samples = [0.78, 0.79, 0.80, 0.81, 1.42]
```

Mean:

```text
(0.78 + 0.79 + 0.80 + 0.81 + 1.42) / 5 = 0.92 ms
```

Median:

```text
0.80 ms
```

The median is less affected by one slow outlier.

For small benchmark reports, median is often a good default.

Mean is still useful, especially when paired with standard deviation or
percentiles, but median is harder for one noisy sample to ruin.

## A Small CPU Timing Harness

This is the shape of a simple timing helper:

```python
import statistics
import time


def benchmark(work, warmups=5, repeats=20):
    for _ in range(warmups):
        work()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        work()
        end = time.perf_counter()
        samples.append(end - start)

    return {
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
    }
```

This helper does four useful things:

```text
runs warmups
collects repeated samples
reports more than one statistic
keeps the raw samples visible
```

The raw samples matter because they show whether the benchmark was stable.

## A GPU-Aware Timing Shape

For GPU work, the CPU timer shape needs synchronization.

In PyTorch-shaped pseudocode:

```python
def benchmark_cuda(work, warmups=10, repeats=50):
    for _ in range(warmups):
        work()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()

        work()

        torch.cuda.synchronize()
        end = time.perf_counter()
        samples.append(end - start)

    return statistics.median(samples)
```

The important part is not the exact function.

The important part is the timing boundary:

```text
synchronize
start timer
launch work
synchronize
stop timer
```

That boundary makes sure the measured time includes the GPU work.

## What Not To Include

A benchmark should measure the operation you care about.

If you want to time a kernel, avoid measuring unrelated setup inside the timed
region.

Usually do not include:

```text
random input generation
array allocation
CPU-to-GPU copies
GPU-to-CPU copies
printing
debug checks
file writes
```

Unless those costs are intentionally part of what you want to measure.

For example, these are different questions:

```text
How fast is the kernel?
How fast is the full pipeline including data transfer?
How fast is the Python wrapper?
```

All three can be valid.

They should not be mixed accidentally.

## Allocation Noise

Memory allocation can be expensive and noisy.

This benchmark mixes allocation with math:

```python
def work():
    x = torch.empty((1_000_000,), device="cuda")
    return x * 2
```

This benchmark separates allocation from the measured work:

```python
x = torch.empty((1_000_000,), device="cuda")


def work():
    return x * 2
```

The second version is cleaner if the goal is to time the multiply.

The first version is valid only if the goal is to measure allocation plus
multiply.

## Correctness Still Comes First

A benchmark result is only meaningful if the output is correct.

The safe order is:

```text
write the reference
check correctness
then benchmark
```

Do not optimize a kernel that has not been checked.

Do not compare performance between two implementations unless both compute the
same result.

Bad correctness plus fast timing is not a win.

It is just a fast bug.

## Comparing Two Kernels

When comparing two kernels, keep the setup fair.

The input should be the same shape:

```text
same dtype
same number of elements
same layout
same device
same warmup count
same repeat count
```

If one kernel uses `float32` and another uses `float16`, the comparison is no
longer just about kernel structure.

If one tensor is contiguous and the other is strided, the benchmark also tests
layout.

That can be useful, but it should be named clearly.

## Relating Time To Bandwidth

Week 05 estimated bytes moved.

Week 07 connects that estimate to timing:

```text
bandwidth = bytes_moved / seconds
```

For a copy kernel with `1,000,000` `float32` elements:

```text
bytes moved = 8,000,000 bytes
```

If the median time is `0.00008` seconds:

```text
bandwidth = 8,000,000 / 0.00008
bandwidth = 100,000,000,000 bytes/s
bandwidth = 100 GB/s
```

This number is not useful unless the timing is trustworthy.

That is why the timing harness matters.

## Fake Optimization Wins

Bad timing can make an optimization look real when it is not.

Common fake wins:

```text
measuring async launch instead of GPU completion
including allocation in one version but not another
using one noisy sample
comparing different input sizes
forgetting warmup
accidentally changing dtype
benchmarking incorrect output
```

The timing harness is a defense against these mistakes.

It forces you to compare the thing you meant to compare.

## The Mental Model

When reading or writing a benchmark, ask:

```text
What exactly is inside the timed region?
Was the output checked for correctness first?
Were warmup runs used?
Were repeated samples collected?
Was GPU work synchronized?
Are median, mean, and raw samples available?
Are both implementations using the same input shape, dtype, layout, and device?
```

The real lesson of Week 07 is:

```text
A benchmark is a measurement instrument.
If the instrument is sloppy, the result is not evidence.
```

Week 05 taught what to count.

Week 06 taught how layout changes memory behavior.

Week 07 teaches how to trust the numbers you write down.
