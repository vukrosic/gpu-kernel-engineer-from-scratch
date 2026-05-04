# Week 08: Reading Performance Results

Week 05 taught:

```text
bytes moved
```

Week 06 taught:

```text
memory access pattern
```

Week 07 taught:

```text
trustworthy timing
```

Week 08 teaches the next skill:

```text
how to read performance results and explain what they mean
```

A benchmark number by itself is not a conclusion. It is evidence. You still have
to connect the number to bytes moved, access pattern, timing method, and the
actual question being asked.

## Raw Timing Is Not Enough

Suppose you measure two kernels:

```text
copy: 0.08 ms
axpy: 0.12 ms
```

It is tempting to say:

```text
copy is faster than axpy
```

That may be true, but it is not the full explanation.

A better question is:

```text
how much work did each kernel do?
```

Copy moves:

```text
read x
write out
```

AXPY moves:

```text
read x
read y
write out
```

AXPY also does:

```text
one multiply
one add
```

So the timing difference is not just about the operation name. It is about memory
traffic and arithmetic.

## Use A Small Results Table

A good performance note usually starts with a table.

Example:

```text
| Kernel | Elements | Bytes Moved | Median Time | Estimated Bandwidth |
| --- | ---: | ---: | ---: | ---: |
| copy | 1,000,000 | 8 MB | 0.08 ms | 100 GB/s |
| scale | 1,000,000 | 8 MB | 0.09 ms | 88.9 GB/s |
| axpy | 1,000,000 | 12 MB | 0.12 ms | 100 GB/s |
```

This table is more useful than raw times because it shows:

```text
what was measured
how large the input was
how many bytes moved
which timing statistic was used
what bandwidth estimate came from the timing
```

Without that context, a number like `0.08 ms` is too easy to misread.

## Estimated Bandwidth

The formula from Week 05 and Week 07 is:

```text
bandwidth = bytes_moved / seconds
```

If copy moves `8 MB` and takes `0.08 ms`:

```text
8 MB / 0.08 ms = 100 GB/s
```

The exact unit conversion can be written carefully in code, but the mental model
is simple:

```text
more bytes in less time means higher bandwidth
```

Bandwidth lets you compare kernels that move different amounts of data.

Raw timing says:

```text
which one finished sooner
```

Bandwidth says:

```text
how much memory traffic was handled per second
```

Both are useful. They answer different questions.

## Compare Like With Like

A fair comparison keeps the important conditions the same.

For two kernels, check:

```text
same input size
same dtype
same device
same layout
same timing method
same warmup count
same repeat count
same correctness standard
```

If one benchmark uses `float32` and another uses `float64`, the byte count is
different.

If one input is contiguous and the other is strided, the access pattern is
different.

If one timing includes allocation and another does not, the measured work is
different.

Those comparisons can still be useful, but only if you name what changed.

## Separate Observation From Interpretation

A good performance writeup separates what happened from what you think it means.

Observation:

```text
The strided copy had a higher median time than the contiguous copy.
```

Interpretation:

```text
The strided copy likely used memory less efficiently because neighboring threads
read addresses farther apart.
```

The observation is the measured fact.

The interpretation is your explanation.

Keeping them separate makes the writeup more honest.

## Avoid Overclaiming

Do not write:

```text
This proves strided access is always slow.
```

Write:

```text
In this experiment, the strided access pattern was slower than the contiguous
pattern. That matches the expectation that nearby threads benefit from nearby
addresses.
```

The second version is better because it names the scope:

```text
this experiment
this access pattern
this expectation
```

GPU performance depends on hardware, shape, dtype, layout, kernel structure, and
timing method.

A careful writeup leaves room for that.

## Read Results Backwards

When you see a surprising result, read it backwards.

Start with the number:

```text
median time was higher than expected
```

Then ask:

```text
Was the output correct?
Was the input shape the same?
Was the dtype the same?
Was the tensor contiguous?
Was the benchmark warmed up?
Was GPU work synchronized?
Was allocation inside the timed region?
Were there outliers in the samples?
```

This is how you debug performance results.

You do not immediately invent a deep hardware explanation. You first check
whether the measurement itself is trustworthy.

## A Tiny Result Object

A benchmark result can be represented as structured data:

```python
result = {
    "name": "axpy",
    "elements": 1_000_000,
    "dtype": "float32",
    "bytes_moved": 12_000_000,
    "median_ms": 0.12,
    "bandwidth_gb_s": 100.0,
    "layout": "contiguous",
    "warmups": 10,
    "repeats": 50,
}
```

This is not just bookkeeping.

It records enough context for someone else to understand what the number means.

The worst benchmark result is a lonely number with no context:

```text
0.12 ms
```

The better result says:

```text
AXPY over 1,000,000 contiguous float32 elements moved about 12 MB and had a
median time of 0.12 ms over 50 repeats after 10 warmups.
```

That sentence is much harder to misunderstand.

## What To Put In A Performance Note

A useful performance note has five parts.

First, state the question:

```text
Does contiguous access behave differently from strided access?
```

Second, state the setup:

```text
float32 arrays, 1,000,000 elements, same timing harness, median of 50 repeats
```

Third, show the result:

```text
| Pattern | Median Time | Notes |
| --- | ---: | --- |
| contiguous | 0.08 ms | neighboring threads read neighboring addresses |
| strided | 0.18 ms | neighboring threads read farther apart |
```

Fourth, interpret the result:

```text
The strided pattern was slower in this setup, which matches the coalescing
mental model from Week 06.
```

Fifth, name one limitation:

```text
This does not prove the same ratio will hold for every shape or GPU.
```

That structure keeps the note short and useful.

## Correctness Belongs In The Report

Performance notes should mention correctness.

Before comparing speed, say how the output was checked:

```text
Both kernels were compared against the same CPU or NumPy reference.
```

Or:

```text
The output matched the reference within the expected floating-point tolerance.
```

This matters because a faster wrong result is not an optimization.

It is a broken implementation.

## Shape Belongs In The Report

Shape is part of the result.

These are different experiments:

```text
1,000 elements
1,000,000 elements
1024 x 1024 matrix
2048 x 2048 matrix
NCHW tensor
transposed matrix view
```

A result without shape is incomplete.

Always include the shape or number of elements when reporting performance.

## Layout Belongs In The Report

After Week 06, layout should also be visible.

At minimum, say whether the input was:

```text
contiguous
strided
transposed
unknown
```

If the layout is unknown, the result is harder to interpret.

Performance work is partly detective work. Layout is one of the first clues.

## Timing Method Belongs In The Report

After Week 07, timing method should be visible too.

Mention:

```text
warmups
repeats
median or mean
synchronization if GPU timing was involved
whether allocation was inside the timed region
```

You do not need a long paragraph every time.

A short note is enough:

```text
Median of 50 repeats after 10 warmups; GPU synchronized before reading times.
```

That one sentence makes the result more trustworthy.

## Preparing For Reductions

Week 09 starts reductions.

Reductions change the shape of the problem.

Elementwise kernels usually follow:

```text
one input position -> one output position
```

Reductions follow:

```text
many input positions -> fewer output positions
```

Example:

```text
[1, 2, 3, 4] -> 10
```

That means reductions introduce a new question:

```text
how do multiple threads cooperate to produce one result?
```

The benchmarking discipline from Weeks 05-08 still matters. But the kernel
structure becomes less direct than elementwise copy, add, or AXPY.

## The Mental Model

When reading a performance result, ask:

```text
What was measured?
Was the output correct?
What shape and dtype were used?
How many bytes moved?
Was the memory layout contiguous or strided?
How was timing done?
What changed between the compared runs?
What conclusion is supported by the data?
What conclusion would be overclaiming?
```

The real lesson of Week 08 is:

```text
Performance numbers need context before they become evidence.
```

A good GPU engineer does not just make kernels faster.

A good GPU engineer explains what was measured, why the result is believable,
and what the result does and does not prove.
