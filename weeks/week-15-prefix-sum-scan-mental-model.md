# Week 15: Prefix Sum And Scan Mental Model

Reductions shrink data:

```text
[1, 2, 3, 4] -> 10
```

Scan keeps the same number of outputs:

```text
[1, 2, 3, 4] -> [1, 3, 6, 10]
```

Week 15 teaches the mental model.

Week 16 will teach the parallel implementation shape.

## Inclusive Scan

An inclusive scan includes the current value.

For:

```text
[3, 1, 4, 2]
```

The inclusive sum scan is:

```text
[3, 4, 8, 10]
```

Read each output as:

```text
out[0] = x[0]
out[1] = x[0] + x[1]
out[2] = x[0] + x[1] + x[2]
out[3] = x[0] + x[1] + x[2] + x[3]
```

The current element is included in its own output.

Python version:

```python
def inclusive_scan(values):
    out = []
    running = 0

    for value in values:
        running += value
        out.append(running)

    return out
```

## Exclusive Scan

An exclusive scan excludes the current value.

For:

```text
[3, 1, 4, 2]
```

The exclusive sum scan is:

```text
[0, 3, 4, 8]
```

Read each output as:

```text
out[0] = 0
out[1] = x[0]
out[2] = x[0] + x[1]
out[3] = x[0] + x[1] + x[2]
```

Python version:

```python
def exclusive_scan(values):
    out = []
    running = 0

    for value in values:
        out.append(running)
        running += value

    return out
```

Inclusive scan answers:

```text
how much have I seen through this position?
```

Exclusive scan answers:

```text
how much came before this position?
```

That second question is extremely useful for building offsets.

## Scan Vs Reduction

Reduction:

```text
input length:  4
output length: 1
```

Scan:

```text
input length:  4
output length: 4
```

Reduction tells you the total.

Scan tells you every prefix total.

For the same input:

```text
[3, 1, 4, 2]
```

Reduction:

```text
10
```

Inclusive scan:

```text
[3, 4, 8, 10]
```

The last element of an inclusive sum scan is the reduction result.

That is a useful connection:

```text
scan is like keeping all the intermediate reduction totals
```

## Why GPU Engineers Care About Scan

Scan appears whenever threads need positions, offsets, or compacted output.

Example: keep only positive values.

Input:

```text
[5, -1, 7, 0, 3]
```

First build flags:

```text
[1, 0, 1, 0, 1]
```

Exclusive scan of flags:

```text
[0, 1, 1, 2, 2]
```

Those scan values are output positions.

So:

```text
value 5 goes to output position 0
value 7 goes to output position 1
value 3 goes to output position 2
```

The compacted output is:

```text
[5, 7, 3]
```

This is why scan matters.

It turns local yes/no decisions into global write positions.

## Scan As Data Flow

In a reduction, information flows into one value:

```text
many positions -> one result
```

In scan, information flows across positions:

```text
earlier positions -> later positions
```

For output index `i`, scan needs information from all positions before `i`.

That dependency is why scan is less obvious than elementwise kernels.

Elementwise:

```text
out[i] depends only on x[i]
```

Scan:

```text
out[i] depends on x[0] through x[i]
```

The trick is to compute those prefixes in parallel stages instead of one serial
loop.

## A Stage View

For inclusive scan:

```text
[1, 2, 3, 4, 5, 6, 7, 8]
```

One way to think about the stages is:

```text
distance 1: each position adds the value 1 step behind it
distance 2: each position adds the value 2 steps behind it
distance 4: each position adds the value 4 steps behind it
```

The distances double:

```text
1, 2, 4, 8, ...
```

That should feel familiar.

Reductions shrink by powers of two.

Scans spread information by powers of two.

## Where Inclusive And Exclusive Differ

Inclusive scan is natural when each output should include the current element:

```text
running totals
cumulative probabilities
prefix sums for reporting
```

Exclusive scan is natural when each output should be a write offset:

```text
stream compaction
placing variable-length outputs
building row offsets
allocating slots from flags
```

For GPU kernels, exclusive scan often feels more "systems-y" because it answers:

```text
where should this thread write?
```

## The Core Pattern

When you see scan, ask:

```text
Is it inclusive or exclusive?
What operation is being scanned?
What does each output position mean?
Is the scan being used as values or as offsets?
Does the final element also give a total?
```

Scan is not just cumulative sum.

It is a way to make local parallel decisions produce organized global output.

## Bridge To Week 16

Week 15 taught what scan means.

Week 16 teaches how a block of threads can build one scan result in shared
memory.
