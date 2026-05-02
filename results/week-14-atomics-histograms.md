# Week 14

Status: writing template

## What To Capture

- one counter that many workers want to update
- one histogram example with a small number of bins
- one sentence on the tradeoff between correctness and contention

## What Was Built

Describe the histogram or counting example you used. Name the bins, the input
shape, and where the shared update happens.

## Correctness Check

Record what the atomic protects. If you can, note whether a private-counter
approach and a shared-counter approach would produce the same counts.

## Benchmark Or Observation

If you measured anything, note whether skewed inputs made one bucket hot or
whether the contention changed the shape of the run. If you did not measure,
write down the comparison you would make next.

## Lesson Learned

Summarize why atomics are simple to reason about but can get expensive.

## Limitation Or Next Step

Write one line about when you would stop using atomics as the default answer.

## Write-Back Prompts

1. What exactly does the atomic protect?
2. Why is a histogram the easiest way to see the problem?
3. When would you still choose atomics?
