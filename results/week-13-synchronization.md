# Week 13

Status: writing template

## What To Capture

- one tiny barrier or wait-for-ready example
- one race condition before the wait
- one sentence about why this is correctness work, not math work

## What Was Built

Describe the smallest coordination example you wrote or studied. Name the shared
value, the waiting condition, and who is allowed to proceed first.

## Correctness Check

Record exactly what becomes safe after the barrier. If you found a race, write
the failure mode in one sentence and how the barrier removes it.

## Benchmark Or Observation

If you ran anything, note whether the wait changed behavior, exposed a bug, or
added overhead. If you did not benchmark, say what you would compare next.

## Lesson Learned

Summarize synchronization in plain language.

## Limitation Or Next Step

Write one line about where waiting becomes awkward or expensive.

## Write-Back Prompts

1. What did the worker wait for?
2. What broke before the wait?
3. Why is a race condition a correctness problem?
