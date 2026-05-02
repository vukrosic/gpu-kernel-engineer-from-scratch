# Week 19

Status: writing template

## What To Capture

- one LayerNorm row example
- which parts are reduction-like
- which parts are elementwise
- why normalization helps training

## Row Sketch

| Input | Mean | Variance | Normalized Output |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

## What Was Built

Describe the LayerNorm example you wrote or studied. Name the row shape and the
statistics you computed.

## Correctness Check

Record why the mean and variance are shared across the row. If you compared a
reference version and a starter kernel, note what had to match.

## Benchmark Or Observation

If you measured anything, note whether the row size affected the work. If you
did not measure, write the comparison you would make.

## Lesson Learned

Summarize LayerNorm in plain language.

## Limitation Or Next Step

Write one sentence about what still needs to be true for a GPU version.

## Write-Back Prompts

1. What does LayerNorm normalize?
2. Why is it both reduction-like and elementwise?
3. Why does normalization help model training?
