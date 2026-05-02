# Week 18

Status: writing template

## What To Capture

- what "fused" means in your own words
- which softmax steps can share one pipeline
- one note on why fewer passes can help

## Pipeline Sketch

```text
read scores -> shift by max -> exponentiate -> accumulate sum -> normalize
```

## What Was Built

Describe the unfused and fused versions you compared. Name the steps that moved
into one pass or one pipeline.

## Correctness Check

Record why the fused version should still match the stable softmax result.

## Benchmark Or Observation

If you measured anything, note whether reducing passes changed the amount of
data movement. If you did not measure, write the comparison you would make.

## Lesson Learned

Summarize why fusion is a memory story as much as a math story.

## Limitation Or Next Step

Write one sentence about what fusion does not solve by itself.

## Write-Back Prompts

1. What is the benefit of fusion?
2. Which softmax steps can be combined?
3. What still has to stay numerically stable?
