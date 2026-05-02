# Week 21

Status: writing template

## What To Capture

- the matrix shapes you tried
- the loop order you used
- one correctness check
- one note about what makes the naive version slow

## Shapes Tried

| A Shape | B Shape | Output Shape | What To Verify |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |

## What Was Built

Describe the baseline matmul you wrote or studied. Name the shapes, the loop
nest, and the output layout.

## Correctness Check

Record why the dot product for one output cell is enough to prove the result for
that cell. If you compared against `gputriton/reference.py`, say what matched.

## Benchmark Or Observation

If you measured anything, note one observation about the naive version. If you
did not measure, write the comparison you would make next.

## Lesson Learned

Summarize naive matmul in one or two sentences.

## Limitation Or Next Step

Write one sentence about why tiling is the obvious next move.

## Write-Back Prompts

1. How does each output cell get computed?
2. What is slow about the naive version?
3. What shape checks mattered most?
