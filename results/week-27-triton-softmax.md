# Week 27

Status: writing prompt

## What Was Built

- Describe the row-wise softmax sketch you wrote.
- List the three steps in order and note where you would fuse them.
- Mention the input shape or row length you used for the example.

## Correctness Check

- Explain why subtracting the max keeps the row numerically stable.
- Say how you verified that the probabilities still sum to one.
- Note one edge case the reference implementation protects against.

## Benchmark Or Observation

- Record the command you ran and the shape or batch you used.
- Note whether the important observation was stability, reuse, or shape.
- If you did not measure, write the next comparison you would run.

## Lesson Learned

- Finish this sentence: "Softmax becomes easier to reason about when ..."
- Write one sentence about why the stable version is the one to keep.

## Limitation Or Next Step

- Name the part of softmax that still feels like a bottleneck.
- Write the next question Month 8 should answer about matmul-shaped work.
