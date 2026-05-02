# Week 03 Indexing

## What Was Built

Write the indexing note that explains row-major layout, one worker per element,
and how the same worker mapping can support add, multiply, square, and ReLU.
Include the mapping table you made in plain text or markdown.

## Correctness Check

Record a tiny row-major experiment, such as `ravel()` or a reshape check, and
say how it confirmed your understanding of the layout.

## Benchmark Or Observation

Describe the shape or data layout you compared. If you did not benchmark
anything, note the observation you made about why indexing bugs are still a
performance and correctness risk.

## Lesson Learned

Summarize why indexing is not just bookkeeping. Explain, in your own words, how
layout and worker assignment have to agree for the kernel to be correct.

## Limitation Or Next Step

Name the indexing case you still want to practice, such as a 2D grid or a batch
dimension, and say what Week 04 should preserve.

## Write Down

- Why is indexing a correctness problem, not just a performance problem?
- What breaks if two workers write to the same output element?
- How does a batch dimension change the way you think about a tensor?
- What simple rule helps you avoid row-major mistakes?
