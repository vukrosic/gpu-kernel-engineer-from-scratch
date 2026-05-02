# Week 10 Shared Memory Reductions

## What Was Built

Write the note that explains the block-level reduction idea, the role of
partial sums, and the final combine step. Mention how the block shape differs
from the naive row-by-row reduction you wrote in Week 09.

## Correctness Check

Record the partials you created in the scratch experiment and how you confirmed
that they summed to the final answer.

## Benchmark Or Observation

Describe the difference you would expect between naive reduction and
block-style reduction, especially around repeated global memory traffic.

## Lesson Learned

Summarize why shared work helps and why a smaller final combine step is easier
to reason about than doing everything in one long chain.

## Limitation Or Next Step

Write one sentence about what you still want to understand before warp-level
thinking feels natural.

## Write Down

- Why do partial sums help?
- What work happens inside a block?
- Why is shared coordination better than every worker doing everything alone?
- How does this set you up for warp-level thinking later?
