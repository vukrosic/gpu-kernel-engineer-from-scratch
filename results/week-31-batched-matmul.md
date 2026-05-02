# Week 31

Status: writing prompt

## What Was Built

- Describe the batched matmul sketch you wrote.
- State where the batch dimension sits in your mental model.
- Mention one example of a batch size you used to reason about the code.

## Correctness Check

- Explain why looping across the batch preserves the matmul result.
- Note one indexing detail that changes when the batch dimension appears.
- State one shape pair you would test first if you could run code.

## Benchmark Or Observation

- Record the command you ran and the batch shape you used.
- Note whether batching changed latency, throughput, or neither in your story.
- If you did not measure, write the next comparison you would want.

## Lesson Learned

- Finish this sentence: "Batching matters because ..."
- Capture one inference-workload connection in plain language.

## Limitation Or Next Step

- Name one layout issue that still needs a second look.
- Write the one point Month 8 checkpoint should emphasize.
