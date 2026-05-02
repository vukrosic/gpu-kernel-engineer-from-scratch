# Week 01 Baseline

## What Was Built

Write a short baseline note that captures your setup, the reference benchmark
run, and your first CPU vs GPU explanation. Include the machine, Python version,
repo command, and the four reference workloads.

## Correctness Check

Record the command you trusted most for correctness, usually `pytest` plus one
or two direct reference calls. Add one sentence about why these checks matter
before any GPU optimization.

## Benchmark Or Observation

Paste the reference timings or observations for `vector_add`, `matmul`,
`softmax`, and `attention`. Add one line about what stood out and what you
would compare later.

## Lesson Learned

Summarize the mental model in your own words. Keep it to a short paragraph:
CPU and GPU solve different shapes of work, and the baseline exists to prove
the output first.

## Limitation Or Next Step

Name the one thing Week 02 should build next and one thing you do not understand
yet.

## Write Down

- Why do baselines come before kernels?
- What did the reference benchmark teach you?
- What would you compare once a real kernel exists?
- What is still fuzzy about GPU execution?
