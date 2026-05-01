# Week 17: Softmax Math

## What This Week Is

You are starting Month 5, and the goal is to understand the math of softmax
before talking about fused kernels or performance.

## What To Read

- [../course/month-05-softmax-and-normalization.md](../course/month-05-softmax-and-normalization.md)
- [../weeks/week-16-month-04-checkpoint.md](../weeks/week-16-month-04-checkpoint.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

## Code Sketch

```python
# Sketch the smallest working version of this week's idea.
# Keep it tiny: one loop, one mask, one tile, or one benchmark.
```

Write one sentence explaining why the sketch is correct before you optimize it.

Create `results/week-17-softmax-math.md` and include:

- a tiny hand-worked softmax example
- why subtracting the max helps numerical stability
- why softmax output sums to 1

## Write Down

Answer:

1. What does softmax turn scores into?
2. Why do we subtract the max?
3. Why is softmax common in classification and transformers?

## Minimum

- the note exists
- you can explain softmax with a tiny example

## Standard

- you compare stable and unstable softmax behavior
- you write one sentence about probabilities

## Stretch

- you sketch the softmax steps as a pipeline
- you connect softmax to the memory and reduction lessons

## If You Are Behind

Focus on the math story. Do not worry about fused implementation yet.

## Next Week

Week 18 turns the math into a fused version and starts caring about memory
traffic again.
