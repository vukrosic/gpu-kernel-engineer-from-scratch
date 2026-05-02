# Week 17: Softmax Math

## What This Week Is

You are starting Month 5, and the goal is to understand the math of softmax
before talking about fused kernels or performance. The point is to see how raw
scores become probabilities without losing numerical stability along the way.

## What To Read

- [../course/month-05-softmax-and-normalization.md](../course/month-05-softmax-and-normalization.md)
- [../weeks/week-16-month-04-checkpoint.md](../weeks/week-16-month-04-checkpoint.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-17-softmax-math.md` with a tiny hand-worked example, a
plain explanation of the max-shift trick, and one sentence about why the
outputs sum to 1.

## Code Sketch

```python
import math

def softmax(xs):
    shift = max(xs)
    exps = [math.exp(x - shift) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]
```

This sketch is correct because subtracting the max keeps the exponentials
stable while preserving the final normalized probabilities.

## Write Down

Answer:

1. What does softmax turn scores into?
2. Why do we subtract the max?
3. Why is softmax common in classification and transformers?
4. What does it mean for the outputs to sum to 1?

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
