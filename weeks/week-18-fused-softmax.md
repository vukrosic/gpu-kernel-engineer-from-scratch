# Week 18: Fused Softmax

## What This Week Is

The softmax math from Week 17 now becomes a performance question: can you do
more work in one pass and move less data around?

## What To Read

- [../course/month-05-softmax-and-normalization.md](../course/month-05-softmax-and-normalization.md)
- [../weeks/week-17-softmax-math.md](../weeks/week-17-softmax-math.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

## Code Sketch

```python
import math

def fused_softmax(xs):
    shift = max(xs)
    exps = []
    total = 0
    for x in xs:
        value = math.exp(x - shift)
        exps.append(value)
        total += value
    return [value / total for value in exps]
```

This sketch is correct because it fuses the exponentiation and summation into
one pass while keeping the same math as the unfused version.

Create `results/week-18-fused-softmax.md` with:

- what “fused” means
- why fewer passes can help
- what steps softmax can share in one pipeline

## Write Down

Answer:

1. What is the benefit of fusion?
2. Which softmax steps can be combined?
3. Why is this a memory story as much as a math story?

## Minimum

- the note exists
- you can explain the idea of fusion

## Standard

- you compare unfused and fused softmax conceptually
- you write one note on why stable softmax still matters

## Stretch

- you sketch a fused softmax pipeline
- you explain what you would benchmark if you had GPU code

## If You Are Behind

Keep the idea-focused explanation. You do not need a full kernel implementation
yet.

## Next Week

Week 19 adds LayerNorm, another core normalization pattern in transformer code.
