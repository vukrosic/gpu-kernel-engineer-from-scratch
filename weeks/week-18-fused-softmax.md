# Week 18: Fused Softmax

## What This Week Is

The softmax math from Week 17 now becomes a performance question: can you do
more work in one pass and move less data around? This is the first time the
course pushes you to ask what can be shared inside a pipeline instead of what
just needs to be computed.

## What To Read

- [../course/month-05-softmax-and-normalization.md](../course/month-05-softmax-and-normalization.md)
- [../weeks/week-17-softmax-math.md](../weeks/week-17-softmax-math.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-18-fused-softmax.md` with a comparison between unfused and
fused softmax, plus one sentence about why fewer passes can help.

## Code Sketch

```python
import math

def fused_softmax(xs):
    shift = max(xs)
    exps = []
    total = 0.0
    for x in xs:
        value = math.exp(x - shift)
        exps.append(value)
        total += value
    return [value / total for value in exps]
```

This sketch is correct because it keeps the same stable math as the unfused
version while showing how the intermediate values can stay in one pipeline.

## Write Down

Answer:

1. What is the benefit of fusion?
2. Which softmax steps can be combined?
3. Why is this a memory story as much as a math story?
4. What intermediate value do you want to keep around instead of recomputing?

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
