# Week 19: LayerNorm

## What This Week Is

LayerNorm is the other major normalization pattern in this part of the course.
It is useful because it mixes reduction thinking with per-element output.

## What To Read

- [../course/month-05-softmax-and-normalization.md](../course/month-05-softmax-and-normalization.md)
- [../weeks/week-18-fused-softmax.md](../weeks/week-18-fused-softmax.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

## Code Sketch

```python
def layernorm(xs, eps=1e-5):
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return [(x - mean) / (var + eps) ** 0.5 for x in xs]
```

This sketch is correct because it normalizes the whole row with a shared mean
and variance before returning one adjusted value per input.

Create `results/week-19-layernorm.md` with:

- what LayerNorm does
- which parts are reduction-like
- which parts are elementwise
- why it shows up in deep learning code

## Write Down

Answer:

1. What does LayerNorm normalize?
2. Why is it both reduction-like and elementwise?
3. Why does normalization help model training?

## Minimum

- the note exists
- you can explain LayerNorm in plain language

## Standard

- you compare LayerNorm to softmax
- you describe the role of mean and variance

## Stretch

- you sketch the forward-pass stages
- you explain what would need to be benchmarked in a GPU version

## If You Are Behind

Do the explanation first. The implementation details can wait until later.

## Next Week

Week 20 is the Month 5 checkpoint, where you package softmax and normalization
into one readable summary.
