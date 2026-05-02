# Week 37: GELU Fusion

## What This Week Is

You treat GELU as the small activation that often sits right next to a bias add
or projection output. The point is to see that the math is simple, but the
extra write between steps is not free.

## What To Read

- [../course/month-10-transformer-kernels.md](../course/month-10-transformer-kernels.md)
- [week-36-month-09-checkpoint.md](week-36-month-09-checkpoint.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write a tiny fused bias-plus-GELU path. Start with a reference version that
adds bias, then applies GELU, then write the same logic as one pass over the
inputs so you can compare the two shapes directly.

## Code Sketch

```python
import math


def gelu(x):
    return 0.5 * x * (
        1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x))
    )


def fused_bias_gelu(xs, bias):
    out = []
    for x, b in zip(xs, bias):
        y = x + b
        out.append(gelu(y))
    return out
```

The fused version is correct because it computes the same value as the two-step
path, just without storing the intermediate result in a separate buffer.

Write `results/week-37-gelu-fusion.md` with the GELU formula, the fused-vs-
unfused comparison, and one note about memory traffic.

## Write Down

- What does GELU approximate?
- What disappears when you fuse the bias add with the activation?
- Why does one fewer memory pass matter?

## Minimum

- one GELU formula
- one fused-vs-unfused note
- one plain-language correctness sentence

## Standard

- compare exact and approximate GELU
- note one bandwidth benefit
- mention one transformer block where GELU appears

## Stretch

- sketch a fused bias, residual, and GELU block
- explain why this is a good first fusion target

## If You Are Behind

Keep the formula and the one-pass explanation.

## Next Week

You will study RMSNorm and see how a normalization kernel can be written as
another tight pass over data.
