# Week 38: RMSNorm

## What This Week Is

You learn the shape of RMSNorm and why modern transformers like it. The point
is to see how one reduction and one scale can replace a more complicated
normalization path.

## What To Read

- [../course/month-10-transformer-kernels.md](../course/month-10-transformer-kernels.md)
- [week-37-gelu-fusion.md](week-37-gelu-fusion.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write RMSNorm from scratch on a tiny vector. Compare it to a reference
LayerNorm-style note so you can name the shared pieces and the pieces RMSNorm
leaves out.

## Code Sketch

```python
import math


def rmsnorm(xs, gamma, eps=1e-6):
    mean_square = sum(x * x for x in xs) / len(xs)
    scale = 1.0 / math.sqrt(mean_square + eps)
    return [x * scale * g for x, g in zip(xs, gamma)]
```

The sketch is correct because it normalizes by root mean square, then applies
the learned scale, which is the core of RMSNorm.

Write `results/week-38-rmsnorm.md` with the RMSNorm steps, the epsilon term,
and one note about why it differs from LayerNorm.

## Write Down

- What does RMSNorm measure?
- Which subtraction from LayerNorm disappears?
- Why does a simpler normalization path matter in a transformer block?

## Minimum

- one RMSNorm note
- one formula sketch
- one plain-language summary

## Standard

- compare RMSNorm and LayerNorm
- note one implementation simplification

## Stretch

- sketch a fused RMSNorm pass
- explain one stability concern

## If You Are Behind

Keep the comparison to LayerNorm very short.

## Next Week

You will break attention into smaller pieces so the final kernel feels less
mysterious.
