# Week 39: Attention Pieces

## What This Week Is

You split attention into the parts you actually need to reason about. The goal
is to see the query, key, value, score, mask, and softmax pieces separately.

## What To Read

- [../course/month-10-transformer-kernels.md](../course/month-10-transformer-kernels.md)
- [week-38-rmsnorm.md](week-38-rmsnorm.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-39-attention-pieces.md` with a labeled attention diagram
and one note about where memory pressure comes from.

## Write Down

- What are the attention pieces?
- Which part is expensive?
- Which part can be fused later?

## Minimum

- one attention diagram
- one note file
- one plain-language summary

## Standard

- compare masked and unmasked attention
- note one memory issue

## Stretch

- sketch a row-wise attention path
- explain one reason attention is hard to optimize

## If You Are Behind

Keep the diagram and one summary sentence.

## Next Week

You will pause for a checkpoint and package the transformer-kernel month.
