# Week 38: RMSNorm

## What This Week Is

You learn the shape of RMSNorm and why it shows up in modern transformer
families. The main goal is to understand the normalization steps clearly.

## What To Read

- [../course/month-10-transformer-kernels.md](../course/month-10-transformer-kernels.md)
- [week-37-gelu-fusion.md](week-37-gelu-fusion.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-38-rmsnorm.md` with the RMSNorm steps and one note about
why it differs from LayerNorm.

## Write Down

- What does RMSNorm normalize?
- How is it simpler than LayerNorm?
- Why does it matter in transformer blocks?

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

You will break attention into smaller pieces so the final kernel feels less mysterious.
