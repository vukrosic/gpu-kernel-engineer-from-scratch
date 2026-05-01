# Week 37: GELU Fusion

## What This Week Is

You look at GELU as an activation that often gets fused with nearby work. The
goal is to see why one extra pass over data can matter.

## What To Read

- [../course/month-10-transformer-kernels.md](../course/month-10-transformer-kernels.md)
- [week-template.md](week-template.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-37-gelu-fusion.md` with the GELU formula and one note
about why fusion is attractive.

## Write Down

- What does GELU do?
- Why would you fuse it?
- Where does the extra cost come from?

## Minimum

- one GELU note
- one simple formula
- one short explanation

## Standard

- compare fused and unfused thinking
- note one bandwidth benefit

## Stretch

- sketch a fused activation block
- explain one transformer use case

## If You Are Behind

Keep the formula and one fusion note.

## Next Week

You will study RMSNorm and why normalization is often paired with transformer blocks.
