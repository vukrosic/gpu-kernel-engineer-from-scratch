# Week 31: Batched Matmul

## What This Week Is

You take the matmul pattern and repeat it across a batch dimension. The goal
is to see how one good kernel idea scales across many small problems.

## What To Read

- [../course/month-08-triton-matmul-and-tuning.md](../course/month-08-triton-matmul-and-tuning.md)
- [week-30-autotuning.md](week-30-autotuning.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-31-batched-matmul.md` with one batch sketch and one note
about whether batching changes the bottleneck.

## Write Down

- What stays the same across the batch?
- What changes for indexing?
- What gets easier to reuse?

## Minimum

- one batch sketch
- one note file
- one short summary

## Standard

- compare two batch sizes
- note one indexing detail

## Stretch

- connect batching to inference workloads
- explain one memory-layout issue

## If You Are Behind

Keep the batch example tiny.

## Next Week

You will package Month 8 and decide what belongs in a portfolio note.
