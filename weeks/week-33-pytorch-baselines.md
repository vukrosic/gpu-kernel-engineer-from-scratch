# Week 33: PyTorch Baselines

## What This Week Is

You learn to trust a clean reference before you touch a custom op. The point
is to know what correct looks like and how to compare against it.

## What To Read

- [../course/month-09-pytorch-integration.md](../course/month-09-pytorch-integration.md)
- [week-template.md](week-template.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-33-pytorch-baselines.md` with one baseline table and one
note about why references make debugging easier.

## Write Down

- What makes a baseline trustworthy?
- What should be compared against it?
- What is the simplest correctness check?

## Minimum

- one baseline note
- one comparison table
- one short explanation

## Standard

- compare two baseline cases
- note one bug the baseline could catch

## Stretch

- sketch a tiny test matrix
- explain one tradeoff in testing

## If You Are Behind

Keep the baseline small and readable.

## Next Week

You will wrap a custom op so the baseline can meet your code at the boundary.
