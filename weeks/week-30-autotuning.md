# Week 30: Autotuning

## What This Week Is

You learn why one matmul kernel is rarely the final answer. This week is about
trying more than one configuration and recording the result clearly.

## What To Read

- [../course/month-08-triton-matmul-and-tuning.md](../course/month-08-triton-matmul-and-tuning.md)
- [week-29-triton-matmul.md](week-29-triton-matmul.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-30-autotuning.md` with two or three candidate settings and
one note about which one would likely win.

## Write Down

- What changes when you tune?
- What costs make tuning worth it?
- How would you compare candidates fairly?

## Minimum

- one tuning note
- one candidate table
- one plain-language summary

## Standard

- compare at least two settings
- note one reason a setting might fail

## Stretch

- propose a tiny search strategy
- explain why tuning is workload-specific

## If You Are Behind

Keep the candidate list very small.

## Next Week

You will extend the same kernel thinking to batched matmul.
