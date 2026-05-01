# Week 29: Triton Matmul

## What This Week Is

You map matrix multiplication onto Triton blocks. The goal is to see how the
same math from Month 6 becomes a kernel-shaped implementation strategy.

## What To Read

- [../course/month-08-triton-matmul-and-tuning.md](../course/month-08-triton-matmul-and-tuning.md)
- [week-24-month-06-checkpoint.md](week-24-month-06-checkpoint.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

## Code Sketch

```python
# Sketch the smallest working version of this week's idea.
# Keep it tiny: one loop, one mask, one tile, or one benchmark.
```

Write one sentence explaining why the sketch is correct before you optimize it.

Write `results/week-29-triton-matmul.md` with a tile sketch and one note on
how Triton expresses the same idea as the earlier matmul weeks.

## Write Down

- What is shared between CUDA and Triton matmul thinking?
- What changes in the code shape?
- What should be benchmarked later?

## Minimum

- one Triton matmul sketch
- one note file
- one plain-language summary

## Standard

- compare two tile choices
- note one reuse or masking detail

## Stretch

- sketch a batched matmul variant
- explain one performance tradeoff

## If You Are Behind

Keep the sketch small and the note short.

## Next Week

You will learn how tuning changes the shape of a Triton kernel search.
