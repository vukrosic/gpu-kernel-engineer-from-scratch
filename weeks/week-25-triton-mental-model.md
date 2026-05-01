# Week 25: Triton Mental Model

## What This Week Is

You learn how Triton thinks about programs, blocks, and masks. This week is
about reading Triton code without panic and recognizing the shape of a kernel.

## What To Read

- [../course/month-07-triton-for-ai-kernels.md](../course/month-07-triton-for-ai-kernels.md)
- [week-template.md](week-template.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-25-triton-mental-model.md` with a block diagram and three
questions about Triton's execution model.

## Write Down

- What is a program in Triton?
- What is a block?
- Why do masks matter?

## Minimum

- one Triton vocabulary note
- one block diagram
- one plain-language summary

## Standard

- compare Triton blocks to CUDA blocks
- explain one mask example

## Stretch

- sketch a tiny elementwise kernel in pseudocode
- mention one reason Triton is readable

## If You Are Behind

Focus on the vocabulary and one diagram.

## Next Week

You will learn how Triton uses blocks and masks to handle edges safely.
