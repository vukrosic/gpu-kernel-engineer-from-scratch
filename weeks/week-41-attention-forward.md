# Week 41: Attention Forward Pass

## What This Week Is

You connect the attention pieces into one forward pass story. The goal is to
understand the full path from inputs to output probabilities or scores.

## What To Read

- [../course/month-11-attention-and-inference.md](../course/month-11-attention-and-inference.md)
- [week-39-attention-pieces.md](week-39-attention-pieces.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write `results/week-41-attention-forward.md` with the full forward-pass flow
and one note about where the math is easiest to check.

## Write Down

- What is the full forward path?
- Where can errors hide?
- What should be checked first?

## Minimum

- one forward-pass sketch
- one note file
- one short explanation

## Standard

- compare two sequence lengths
- note one correctness check

## Stretch

- sketch a causal mask
- explain one inference use case

## If You Are Behind

Keep the forward pass to one simple diagram.

## Next Week

You will learn the core ideas behind FlashAttention-style thinking.
