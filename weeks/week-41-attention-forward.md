# Week 41: Attention Forward Pass

## What This Week Is

You connect the attention pieces into one forward-pass story. The goal is to
understand the full path from inputs to output and to know where the math is
easiest to check.

## What To Read

- [../course/month-11-attention-and-inference.md](../course/month-11-attention-and-inference.md)
- [week-39-attention-pieces.md](week-39-attention-pieces.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write a simplified attention forward pass for one head. Compare it against a
reference path on a short sequence and a slightly longer sequence so you can
spot shape or masking mistakes early.

## Code Sketch

```python
import math


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(xs):
    peak = max(xs)
    exps = [math.exp(x - peak) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def apply_mask(scores, row_mask):
    return [s if keep else float("-inf") for s, keep in zip(scores, row_mask)]


def attention_forward(q, k, v, masks=None):
    d = len(q[0])
    out = []
    for row, qi in enumerate(q):
        scores = [dot(qi, kj) / math.sqrt(d) for kj in k]
        if masks is not None:
            scores = apply_mask(scores, masks[row])
        weights = softmax(scores)
        out.append([
            sum(w * val[i] for w, val in zip(weights, v))
            for i in range(len(v[0]))
        ])
    return out
```

The sketch is correct because it follows the same forward math as a full
attention block, just without batching and without extra framework plumbing.

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
