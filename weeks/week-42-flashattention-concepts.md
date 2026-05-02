# Week 42: FlashAttention Concepts

## What This Week Is

You learn why attention can be rethought around memory savings and tiles. The
goal is to understand the idea before worrying about a full implementation.

## What To Read

- [../course/month-11-attention-and-inference.md](../course/month-11-attention-and-inference.md)
- [week-41-attention-forward.md](week-41-attention-forward.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Draw a tiled attention flow and explain the online-softmax trick in words. The
goal is to show how you can process chunks of K and V without materializing the
whole score matrix.

## Code Sketch

```python
import math


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def tiled_attention_step(q, k_tiles, v_tiles):
    running_max = float("-inf")
    running_sum = 0.0
    running_out = None

    for k_tile, v_tile in zip(k_tiles, v_tiles):
        scores = [dot(q, key) for key in k_tile]
        block_max = max(scores)
        new_max = max(running_max, block_max)
        exp_scores = [math.exp(s - new_max) for s in scores]
        if running_out is None:
            running_out = [0.0 for _ in v_tile[0]]
        running_sum = running_sum * math.exp(running_max - new_max) + sum(exp_scores)
        # The output accumulator would be rescaled and updated here.
        running_max = new_max
    return running_out
```

The sketch is correct as a concept because it keeps the softmax state up to
date across tiles instead of storing every score at once.

Write `results/week-42-flashattention-concepts.md` with one memory diagram and
one note about why the idea is important.

## Write Down

- What problem does FlashAttention solve?
- What changes about memory use?
- Why does tiling help?

## Minimum

- one concept note
- one memory sketch
- one plain-language summary

## Standard

- compare normal attention and FlashAttention thinking
- note one savings idea

## Stretch

- sketch a tiled attention flow
- explain one implementation challenge

## If You Are Behind

Keep the memory idea and one summary sentence.

## Next Week

You will learn how KV cache changes inference-time attention behavior.
