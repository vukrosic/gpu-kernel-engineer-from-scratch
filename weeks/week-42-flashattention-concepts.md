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

## Code Sketch

```python
# Sketch the smallest working version of this week's idea.
# Keep it tiny: one loop, one mask, one tile, or one benchmark.
```

Write one sentence explaining why the sketch is correct before you optimize it.

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
