# Week 43: KV Cache

## What This Week Is

You study KV cache as the mechanism that makes repeated decoding cheaper. The
point is to understand what gets stored and why that helps inference.

## What To Read

- [../course/month-11-attention-and-inference.md](../course/month-11-attention-and-inference.md)
- [week-42-flashattention-concepts.md](week-42-flashattention-concepts.md)

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

Write `results/week-43-kv-cache.md` with one cache diagram and one note about
what is reused during decoding.

## Write Down

- What does KV cache store?
- What does it save?
- What is the tradeoff?

## Minimum

- one cache diagram
- one note file
- one short explanation

## Standard

- compare cached and uncached decoding
- note one memory tradeoff

## Stretch

- sketch a cache growth example
- explain one latency benefit

## If You Are Behind

Keep the cache idea simple.

## Next Week

You will pause for a checkpoint and turn the attention month into a clean summary.
