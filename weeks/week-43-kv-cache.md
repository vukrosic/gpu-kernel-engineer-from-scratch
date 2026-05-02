# Week 43: KV Cache

## What This Week Is

You study KV cache as the mechanism that makes repeated decoding cheaper. The
point is to understand what gets stored, what gets reused, and what tradeoff
you pay in memory.

## What To Read

- [../course/month-11-attention-and-inference.md](../course/month-11-attention-and-inference.md)
- [week-42-flashattention-concepts.md](week-42-flashattention-concepts.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write a decode-step simulation that appends each new key and value to a cache
and reuses the full cache on the next token. Compare it to the uncached path so
the reuse is obvious.

## Code Sketch

```python
import math


def softmax(xs):
    peak = max(xs)
    exps = [math.exp(x - peak) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def decode_step(q, new_k, new_v, cache_k, cache_v):
    cache_k.append(new_k)
    cache_v.append(new_v)
    scores = [sum(qi * ki for qi, ki in zip(q, key)) / math.sqrt(len(q)) for key in cache_k]
    weights = softmax(scores)
    return [
        sum(w * value[i] for w, value in zip(weights, cache_v))
        for i in range(len(cache_v[0]))
    ]
```

The sketch is correct because it appends the new token once, then reuses all
stored keys and values for the next attention calculation.

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

You will pause for a checkpoint and turn the attention month into a clean
summary.
