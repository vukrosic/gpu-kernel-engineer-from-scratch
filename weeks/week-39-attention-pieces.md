# Week 39: Attention Pieces

## What This Week Is

You break attention into the pieces you actually need to reason about. The goal
is to see query, key, value, scores, mask, softmax, and weighted sum as
separate steps before you try to fuse anything.

## What To Read

- [../course/month-10-transformer-kernels.md](../course/month-10-transformer-kernels.md)
- [week-38-rmsnorm.md](week-38-rmsnorm.md)

## Exact Commands

```bash
pytest
python examples/reference_bench.py
```

## Build This

Write one-query attention as a reference walkthrough. Keep each intermediate
named so you can point to the score matrix, the mask, and the final mix without
guessing where the work happens.

## Code Sketch

```python
import math


def softmax(xs):
    peak = max(xs)
    exps = [math.exp(x - peak) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def attention_one_query(q, keys, values, mask=None):
    scores = [
        sum(qi * ki for qi, ki in zip(q, key)) / math.sqrt(len(q))
        for key in keys
    ]
    if mask is not None:
        scores = [s if keep else float("-inf") for s, keep in zip(scores, mask)]
    weights = softmax(scores)
    return [
        sum(w * value[i] for w, value in zip(weights, values))
        for i in range(len(values[0]))
    ]
```

The sketch is correct because it follows the attention recipe in order: score,
mask, normalize, then mix the values.

Write `results/week-39-attention-pieces.md` with a labeled attention diagram
and one note about where memory pressure comes from.

## Write Down

- What are the attention pieces?
- Which intermediate is easiest to name?
- Which part is most expensive to keep around?

## Minimum

- one attention diagram
- one note file
- one plain-language summary

## Standard

- compare masked and unmasked attention
- note one memory issue

## Stretch

- sketch a row-wise attention path
- explain one reason attention is hard to optimize

## If You Are Behind

Keep the diagram and one summary sentence.

## Next Week

You will pause for a checkpoint and package the transformer-kernel month.
